#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ქართული ენის ფონემების მოდელის დამუშავება XTTS-v2 მოდელისთვის
ავტორი:
თარიღი: 2025-04-15
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import time
from pathlib import Path
from tqdm import tqdm
import pickle

# ლოგების კონფიგურაცია
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("phoneme_processor")


class PhonemeEncoder(nn.Module):
    """
    ფონემების ენკოდერი მოდელი
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, dropout=0.1):
        """
        ინიციალიზაცია

        Args:
            input_dim (int): შემავალი განზომილება (ფონემების/გრაფემების რაოდენობა)
            hidden_dim (int): ფარული შრის განზომილება
            output_dim (int): გამავალი განზომილება
            num_layers (int): ფარული შრეების რაოდენობა
            dropout (float): Dropout-ის ალბათობა
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # ემბედინგის შრე
        self.embedding = nn.Embedding(input_dim, hidden_dim)

        # BiLSTM შრე
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # გამავალი შრე
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_lengths=None):
        """
        წინა გასვლა

        Args:
            x (torch.Tensor): შემავალი ტენზორი [batch_size, seq_len]
            x_lengths (torch.Tensor): მიმდევრობების სიგრძეები

        Returns:
            torch.Tensor: გამავალი ტენზორი [batch_size, seq_len, output_dim]
        """
        # ემბედინგის გამოყენება
        embedded = self.dropout(self.embedding(x))  # [batch_size, seq_len, hidden_dim]

        # LSTM-ის გამოყენება
        if x_lengths is not None:
            # თუ გვაქვს სიგრძეები, პეკინგის გამოყენება
            embedded_packed = nn.utils.rnn.pack_padded_sequence(
                embedded, x_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            outputs_packed, _ = self.lstm(embedded_packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs_packed, batch_first=True)
        else:
            outputs, _ = self.lstm(embedded)

        # გამავალი შრის გამოყენება
        outputs = self.fc(self.dropout(outputs))  # [batch_size, seq_len, output_dim]

        return outputs


class GraphemeToPhoneme(nn.Module):
    """
    გრაფემა-ფონემა გარდამქმნელი მოდელი
    """

    def __init__(self, grapheme_dim, phoneme_dim, hidden_dim, num_layers=4, dropout=0.1):
        """
        ინიციალიზაცია

        Args:
            grapheme_dim (int): გრაფემების რაოდენობა
            phoneme_dim (int): ფონემების რაოდენობა
            hidden_dim (int): ფარული შრის განზომილება
            num_layers (int): ფარული შრეების რაოდენობა
            dropout (float): Dropout-ის ალბათობა
        """
        super().__init__()

        self.grapheme_dim = grapheme_dim
        self.phoneme_dim = phoneme_dim
        self.hidden_dim = hidden_dim

        # გრაფემების ენკოდერი
        self.encoder = PhonemeEncoder(
            grapheme_dim, hidden_dim, hidden_dim, num_layers, dropout
        )

        # გამავალი შრე ფონემების პროგნოზისთვის
        self.classifier = nn.Linear(hidden_dim, phoneme_dim)

    def forward(self, x, x_lengths=None):
        """
        წინა გასვლა

        Args:
            x (torch.Tensor): შემავალი გრაფემების ტენზორი [batch_size, seq_len]
            x_lengths (torch.Tensor): მიმდევრობების სიგრძეები

        Returns:
            torch.Tensor: ფონემების ალბათობები [batch_size, seq_len, phoneme_dim]
        """
        # ენკოდერის გამოყენება
        encoded = self.encoder(x, x_lengths)  # [batch_size, seq_len, hidden_dim]

        # კლასიფიკატორის გამოყენება
        logits = self.classifier(encoded)  # [batch_size, seq_len, phoneme_dim]

        return logits


class PhonemeVocab:
    """
    ფონემების ლექსიკონი
    """

    def __init__(self, language="ka"):
        """
        ინიციალიზაცია

        Args:
            language (str): ენის კოდი
        """
        self.language = language
        self.phoneme_to_id = {}
        self.id_to_phoneme = {}
        self.grapheme_to_id = {}
        self.id_to_grapheme = {}

        # სპეციალური ტოკენები
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3
        }

        # ლექსიკონის ინიციალიზაცია
        self._initialize_vocab()

    def _initialize_vocab(self):
        """
        ლექსიკონის ინიციალიზაცია

        ქართული ენის გრაფემები და ფონემები
        """
        # ქართული ანბანის გრაფემები
        georgian_graphemes = [
            'ა', 'ბ', 'გ', 'დ', 'ე', 'ვ', 'ზ', 'თ', 'ი', 'კ',
            'ლ', 'მ', 'ნ', 'ო', 'პ', 'ჟ', 'რ', 'ს', 'ტ', 'უ',
            'ფ', 'ქ', 'ღ', 'ყ', 'შ', 'ჩ', 'ც', 'ძ', 'წ', 'ჭ',
            'ხ', 'ჯ', 'ჰ'
        ]

        # ქართული ფონემები (IPA წარმოდგენა)
        georgian_phonemes = [
            'a', 'b', 'g', 'd', 'ɛ', 'v', 'z', 'tʰ', 'i', 'kʼ',
            'l', 'm', 'n', 'ɔ', 'pʼ', 'ʒ', 'r', 's', 'tʼ', 'u',
            'pʰ', 'kʰ', 'ɣ', 'qʼ', 'ʃ', 'tʃʰ', 'tsʰ', 'dz', 'tsʼ', 'tʃʼ',
            'x', 'dʒ', 'h'
        ]

        # დამატებითი ფონემები
        extra_phonemes = [
            '.', ',', '?', '!', '-', ':', ';', ' ', "'", '"',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
        ]

        # გრაფემების ლექსიკონის შექმნა
        for i, token in enumerate(self.special_tokens.keys()):
            self.grapheme_to_id[token] = i
            self.id_to_grapheme[i] = token

        next_id = len(self.special_tokens)
        for grapheme in georgian_graphemes + extra_phonemes:
            self.grapheme_to_id[grapheme] = next_id
            self.id_to_grapheme[next_id] = grapheme
            next_id += 1

        # ფონემების ლექსიკონის შექმნა
        for i, token in enumerate(self.special_tokens.keys()):
            self.phoneme_to_id[token] = i
            self.id_to_phoneme[i] = token

        next_id = len(self.special_tokens)
        for phoneme in georgian_phonemes + extra_phonemes:
            self.phoneme_to_id[phoneme] = next_id
            self.id_to_phoneme[next_id] = phoneme
            next_id += 1

    def grapheme_to_phoneme_dict(self):
        """
        გრაფემა-ფონემა ლექსიკონის მიღება

        Returns:
            dict: გრაფემა-ფონემა წყვილები
        """
        g2p_dict = {}

        # ქართული ანბანის გრაფემები
        georgian_graphemes = [
            'ა', 'ბ', 'გ', 'დ', 'ე', 'ვ', 'ზ', 'თ', 'ი', 'კ',
            'ლ', 'მ', 'ნ', 'ო', 'პ', 'ჟ', 'რ', 'ს', 'ტ', 'უ',
            'ფ', 'ქ', 'ღ', 'ყ', 'შ', 'ჩ', 'ც', 'ძ', 'წ', 'ჭ',
            'ხ', 'ჯ', 'ჰ'
        ]

        # ქართული ფონემები (IPA წარმოდგენა)
        georgian_phonemes = [
            'a', 'b', 'g', 'd', 'ɛ', 'v', 'z', 'tʰ', 'i', 'kʼ',
            'l', 'm', 'n', 'ɔ', 'pʼ', 'ʒ', 'r', 's', 'tʼ', 'u',
            'pʰ', 'kʰ', 'ɣ', 'qʼ', 'ʃ', 'tʃʰ', 'tsʰ', 'dz', 'tsʼ', 'tʃʼ',
            'x', 'dʒ', 'h'
        ]

        # გრაფემა-ფონემა წყვილების შექმნა
        for grapheme, phoneme in zip(georgian_graphemes, georgian_phonemes):
            g2p_dict[grapheme] = phoneme

        # დამატებითი ფონემები
        extra_graphemes = [
            '.', ',', '?', '!', '-', ':', ';', ' ', "'", '"',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
        ]

        for char in extra_graphemes:
            g2p_dict[char] = char

        return g2p_dict

    def encode_graphemes(self, text):
        """
        ტექსტის გრაფემებად კოდირება

        Args:
            text (str): საწყისი ტექსტი

        Returns:
            list: გრაფემების ID-ების სია
        """
        grapheme_ids = []

        for char in text:
            if char in self.grapheme_to_id:
                grapheme_ids.append(self.grapheme_to_id[char])
            else:
                grapheme_ids.append(self.grapheme_to_id["<unk>"])

        return grapheme_ids

    def decode_graphemes(self, grapheme_ids):
        """
        გრაფემების ID-ების დეკოდირება ტექსტად

        Args:
            grapheme_ids (list): გრაფემების ID-ების სია

        Returns:
            str: დეკოდირებული ტექსტი
        """
        text = ""

        for id in grapheme_ids:
            if id in self.id_to_grapheme:
                text += self.id_to_grapheme[id]

        return text

    def encode_phonemes(self, phonemes):
        """
        ფონემების კოდირება

        Args:
            phonemes (list): ფონემების სია

        Returns:
            list: ფონემების ID-ების სია
        """
        phoneme_ids = []

        for phoneme in phonemes:
            if phoneme in self.phoneme_to_id:
                phoneme_ids.append(self.phoneme_to_id[phoneme])
            else:
                phoneme_ids.append(self.phoneme_to_id["<unk>"])

        return phoneme_ids

    def decode_phonemes(self, phoneme_ids):
        """
        ფონემების ID-ების დეკოდირება

        Args:
            phoneme_ids (list): ფონემების ID-ების სია

        Returns:
            list: ფონემების სია
        """
        phonemes = []

        for id in phoneme_ids:
            if id in self.id_to_phoneme:
                phonemes.append(self.id_to_phoneme[id])

        return phonemes

    def text_to_phonemes(self, text, g2p_dict=None):
        """
        ტექსტის ფონემებად გარდაქმნა

        Args:
            text (str): საწყისი ტექსტი
            g2p_dict (dict): გრაფემა-ფონემა ლექსიკონი

        Returns:
            list: ფონემების სია
        """
        if g2p_dict is None:
            g2p_dict = self.grapheme_to_phoneme_dict()

        phonemes = []

        for char in text:
            if char in g2p_dict:
                phonemes.append(g2p_dict[char])
            else:
                phonemes.append("<unk>")

        return phonemes

    def save(self, path):
        """
        ლექსიკონის შენახვა

        Args:
            path (str): შესანახი გზა
        """
        vocab_data = {
            "language": self.language,
            "phoneme_to_id": self.phoneme_to_id,
            "id_to_phoneme": self.id_to_phoneme,
            "grapheme_to_id": self.grapheme_to_id,
            "id_to_grapheme": self.id_to_grapheme,
            "special_tokens": self.special_tokens,
            "g2p_dict": self.grapheme_to_phoneme_dict()
        }

        with open(path, 'wb') as f:
            pickle.dump(vocab_data, f)

        logger.info(f"ფონემების ლექსიკონი შენახულია: {path}")

    @classmethod
    def load(cls, path):
        """
        ლექსიკონის ჩატვირთვა

        Args:
            path (str): ჩასატვირთი გზა

        Returns:
            PhonemeVocab: ლექსიკონის ობიექტი
        """
        with open(path, 'rb') as f:
            vocab_data = pickle.load(f)

        vocab = cls(vocab_data["language"])
        vocab.phoneme_to_id = vocab_data["phoneme_to_id"]
        vocab.id_to_phoneme = vocab_data["id_to_phoneme"]
        vocab.grapheme_to_id = vocab_data["grapheme_to_id"]
        vocab.id_to_grapheme = vocab_data["id_to_grapheme"]
        vocab.special_tokens = vocab_data["special_tokens"]

        logger.info(f"ფონემების ლექსიკონი ჩატვირთულია: {path}")
        return vocab


class PhonemeProcessor:
    """
    ფონემების დამუშავების კლასი
    """

    def __init__(self, config_path):
        """
        ინიციალიზაცია

        Args:
            config_path (str): კონფიგურაციის ფაილის გზა
        """
        # კონფიგურაციის ჩატვირთვა
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.output_dir = Path(self.config["output_model_path"])
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # მოწყობილობის ინიციალიზაცია - აქ იყო პრობლემა
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"გამოყენებული მოწყობილობა: {self.device}")

        # ფონემების ლექსიკონის შექმნა
        self.vocab = PhonemeVocab(self.config["language"])

        # ფონემების ლექსიკონის შენახვა
        vocab_path = self.output_dir / "phoneme_vocab.pkl"
        if not vocab_path.exists():
            self.vocab.save(vocab_path)

        # G2P მოდელის ინიციალიზაცია
        phoneme_config = self.config.get("phoneme_model", {})
        self.g2p_model = GraphemeToPhoneme(
            len(self.vocab.grapheme_to_id),
            len(self.vocab.phoneme_to_id),
            phoneme_config.get("hidden_size", 512),
            phoneme_config.get("num_hidden_layers", 4),
            phoneme_config.get("dropout_prob", 0.1)
        ).to(self.device)

    def process_text(self, text):
        """
        ტექსტის დამუშავება ფონემებად

        Args:
            text (str): საწყისი ტექსტი

        Returns:
            list: ფონემების სია
        """
        # ტექსტის გარდაქმნა ფონემებად
        phonemes = self.vocab.text_to_phonemes(text)

        return phonemes

    def save_phoneme_dict(self):
        """
        ფონემების ლექსიკონის შენახვა

        Returns:
            str: შენახული ფაილის გზა
        """
        # ფონემების ლექსიკონის შენახვა
        g2p_dict = self.vocab.grapheme_to_phoneme_dict()
        phoneme_dict_path = self.output_dir / "phoneme_dict_ka.json"

        with open(phoneme_dict_path, 'w', encoding='utf-8') as f:
            json.dump(g2p_dict, f, ensure_ascii=False, indent=2)

        logger.info(f"ფონემების ლექსიკონი შენახულია: {phoneme_dict_path}")
        return str(phoneme_dict_path)

    def create_phoneme_model(self):
        """
        ფონემების მოდელის შექმნა და შენახვა

        Returns:
            tuple: (მოდელი, გრაფემა-ფონემა ლექსიკონი)
        """
        logger.info("ფონემების მოდელის შექმნა")

        # G2P მოდელის ინიციალიზაცია
        model = self.g2p_model

        # ფონემების ლექსიკონის შენახვა
        g2p_dict_path = self.save_phoneme_dict()

        # G2P მოდელის შენახვა
        model_path = self.output_dir / "g2p_model.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"G2P მოდელი შენახულია: {model_path}")

        # G2P კონფიგურაციის შენახვა
        g2p_config = {
            "language": self.config["language"],
            "use_espeak": self.config.get("phoneme_model", {}).get("use_espeak", True),
            "phoneme_dict_path": g2p_dict_path,
            "model_path": str(model_path),
            "grapheme_dim": len(self.vocab.grapheme_to_id),
            "phoneme_dim": len(self.vocab.phoneme_to_id)
        }

        g2p_config_path = self.output_dir / "g2p_config.json"
        with open(g2p_config_path, 'w', encoding='utf-8') as f:
            json.dump(g2p_config, f, ensure_ascii=False, indent=2)

        logger.info(f"G2P კონფიგურაცია შენახულია: {g2p_config_path}")

        return model, g2p_config

    def run(self):
        """
        სრული პროცესის გაშვება

        Returns:
            tuple: (მოდელი, გრაფემა-ფონემა ლექსიკონი)
        """
        logger.info("ფონემების დამუშავების პროცესის დაწყება")
        start_time = time.time()

        # ფონემების მოდელის შექმნა
        model, g2p_config = self.create_phoneme_model()

        elapsed_time = time.time() - start_time
        logger.info(f"ფონემების დამუშავების პროცესი დასრულებულია")
        logger.info(f"მთლიანი დრო: {elapsed_time:.2f} წამი")

        return model, g2p_config


def main():
    """
    მთავარი ფუნქცია
    """
    parser = argparse.ArgumentParser(description="ფონემების დამუშავება XTTS-v2 მოდელისთვის")
    parser.add_argument("--config", type=str, default="xtts_pretrain_config.json",
                        help="კონფიგურაციის ფაილის გზა")
    args = parser.parse_args()

    # ფონემების დამუშავების პროცესის გაშვება
    processor = PhonemeProcessor(args.config)
    processor.run()


if __name__ == "__main__":
    main()