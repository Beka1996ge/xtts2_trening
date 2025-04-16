#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ქართული XTTS-v2 მოდელის ტრენინგისთვის მონაცემთა კლასები
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Union
from torch.utils.data import Dataset
import torchaudio


class XTTSGeorgianDataset(Dataset):
    """
    ქართული ტექსტი-მეტყველების მონაცემთა კლასი XTTS-v2 მოდელისთვის
    """

    def __init__(
            self,
            metadata_path: str,
            embeddings_dir: str,
            audio_dir: str,
            max_audio_length: int = 10,  # წამებში
            sample_rate: int = 16000,
            is_eval: bool = False,
            return_wave: bool = False,
            return_mels: bool = True,
            phoneme_processor=None
    ):
        """
        Args:
            metadata_path (str): მეტადატა ფაილის მისამართი
            embeddings_dir (str): ხმოვანი ემბედინგების დირექტორია
            audio_dir (str): აუდიო ფაილების დირექტორია
            max_audio_length (int): მაქსიმალური აუდიო სიგრძე წამებში
            sample_rate (int): ხმის გაციფრების სიხშირე
            is_eval (bool): არის თუ არა ეს შეფასების (ვალიდაციის) ნაკრები
            return_wave (bool): დააბრუნოს თუ არა ხმოვანი ტალღა
            return_mels (bool): დააბრუნოს თუ არა მელ-სპექტროგრამები
            phoneme_processor: ფონემების პროცესორი (თუ გვაქვს)
        """
        self.metadata_path = metadata_path
        self.embeddings_dir = embeddings_dir
        self.audio_dir = audio_dir
        self.max_audio_length = max_audio_length
        self.sample_rate = sample_rate
        self.is_eval = is_eval
        self.return_wave = return_wave
        self.return_mels = return_mels
        self.phoneme_processor = phoneme_processor

        # მაქსიმალური აუდიო სიგრძე სემპლებში
        self.max_audio_length_samples = max_audio_length * sample_rate

        # მეტადატას ჩატვირთვა
        self.metadata = self._load_metadata()

        # მეტადატას ფილტრაცია ზომის მიხედვით
        self._filter_by_length()

        print(f"ჩატვირთულია {len(self.metadata)} აუდიო ფაილი.")

    def _load_metadata(self) -> List[Dict]:
        """
        მეტადატა ფაილის ჩატვირთვა

        Returns:
            List[Dict]: მეტადატა ჩანაწერების სია
        """
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"მეტადატა ფაილი ვერ მოიძებნა: {self.metadata_path}")

        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        return metadata

    def _filter_by_length(self):
        """
        მეტადატას ფილტრაცია აუდიო ფაილის ზომის მიხედვით
        """
        filtered_metadata = []

        for item in self.metadata:
            # შემოწმება, არსებობს თუ არა აუდიო და ემბედინგ ფაილები
            audio_path = os.path.join(self.audio_dir, item['audio_file'])
            embedding_path = os.path.join(self.embeddings_dir, item['speaker_embedding'])

            if not os.path.exists(audio_path):
                continue

            if not os.path.exists(embedding_path):
                continue

            # აუდიოს ხანგრძლივობის შემოწმება (თუ საჭიროა)
            if not self.is_eval:  # ტრენინგის ფაზაში ვფილტრავთ ხანგრძლივობით
                try:
                    info = torchaudio.info(audio_path)
                    duration = info.num_frames / info.sample_rate

                    if duration > self.max_audio_length:
                        continue
                except Exception as e:
                    print(f"შეცდომა აუდიო ფაილის დამუშავებისას {audio_path}: {e}")
                    continue

            filtered_metadata.append(item)

        self.metadata = filtered_metadata

    def __len__(self):
        """
        მონაცემთა ნაკრების ზომა

        Returns:
            int: ელემენტების რაოდენობა
        """
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        მონაცემთა ელემენტის მიღება ინდექსის მიხედვით

        Args:
            idx (int): ელემენტის ინდექსი

        Returns:
            Dict: მონაცემთა ბატჩი
        """
        item = self.metadata[idx]

        # აუდიო ფაილის მისამართი
        audio_path = os.path.join(self.audio_dir, item['audio_file'])

        # ემბედინგის ფაილის მისამართი
        embedding_path = os.path.join(self.embeddings_dir, item['speaker_embedding'])

        # ემბედინგის ჩატვირთვა
        speaker_embedding = np.load(embedding_path)
        speaker_embedding = torch.FloatTensor(speaker_embedding)

        # აუდიოს ჩატვირთვა
        waveform, sample_rate = torchaudio.load(audio_path)

        # მონო აუდიოს შემოწმება
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # სემპლირების სიხშირის დაახლოება (თუ საჭიროა)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.sample_rate
            )

        # ტექსტის მიღება
        text = item['text']

        # ფონემების გარდაქმნა (თუ გვაქვს პროცესორი)
        phonemes = None
        if self.phoneme_processor is not None:
            phonemes = self.phoneme_processor.text_to_phonemes(text)

        # შედეგის მომზადება
        result = {
            'text': text,
            'speaker_embedding': speaker_embedding,
            'audio_file': item['audio_file'],
            'speaker_id': item.get('speaker_id', 'unknown')
        }

        # ფონემების დამატება (თუ გვაქვს)
        if phonemes is not None:
            result['phonemes'] = phonemes

        # ხმოვანი ტალღის დამატება (თუ მოთხოვნილია)
        if self.return_wave:
            result['waveform'] = waveform.squeeze(0)

        # მელ-სპექტროგრამების დამატება (თუ მოთხოვნილია)
        if self.return_mels:
            # აქ შეგიძლიათ დაამატოთ მელ-სპექტროგრამების გამოთვლის ლოგიკა
            # ან გამოიყენოთ წინასწარ მომზადებული მელ-სპექტროგრამები
            pass

        return result


class XTTSCollator:
    """
    მონაცემთა კოლატორი, რომელიც აწარმოებს ბატჩების მომზადებას
    """

    def __init__(
            self,
            pad_token_id: int = 0,
            pad_to_multiple: int = 1,
            max_audio_length: int = None
    ):
        """
        Args:
            pad_token_id (int): შევსების ტოკენის ID
            pad_to_multiple (int): რამდენჯერ უნდა იყოს გაყოფილი შევსების შემდეგ სიგრძე
            max_audio_length (int, optional): მაქსიმალური აუდიო სიგრძე სემპლებში
        """
        self.pad_token_id = pad_token_id
        self.pad_to_multiple = pad_to_multiple
        self.max_audio_length = max_audio_length

    def __call__(self, batch):
        """
        ბატჩის მომზადება

        Args:
            batch: მონაცემთა სია

        Returns:
            Dict: მომზადებული ბატჩი
        """
        # ტექსტის დასაპადინგება და მასკირება
        input_ids, attention_mask = self._prepare_text([item['text'] for item in batch])

        # ფონემების დასაპადინგება (თუ გვაქვს)
        if 'phonemes' in batch[0]:
            phoneme_ids, phoneme_mask = self._prepare_text([item['phonemes'] for item in batch])
        else:
            phoneme_ids, phoneme_mask = None, None

        # ხმოვანი ტალღების დასაპადინგება (თუ გვაქვს)
        waveforms = None
        if 'waveform' in batch[0]:
            waveforms, waveform_lengths = self._prepare_audio([item['waveform'] for item in batch])

        # ემბედინგების მომზადება
        speaker_embeddings = torch.stack([item['speaker_embedding'] for item in batch])

        # შედეგის მომზადება
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'speaker_embeddings': speaker_embeddings,
            'audio_files': [item['audio_file'] for item in batch],
            'speaker_ids': [item.get('speaker_id', 'unknown') for item in batch]
        }

        # ფონემების დამატება (თუ გვაქვს)
        if phoneme_ids is not None:
            result['phoneme_ids'] = phoneme_ids
            result['phoneme_mask'] = phoneme_mask

        # ხმოვანი ტალღების დამატება (თუ გვაქვს)
        if waveforms is not None:
            result['waveforms'] = waveforms
            result['waveform_lengths'] = waveform_lengths

        return result

    def _prepare_text(self, texts):
        """
        ტექსტის მომზადება ბატჩისთვის

        Args:
            texts: ტექსტების სია

        Returns:
            Tuple: დასაპადინგებული ტექსტები და ყურადღების მასკა
        """
        # აქ განახორციელეთ ტექსტის დასაპადინგება და მასკირება
        # მაგალითად, გამოიყენეთ არსებული tokenizer ან საკუთარი მეთოდი

        # ეს არის მხოლოდ მარტივი მაგალითი:
        max_length = max(len(t) for t in texts)

        # დავამრგვალოთ pad_to_multiple-ის ჯერად რიცხვამდე
        if self.pad_to_multiple > 1:
            max_length = ((max_length + self.pad_to_multiple - 1) // self.pad_to_multiple) * self.pad_to_multiple

        padded_texts = []
        masks = []

        for text in texts:
            padding_length = max_length - len(text)
            padded_text = text + [self.pad_token_id] * padding_length
            mask = [1] * len(text) + [0] * padding_length

            padded_texts.append(padded_text)
            masks.append(mask)

        return torch.LongTensor(padded_texts), torch.BoolTensor(masks)

    def _prepare_audio(self, waveforms):
        """
        აუდიოს მომზადება ბატჩისთვის

        Args:
            waveforms: ხმოვანი ტალღების სია

        Returns:
            Tuple: დასაპადინგებული ხმოვანი ტალღები და სიგრძეები
        """
        # სიგრძეების განსაზღვრა
        lengths = [w.size(0) for w in waveforms]

        # მაქსიმალური სიგრძის განსაზღვრა (შეზღუდვით, თუ მითითებულია)
        max_length = max(lengths)
        if self.max_audio_length is not None:
            max_length = min(max_length, self.max_audio_length)

        # დავამრგვალოთ pad_to_multiple-ის ჯერად რიცხვამდე
        if self.pad_to_multiple > 1:
            max_length = ((max_length + self.pad_to_multiple - 1) // self.pad_to_multiple) * self.pad_to_multiple

        # დასაპადინგებული ტენზორის მომზადება
        padded = torch.zeros(len(waveforms), max_length)

        # თითოეული ხმოვანი ტალღის განთავსება
        for i, waveform in enumerate(waveforms):
            padded_length = min(waveform.size(0), max_length)
            padded[i, :padded_length] = waveform[:padded_length]

            # სიგრძის განახლება, თუ შევზღუდეთ
            if padded_length < lengths[i]:
                lengths[i] = padded_length

        return padded, torch.LongTensor(lengths)