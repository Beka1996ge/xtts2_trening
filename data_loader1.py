#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XTTS-v2 მოდელის პრეტრეინინგისთვის მონაცემების ჩამტვირთავი
ავტორი:
თარიღი: 2025-04-15
"""

import os
import json
import logging
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torchaudio
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("data_loader")


class XTTSDataset(Dataset):
    """
    XTTS-v2 მოდელის პრეტრეინინგისთვის მონაცემთა ნაკრები
    """

    def __init__(self, data_dir, split="train", config=None):
        """
        ინიციალიზაცია

        Args:
            data_dir (str): მონაცემთა დირექტორიის გზა
            split (str): ნაკრების ტიპი (train, val, test)
            config (dict): კონფიგურაციის პარამეტრები
        """
        self.data_dir = Path(data_dir)
        self.split_dir = self.data_dir / "split" / split
        self.config = config or {}

        # მეტადატას ჩატვირთვა
        self.metadata_path = self.split_dir / f"{split}_metadata.csv"
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"მეტადატას ფაილი ვერ მოიძებნა: {self.metadata_path}")

        self.metadata = pd.read_csv(self.metadata_path)
        logger.info(f"ჩატვირთულია {len(self.metadata)} ჩანაწერი {split} ნაკრებიდან")

        # ემბედინგების გზა
        self.embeddings_dir = self.data_dir / "embeddings" / "embeddings"

        # შევამოწმოთ, არსებობს თუ არა ყველა საჭირო ფაილი
        self._validate_files()

    def _validate_files(self):
        """
        შევამოწმოთ, რომ ყველა საჭირო ფაილი არსებობს
        """
        missing_audio = 0
        missing_embeds = 0
        valid_indices = []

        for idx, row in tqdm(self.metadata.iterrows(), total=len(self.metadata),
                             desc="მონაცემების ვალიდაცია"):
            audio_path = self.split_dir / "audio" / f"{row['file_id']}.wav"
            embed_path = self.embeddings_dir / f"{row['file_id']}.npy"

            if not audio_path.exists():
                missing_audio += 1
                continue

            if not embed_path.exists():
                missing_embeds += 1
                continue

            valid_indices.append(idx)

        if missing_audio > 0:
            logger.warning(f"ვერ მოიძებნა {missing_audio} აუდიო ფაილი")

        if missing_embeds > 0:
            logger.warning(f"ვერ მოიძებნა {missing_embeds} ემბედინგი")

        # დავტოვოთ მხოლოდ ვალიდური ჩანაწერები
        self.metadata = self.metadata.iloc[valid_indices].reset_index(drop=True)
        logger.info(f"ვალიდაციის შემდეგ დარჩა {len(self.metadata)} ჩანაწერი")

    def __len__(self):
        """
        ნაკრების ზომა
        """
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        კონკრეტული მაგალითის წამოღება

        Args:
            idx (int): მაგალითის ინდექსი

        Returns:
            dict: მაგალითის მონაცემები
        """
        row = self.metadata.iloc[idx]
        file_id = row['file_id']

        # აუდიოს ჩატვირთვა
        audio_path = self.split_dir / "audio" / f"{file_id}.wav"
        waveform, sample_rate = torchaudio.load(audio_path)

        # ემბედინგის ჩატვირთვა
        embed_path = self.embeddings_dir / f"{file_id}.npy"
        embedding = np.load(embed_path)

        # ტექსტის მიღება
        text = row['text']

        # ამოღებული მონაცემების შეკრება სამუშაო განლაგებაში
        sample = {
            "file_id": file_id,
            "waveform": torch.FloatTensor(waveform),
            "sample_rate": sample_rate,
            "text": text,
            "speaker_embedding": torch.FloatTensor(embedding),
            "language": "ka"
        }

        return sample


class DataManager:
    """
    XTTS-v2 პრეტრეინინგისთვის მონაცემების მართვის კლასი
    """

    def __init__(self, config):
        """
        ინიციალიზაცია

        Args:
            config (dict): კონფიგურაციის პარამეტრები
        """
        self.config = config
        self.data_dir = Path(config["data_dir"])

    def get_dataloaders(self):
        """
        ყველა საჭირო DataLoader-ის მიღება

        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        # მონაცემთა ნაკრებების შექმნა
        train_dataset = XTTSDataset(self.data_dir, split="train", config=self.config)
        val_dataset = XTTSDataset(self.data_dir, split="val", config=self.config)
        test_dataset = XTTSDataset(self.data_dir, split="test", config=self.config)

        # DataLoader-ების შექმნა
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get("batch_size", 32),
            shuffle=True,
            num_workers=self.config.get("training", {}).get("num_workers", 4),
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get("batch_size", 32),
            shuffle=False,
            num_workers=self.config.get("training", {}).get("num_workers", 4),
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get("batch_size", 32),
            shuffle=False,
            num_workers=self.config.get("training", {}).get("num_workers", 4),
            pin_memory=True
        )

        logger.info(f"შექმნილია DataLoader-ები: train ({len(train_loader)} batch), "
                    f"val ({len(val_loader)} batch), test ({len(test_loader)} batch)")

        return train_loader, val_loader, test_loader

    def get_metadata_stats(self):
        """
        მეტამონაცემების სტატისტიკის მიღება

        Returns:
            dict: მეტამონაცემების სტატისტიკა
        """
        stats = {}

        for split in ["train", "val", "test"]:
            metadata_path = self.data_dir / "split" / split / f"{split}_metadata.csv"
            if metadata_path.exists():
                df = pd.read_csv(metadata_path)
                stats[split] = {
                    "count": len(df),
                    "avg_length": df.get("duration", df.get("length", pd.Series([0]))).mean(),
                    "total_duration": df.get("duration", df.get("length", pd.Series([0]))).sum()
                }

        return stats

    def get_sample(self, split="train", idx=0):
        """
        კონკრეტული მაგალითის მიღება

        Args:
            split (str): ნაკრების ტიპი (train, val, test)
            idx (int): მაგალითის ინდექსი

        Returns:
            dict: მაგალითის მონაცემები
        """
        dataset = XTTSDataset(self.data_dir, split=split, config=self.config)
        if idx < len(dataset):
            return dataset[idx]
        else:
            logger.error(f"არასწორი ინდექსი {idx}. მაქსიმალური ინდექსია {len(dataset) - 1}")
            return None


class CollateFunction:
    """
    XTTS მონაცემების კოლატ ფუნქცია, რომელიც მაგალითებს პაკეტში აერთიანებს
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.pad_token = 0
        self.pad_value = 0.0

    def __call__(self, batch):
        """
        მაგალითების კოლატი

        Args:
            batch (list): მაგალითების სია

        Returns:
            dict: დამუშავებული პაკეტი
        """
        # ფაილის ID-ების შეგროვება
        file_ids = [sample["file_id"] for sample in batch]

        # ტექსტი
        texts = [sample["text"] for sample in batch]

        # ხმოვანი ვექტორები
        waveforms = [sample["waveform"] for sample in batch]
        # პადინგისთვის მაქსიმალური სიგრძის მოპოვება
        max_audio_len = max(waveform.shape[1] for waveform in waveforms)
        # პადინგი ყველაზე გრძელ მაგალითამდე
        padded_waveforms = []
        waveform_lengths = []

        for waveform in waveforms:
            actual_len = waveform.shape[1]
            waveform_lengths.append(actual_len)

            # თუ სიგრძე ნაკლებია მაქსიმალურზე, დავამატოთ პადინგი
            if actual_len < max_audio_len:
                padding = torch.zeros(1, max_audio_len - actual_len)
                padded_waveform = torch.cat([waveform, padding], dim=1)
            else:
                padded_waveform = waveform

            padded_waveforms.append(padded_waveform)

        # ვექტორების დასტაკება
        waveforms_tensor = torch.stack(padded_waveforms)
        waveform_lengths = torch.tensor(waveform_lengths)

        # სპიკერის ემბედინგები
        speaker_embeddings = torch.stack([sample["speaker_embedding"] for sample in batch])

        # სემპლის რეიტები
        sample_rates = [sample["sample_rate"] for sample in batch]

        # ენები
        languages = [sample["language"] for sample in batch]

        # პაკეტში შეგროვება
        collated_batch = {
            "file_id": file_ids,
            "text": texts,
            "waveform": waveforms_tensor,
            "waveform_length": waveform_lengths,
            "speaker_embedding": speaker_embeddings,
            "sample_rate": torch.tensor(sample_rates),
            "language": languages
        }

        return collated_batch


def load_data_for_pretrain(config_path):
    """
    მონაცემების მომზადება XTTS-v2 პრეტრეინინგისთვის

    Args:
        config_path (str): კონფიგურაციის ფაილის გზა

    Returns:
        tuple: (data_manager, train_loader, val_loader, test_loader)
    """
    # კონფიგურაციის ჩატვირთვა
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # მონაცემების მენეჯერის შექმნა
    data_manager = DataManager(config)

    # დატალოადერების მიღება
    train_loader, val_loader, test_loader = data_manager.get_dataloaders()

    return data_manager, train_loader, val_loader, test_loader


if __name__ == "__main__":
    # ტესტირება
    import argparse

    parser = argparse.ArgumentParser(description="XTTS-v2 მონაცემების ტესტირება")
    parser.add_argument("--config", type=str, default="xtts_pretrain_config.json",
                        help="კონფიგურაციის ფაილის გზა")
    args = parser.parse_args()

    # ლოგირების კონფიგურაცია
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # მონაცემების მომზადება
    data_manager, train_loader, val_loader, test_loader = load_data_for_pretrain(args.config)

    # სტატისტიკის მიღება
    stats = data_manager.get_metadata_stats()
    print("მეტამონაცემების სტატისტიკა:")
    print(json.dumps(stats, indent=2))

    # პირველი ბეჩის ამოღება და ასახვა
    for batch in train_loader:
        print("\nპირველი ბეჩის შაბლონი:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: Tensor ზომით {value.shape}")
            else:
                print(f"{key}: {type(value)} ტიპის {len(value)} ელემენტი")
        break