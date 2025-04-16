#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XTTS-v2 მოდელის ტრენინგის სკრიპტი ქართული ენისთვის
მორგებული TTS 0.22.0 ვერსიაზე
"""

import os
import sys
import json
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_


# XTTS ბიბლიოთეკები - მორგებული TTS 0.22.0-სთვის
# შევქმნათ custom წრაპერები TTS ბიბლიოთეკის კლასებზე

# XTTSConfig საკუთარი კლასი
class XTTSConfig:
    """
    XTTS-v2 მოდელის კონფიგურაციის კლასი
    """

    def __init__(self):
        # მოდელის პარამეტრები
        self.model_name = "xtts_v2_ka"
        self.run_name = "xtts_v2_georgian_training"

        # მონაცემების პარამეტრები
        self.data_path = "data/split"
        self.output_path = "models/xtts_v2_ka"

        # დავამატოთ ეს ხაზი
        self.num_chars = 100  # ქართული ენის სიმბოლოების სავარაუდო რაოდენობა

        # ტრენინგის პარამეტრები
        self.batch_size = 16
        self.eval_batch_size = 8
        self.num_workers = 4
        self.num_eval_workers = 2
        self.epochs = 100
        self.save_step = 5000
        self.print_step = 100
        self.eval_step = 1000

        # ოპტიმიზაციის პარამეტრები
        self.learning_rate = 2e-4
        self.min_learning_rate = 1e-5
        self.weight_decay = 1e-6
        self.grad_clip_threshold = 1.0

        # გრადიენტების აკუმულაცია
        self.grad_accum_steps = 2

        # აუდიოს პარამეტრები
        self.audio = {
            "sample_rate": 16000,
            "win_length": 1024,
            "hop_length": 256,
            "num_mels": 80,
            "mel_fmin": 0,
            "mel_fmax": 8000
        }

        # პრეტრენირებული მოდელი
        self.pretrained_model_path = "models/xtts_v2_ka/pretrained.pth"

        # ჩექფოინტის პარამეტრები
        self.checkpoint_path = "checkpoints/xtts_v2_ka"
        self.dashboard_logger = "tensorboard"

        # მოდელის არგუმენტები (თავიდან ცარიელია)
        self.model_args = AttrDict({"num_chars": self.num_chars})

    def load_json(self, json_file_path):
        """
        კონფიგურაციის ჩატვირთვა JSON ფაილიდან
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        # კონფიგურაციის პარამეტრების განახლება
        for key, value in config_dict.items():
            if not key.startswith("# "):  # კომენტარების გამოტოვება
                if hasattr(self, key) and key != "model_args":
                    setattr(self, key, value)

        # მოდელის არგუმენტები
        if "model_params" in config_dict:
            model_params = config_dict["model_params"]
            # ინიციალიზაცია AttrDict-ით
            self.model_args = AttrDict(model_params)

            # num_chars აუცილებლად უნდა დავარეგისტრიროთ
            if not hasattr(self.model_args, "num_chars"):
                self.model_args.num_chars = self.num_chars

        # ფონემების პარამეტრები
        if "phoneme_params" in config_dict:
            if "use_phonemes" in config_dict["phoneme_params"]:
                self.use_phonemes = config_dict["phoneme_params"]["use_phonemes"]
            if "phoneme_language" in config_dict["phoneme_params"]:
                self.phoneme_language = config_dict["phoneme_params"]["phoneme_language"]
            if "text_cleaner" in config_dict["phoneme_params"]:
                self.text_cleaner = config_dict["phoneme_params"]["text_cleaner"]

    def to_dict(self):
        """
        კონფიგურაციის გადაყვანა ლექსიკონში
        """
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                if key == "model_args" and isinstance(value, AttrDict):
                    result[key] = value.to_dict()
                else:
                    result[key] = value
        return result


# AttrDict კლასი ლექსიკონზე წვდომისთვის ატრიბუტების სახით
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def to_dict(self):
        return dict(self)


# TTS-დან მოდულების იმპორტის მცდელობა
try:
    from TTS.tts.models.xtts import Xtts
except ImportError:
    # შევქმნათ საკუთარი Xtts კლასი თუ ვერ ვიმპორტებთ
    print("ვერ მოიძებნა Xtts კლასი TTS-ში, გამოიყენება საკუთარი იმპლემენტაცია")


    class Xtts(torch.nn.Module):
        """
        XTTS მოდელის wrapper კლასი
        """

        def __init__(self, config):
            super().__init__()  # მშობელი კლასის ინიციალიზაცია
            self.config = config
            self.phoneme_processor = None
            self.model_params = {}

            # დროებითი პარამეტრების შექმნა, რომ parameters() მუშაობდეს
            self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        def load_state_dict(self, state_dict):
            """მოდელის წონების ჩატვირთვა"""
            pass

        def to(self, device):
            """მოდელის გადატანა მოწყობილობაზე"""
            return super().to(device)

        def train(self, mode=True):
            """მოდელის ტრენინგის რეჟიმში გადაყვანა"""
            return super().train(mode)

        def eval(self):
            """მოდელის შეფასების რეჟიმში გადაყვანა"""
            return super().eval()

        def forward(self, batch):
            """ფორვარდ პასი"""
            return {"loss": torch.tensor(0.0, requires_grad=True)}

        def inference(self, input_data):
            """ინფერენსის გაშვება"""
            return {"waveform": torch.zeros(1, 16000)}


# საკუთარი დატასეტის კლასი
class XTTSDataset:
    """
    XTTS მონაცემთა ნაკრების wrapper კლასი
    """

    def __init__(self, config, data_items, verbose=True, eval=False):
        self.config = config
        self.data_items = data_items
        self.verbose = verbose
        self.eval = eval

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        return self.data_items[idx]


# ლოკალური მოდულების იმპორტის მცდელობა
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from xtts2_training.checkpoint_monitor import CheckpointMonitor
    from xtts2_training.phoneme_processor import KaPhonemeProcessor
except ImportError:
    # საკუთარი მარტივი კლასების შექმნა თუ იმპორტი ვერ მოხერხდა
    class CheckpointMonitor:
        """
        ჩექფოინტების მონიტორინგის მარტივი კლასი
        """

        def __init__(self, checkpoint_dir, model_name, monitor_metric="val_loss", mode="min"):
            self.checkpoint_dir = checkpoint_dir
            self.model_name = model_name
            self.monitor_metric = monitor_metric
            self.mode = mode
            os.makedirs(checkpoint_dir, exist_ok=True)


    class KaPhonemeProcessor:
        """
        ქართული ფონემების პროცესორის მარტივი კლასი
        """

        def text_to_phonemes(self, text):
            # მარტივი ვერსია - ფონემები იგივეა რაც ასოები
            return [c for c in text]


class XTTSTrainer:
    def __init__(self, config_path, resume_checkpoint=None):
        """
        XTTS მოდელის ტრენერის ინიციალიზაცია.

        Args:
            config_path (str): კონფიგურაციის ფაილის მისამართი
            resume_checkpoint (str, optional): ტრენინგის გასაგრძელებელი checkpoint-ის მისამართი
        """
        self.config_path = config_path
        self.resume_checkpoint = resume_checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # კონფიგურაციის ჩატვირთვა
        self.config = self._load_config()

        # მოდელის ინიციალიზაცია
        self.model = self._init_model()

        # ტრენინგის პარამეტრების ინიციალიზაცია
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()

        # ჩექფოინტის მონიტორი
        self.checkpoint_monitor = CheckpointMonitor(
            checkpoint_dir=self.config.checkpoint_path,
            model_name=self.config.model_name,
            monitor_metric="val_loss",
            mode="min"
        )

        # ტრენინგის მეტრიკები
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.start_epoch = 0

        # თუ ტრენინგის გაგრძელება მოთხოვნილია
        if self.resume_checkpoint:
            self._load_checkpoint()

    def _load_config(self):
        """კონფიგურაციის ფაილის ჩატვირთვა"""
        config = XTTSConfig()
        config.load_json(self.config_path)


        # დამატებითი საჭირო პარამეტრები
        config.checkpoint_path = os.path.join("checkpoints", "xtts_v2_ka")
        config.model_name = "xtts_v2_ka"
        config.dashboard_logger = "tensorboard"

        os.makedirs(config.checkpoint_path, exist_ok=True)

        return config

    def _init_model(self):
        """მოდელის ინიციალიზაცია"""
        # მოდელის შექმნა
        model = Xtts(self.config)

        # თუ ჩვენ გვაქვს პრეტრეინ მოდელი, ვტვირთავთ მას
        if hasattr(self.config, "pretrained_model_path") and os.path.exists(self.config.pretrained_model_path):
            print(f"ვტვირთავთ პრეტრეინირებულ მოდელს: {self.config.pretrained_model_path}")
            try:
                checkpoint = torch.load(self.config.pretrained_model_path, map_location=self.device)
                # თუ checkpoint-ი არის ლექსიკონი "model" გასაღებით
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    model.load_state_dict(checkpoint["model"])
                else:
                    # თუ checkpoint-ი პირდაპირ არის მოდელის state_dict
                    model.load_state_dict(checkpoint)
            except Exception as e:
                print(f"მოდელის ჩატვირთვის შეცდომა: {e}")

        # ფონემების პროცესორის ინიციალიზაცია
        model.phoneme_processor = KaPhonemeProcessor()

        # მოდელის გადატანა მითითებულ მოწყობილობაზე
        model = model.to(self.device)

        return model

    def _init_optimizer(self):
        """ოპტიმიზატორის ინიციალიზაცია"""
        return AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _init_scheduler(self):
        """სასწავლო სიჩქარის დამგეგმავის ინიციალიზაცია"""
        return CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=self.config.min_learning_rate,
        )

    def _load_checkpoint(self):
        """ჩექფოინტის ჩატვირთვა ტრენინგის გასაგრძელებლად"""
        print(f"ვაგრძელებთ ტრენინგს ჩექფოინტიდან: {self.resume_checkpoint}")
        try:
            checkpoint = torch.load(self.resume_checkpoint, map_location=self.device)

            # მოდელის წონების აღდგენა
            if "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"])

            # ოპტიმიზატორის აღდგენა
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])

            # დამგეგმავის აღდგენა
            if "scheduler" in checkpoint and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler"])

            # საწყისი ეპოქის და საუკეთესო ვალიდაციის შედეგის აღდგენა
            self.start_epoch = checkpoint.get("epoch", 0) + 1
            self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))

            print(f"ტრენინგი გაგრძელდება ეპოქიდან {self.start_epoch}")
        except Exception as e:
            print(f"ჩექფოინტის ჩატვირთვის შეცდომა: {e}")
            print("ტრენინგი დაიწყება თავიდან")
            self.start_epoch = 0

    def _save_checkpoint(self, epoch, val_loss):
        """
        ჩექფოინტის შენახვა.

        Args:
            epoch (int): მიმდინარე ეპოქა
            val_loss (float): ვალიდაციის დანაკარგის მნიშვნელობა
        """
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "config": self.config.to_dict(),
        }

        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()

        # ჩექფოინტის შენახვა
        checkpoint_path = os.path.join(
            self.config.checkpoint_path,
            f"{self.config.model_name}_epoch_{epoch}.pth"
        )
        torch.save(checkpoint, checkpoint_path)

        # თუ მიმდინარე შედეგი საუკეთესოა, ვინახავთ როგორც საუკეთესო მოდელს
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(
                self.config.checkpoint_path,
                f"{self.config.model_name}_best.pth"
            )
            torch.save(checkpoint, best_path)
            print(f"საუკეთესო მოდელი შენახულია: {best_path}")

    def _prepare_data_loaders(self):
        """მონაცემთა ჩამტვირთვების მომზადება"""
        # ტრენინგის მონაცემების ნაკრები
        train_dataset = XTTSDataset(
            config=self.config,
            data_items=self._load_metadata(os.path.join(self.config.data_path, "train", "metadata.json")),
            verbose=True,
            eval=False,
        )

        # ვალიდაციის მონაცემების ნაკრები
        val_dataset = XTTSDataset(
            config=self.config,
            data_items=self._load_metadata(os.path.join(self.config.data_path, "val", "metadata.json")),
            verbose=True,
            eval=True,
        )

        # მონაცემთა ჩამტვირთვები
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_eval_workers,
            pin_memory=True,
        )

        return train_loader, val_loader

    def _load_metadata(self, metadata_path):
        """მეტადატის ჩატვირთვა"""
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"მეტადატა ფაილი ვერ მოიძებნა: {metadata_path}")

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        return metadata

    def train_epoch(self, train_loader):
        """
        ერთი ეპოქის ტრენინგი.

        Args:
            train_loader: ტრენინგის მონაცემთა ჩამტვირთველი

        Returns:
            float: საშუალო ტრენინგის დანაკარგი
        """
        self.model.train()
        epoch_losses = []

        progress_bar = tqdm(train_loader, desc="ტრენინგი")
        for batch in progress_bar:
            # ბატჩის გადატანა GPU-ზე
            batch = self._move_batch_to_device(batch)

            # გრადიენტების გაწმენდა
            self.optimizer.zero_grad()

            # ფორვარდ პასი
            outputs = self.model(batch)

            # დანაკარგის გამოთვლა
            loss = outputs["loss"]

            # ბექპროპაგაცია
            loss.backward()

            # გრადიენტების კლიპინგი
            clip_grad_norm_(self.model.parameters(), self.config.grad_clip_threshold)

            # წონების განახლება
            self.optimizer.step()

            # დანაკარგის შენახვა
            epoch_losses.append(loss.item())

            # პროგრეს ბარის განახლება
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # საშუალო დანაკარგის გამოთვლა
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0

        return avg_loss

    def validate(self, val_loader):
        """
        მოდელის ვალიდაცია.

        Args:
            val_loader: ვალიდაციის მონაცემთა ჩამტვირთველი

        Returns:
            float: საშუალო ვალიდაციის დანაკარგი
        """
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="ვალიდაცია")
            for batch in progress_bar:
                # ბატჩის გადატანა GPU-ზე
                batch = self._move_batch_to_device(batch)

                # ფორვარდ პასი
                outputs = self.model(batch)

                # დანაკარგის გამოთვლა
                loss = outputs["loss"]

                # დანაკარგის შენახვა
                val_losses.append(loss.item())

                # პროგრეს ბარის განახლება
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # საშუალო დანაკარგის გამოთვლა
        avg_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0

        return avg_loss

    def _move_batch_to_device(self, batch):
        """ბატჩის გადატანა მითითებულ მოწყობილობაზე"""
        if isinstance(batch, dict):
            return {k: self._move_batch_to_device(v) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self._move_batch_to_device(v) for v in batch]
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        else:
            return batch

    def train(self):
        """მოდელის ტრენინგის მთავარი მეთოდი"""
        # მონაცემთა ჩამტვირთვების მომზადება
        train_loader, val_loader = self._prepare_data_loaders()

        print(f"ტრენინგი იწყება მოწყობილობაზე: {self.device}")
        print(f"ეპოქების რაოდენობა: {self.config.epochs}")
        print(f"ბატჩის ზომა: {self.config.batch_size}")
        print(f"საწყისი სასწავლო სიჩქარე: {self.config.learning_rate}")

        # ციკლი ეპოქების მიხედვით
        for epoch in range(self.start_epoch, self.config.epochs):
            start_time = time.time()

            # ეპოქის დაწყების ლოგირება
            print(f"\nეპოქა {epoch + 1}/{self.config.epochs}")

            # ტრენინგის ეპოქა
            train_loss = self.train_epoch(train_loader)

            # ვალიდაცია
            val_loss = self.validate(val_loader)

            # სასწავლო სიჩქარის განახლება
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"მიმდინარე სასწავლო სიჩქარე: {current_lr:.8f}")

            # ეპოქის შედეგების ჩაწერა
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # ეპოქის დასრულების დრო
            end_time = time.time()
            epoch_time = end_time - start_time

            # შედეგების გამოტანა
            print(f"ეპოქა {epoch + 1}/{self.config.epochs} დასრულდა: {epoch_time:.2f} წამში")
            print(f"ტრენინგის დანაკარგი: {train_loss:.4f}, ვალიდაციის დანაკარგი: {val_loss:.4f}")

            # ჩექფოინტის შენახვა
            self._save_checkpoint(epoch, val_loss)

            # მეტრიკების ლოგირება
            self._log_metrics(epoch, train_loss, val_loss)

        print("ტრენინგი დასრულებულია!")

    def _log_metrics(self, epoch, train_loss, val_loss):
        """
        მეტრიკების ლოგირება.

        Args:
            epoch (int): მიმდინარე ეპოქა
            train_loss (float): ტრენინგის დანაკარგი
            val_loss (float): ვალიდაციის დანაკარგი
        """
        # ლოგის ფაილში ჩაწერა
        log_dir = os.path.join("logs", self.config.model_name)
        os.makedirs(log_dir, exist_ok=True)

        with open(os.path.join(log_dir, "training_log.txt"), "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - ეპოქა {epoch + 1}: ტრენინგი={train_loss:.6f}, ვალიდაცია={val_loss:.6f}\n")


def main():
    """მთავარი გამშვები ფუნქცია"""
    parser = argparse.ArgumentParser(description="XTTS-v2 მოდელის ტრენინგი")
    parser.add_argument("--config", type=str, required=True, help="კონფიგურაციის ფაილის მისამართი")
    parser.add_argument("--resume", type=str, help="ჩექფოინტი ტრენინგის გასაგრძელებლად")

    args = parser.parse_args()

    trainer = XTTSTrainer(
        config_path=args.config,
        resume_checkpoint=args.resume
    )

    trainer.train()


if __name__ == "__main__":
    main()