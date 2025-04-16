#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XTTS-v2 მოდელის პრეტრეინინგი ქართული ენისთვის
ავტორი:
თარიღი: 2025-04-15
"""

import os
import sys
import json
import logging
import argparse
import shutil
import pickle
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# კონფიგურაციის საწყისი პარამეტრები
DEFAULT_CONFIG = {
    "language": "ka",
    "checkpoint_dir": "checkpoints/",
    "checkpoint_interval": 5000,  # ნაბიჯების რაოდენობა თითოეულ checkpoint-მდე
    "log_dir": "logs/",
    "base_model_path": "models/base_xtts_v2/",
    "output_model_path": "models/xtts_v2_ka/",
    "data_dir": "data/split/",
    "batch_size": 32,
    "learning_rate": 0.0001,
    "epochs": 100,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "max_steps": None,  # None ნიშნავს, რომ მთლიანი ეპოქები გაირბენს
    "optimizer": "AdamW",
    "lr_scheduler": "CosineAnnealingLR"
}

# ლოგების კონფიგურაცია
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("pretrain_xtts")


def setup_logging(config):
    """
    ლოგების კონფიგურაცია, ყველა ლოგის ფაილში შენახვა
    """
    log_dir = Path(config["log_dir"])
    log_dir.mkdir(exist_ok=True, parents=True)

    log_file = log_dir / f"pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info(f"ლოგების ფაილი: {log_file}")
    logger.info(f"კონფიგურაცია: {config}")


def load_config(config_path=None):
    """
    კონფიგურაციის ჩატვირთვა ფაილიდან, ან საწყისი კონფიგურაციის გამოყენება
    """
    config = DEFAULT_CONFIG.copy()

    if config_path and os.path.exists(config_path):
        logger.info(f"კონფიგურაციის ჩატვირთვა ფაილიდან: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
            config.update(loaded_config)

    # მოწყობილობის (device) შემოწმება
    if config["device"] == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA არ არის ხელმისაწვდომი, გამოიყენება CPU")
        config["device"] = "cpu"

    return config


def save_config(config, path):
    """
    კონფიგურაციის შენახვა JSON ფაილში
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(f"კონფიგურაცია შენახულია: {path}")


def setup_directories(config):
    """
    საჭირო დირექტორიების შექმნა
    """
    dirs = [
        config["checkpoint_dir"],
        config["log_dir"],
        config["output_model_path"],
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"დირექტორია შექმნილია: {dir_path}")


def set_seed(seed):
    """
    რეპროდუცირებადობისთვის seed-ის დაყენება
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seed დაყენებულია: {seed}")


class CheckpointManager:
    """
    Checkpoint-ების მართვის კლასი
    """

    def __init__(self, checkpoint_dir, interval=5000):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.interval = interval
        self.last_step = 0
        self.latest_checkpoint = None

    def get_latest_checkpoint(self):
        """
        უახლესი checkpoint-ის პოვნა
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        if not checkpoints:
            return None

        latest = max(checkpoints, key=lambda x: int(x.stem.split("_")[1]))
        self.latest_checkpoint = latest
        self.last_step = int(latest.stem.split("_")[1])
        return latest

    def save_checkpoint(self, model, optimizer, scheduler, step, additional_data=None):
        """
        მოდელის, ოპტიმიზატორის და დამატებითი მონაცემების checkpoint-ის შენახვა
        """
        if step % self.interval != 0:
            return False

        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step}.pkl"

        checkpoint_data = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        if scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()

        if additional_data:
            checkpoint_data.update(additional_data)

        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Checkpoint შენახულია: {checkpoint_path}")

        self.last_step = step
        self.latest_checkpoint = checkpoint_path
        return True

    def load_checkpoint(self, model, optimizer=None, scheduler=None):
        """
        უახლესი checkpoint-ის ჩატვირთვა
        """
        latest = self.get_latest_checkpoint()
        if latest is None:
            logger.info("checkpoint-ები არ მოიძებნა, ახლიდან დაწყება")
            return 0

        logger.info(f"checkpoint-ის ჩატვირთვა: {latest}")
        checkpoint = torch.load(latest, map_location=torch.device('cpu'))

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"checkpoint ჩატვირთულია, ბიჯი: {checkpoint['step']}")
        return checkpoint["step"]


class LanguageModelProcessor:
    """
    ენის მოდელის დამუშავების კლასი
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])

    def load_or_init_language_model(self):
        """
        ენის მოდელის ჩატვირთვა ან ინიციალიზაცია
        """
        # აქ იქნება კოდი მოდელის ჩატვირთვისთვის ან ინიციალიზაციისთვის
        # დროებითი "dummy" მოდელი, რეალური იმპლემენტაციისთვის შეცვალეთ
        model = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)
        ).to(self.device)

        logger.info("ენის მოდელი ინიციალიზებულია")
        return model

    def adapt_for_georgian(self, model):
        """
        ენის მოდელის ადაპტირება ქართული ენისთვის
        """
        # აქ იქნება კოდი ქართული ენისთვის ადაპტაციისთვის
        logger.info("მოდელი ადაპტირებულია ქართული ენისთვის")
        return model

    def process(self):
        """
        ენის მოდელის დამუშავების პროცესი
        """
        logger.info("ენის მოდელის დამუშავების დაწყება")
        model = self.load_or_init_language_model()
        model = self.adapt_for_georgian(model)
        logger.info("ენის მოდელის დამუშავება დასრულებულია")
        return model


class PhonemeModelProcessor:
    """
    ფონემების მოდელის დამუშავების კლასი
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])

    def create_phoneme_model(self):
        """
        ფონემების მოდელის შექმნა
        """
        # აქ იქნება კოდი ფონემების მოდელისთვის
        # დროებითი "dummy" მოდელი, რეალური იმპლემენტაციისთვის შეცვალეთ
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32)
        ).to(self.device)

        logger.info("ფონემების მოდელი შექმნილია")
        return model

    def configure_g2p(self):
        """
        გრაფემა-ფონემა გარდამქმნელის კონფიგურაცია
        """
        # აქ იქნება გრაფემა-ფონემა გარდამქმნელის კონფიგურაცია
        g2p_config = {
            "language": "ka",
            "use_espeak": True,
            "phoneme_dict": {}
        }

        logger.info("გრაფემა-ფონემა გარდამქმნელი კონფიგურირებულია")
        return g2p_config

    def process(self):
        """
        ფონემების მოდელის დამუშავების პროცესი
        """
        logger.info("ფონემების მოდელის დამუშავების დაწყება")
        model = self.create_phoneme_model()
        g2p_config = self.configure_g2p()
        logger.info("ფონემების მოდელის დამუშავება დასრულებულია")
        return model, g2p_config


class XTTSModelInitializer:
    """
    XTTS-v2 მოდელის ინიციალიზაციის კლასი
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])

    def init_from_base_model(self):
        """
        XTTS-v2 მოდელის ინიციალიზაცია არსებული მოდელიდან
        """
        # აქ იქნება კოდი მოდელის ინიციალიზაციისთვის
        # დროებითი "dummy" მოდელი, რეალური იმპლემენტაციისთვის შეცვალეთ
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8)
        ).to(self.device)

        logger.info("XTTS-v2 მოდელი ინიციალიზებულია")
        return model

    def configure_for_georgian(self, model):
        """
        XTTS-v2 მოდელის კონფიგურაცია ქართული ენისთვის
        """
        # აქ იქნება კოდი ქართული ენისთვის კონფიგურაციისთვის
        xtts_config = {
            "language": "ka",
            "speaker_embedding_dim": 512,
            "use_speaker_encoder": True,
            "vocoder": "hifigan",
            "phoneme_language": "ka"
        }

        logger.info("XTTS-v2 მოდელი კონფიგურირებულია ქართული ენისთვის")
        return model, xtts_config

    def process(self):
        """
        XTTS-v2 მოდელის ინიციალიზაციის პროცესი
        """
        logger.info("XTTS-v2 მოდელის ინიციალიზაციის დაწყება")
        model = self.init_from_base_model()
        model, xtts_config = self.configure_for_georgian(model)
        logger.info("XTTS-v2 მოდელის ინიციალიზაცია დასრულებულია")
        return model, xtts_config


class PretrainPipeline:
    """
    XTTS-v2 მოდელის პრეტრეინინგის მთავარი კლასი
    """

    def __init__(self, config_path=None):
        self.config = load_config(config_path)
        setup_logging(self.config)
        setup_directories(self.config)
        set_seed(self.config["seed"])

        self.device = torch.device(self.config["device"])
        self.checkpoint_manager = CheckpointManager(
            self.config["checkpoint_dir"],
            self.config["checkpoint_interval"]
        )

        # კომპონენტების ინიციალიზაცია
        self.language_processor = LanguageModelProcessor(self.config)
        self.phoneme_processor = PhonemeModelProcessor(self.config)
        self.xtts_initializer = XTTSModelInitializer(self.config)

        # შედეგების შესანახი ადგილი
        self.results = {}

    def prepare_language_model(self):
        """
        ენის მოდელის მომზადება
        """
        step_result_path = Path(self.config["output_model_path"]) / "language_model_results.pkl"

        # შევამოწმოთ არსებობს თუ არა შედეგები
        if step_result_path.exists():
            logger.info(f"ენის მოდელის შედეგები უკვე არსებობს: {step_result_path}")
            with open(step_result_path, 'rb') as f:
                language_model = pickle.load(f)
                self.results["language_model"] = language_model
                return language_model

        # თუ არ არსებობს, დავამუშავოთ
        language_model = self.language_processor.process()

        # შედეგების შენახვა
        with open(step_result_path, 'wb') as f:
            pickle.dump(language_model, f)

        self.results["language_model"] = language_model
        return language_model

    def prepare_phoneme_model(self):
        """
        ფონემების მოდელის მომზადება
        """
        step_result_path = Path(self.config["output_model_path"]) / "phoneme_model_results.pkl"

        # შევამოწმოთ არსებობს თუ არა შედეგები
        if step_result_path.exists():
            logger.info(f"ფონემების მოდელის შედეგები უკვე არსებობს: {step_result_path}")
            with open(step_result_path, 'rb') as f:
                results = pickle.load(f)
                self.results["phoneme_model"] = results
                return results

        # თუ არ არსებობს, დავამუშავოთ
        phoneme_model, g2p_config = self.phoneme_processor.process()
        results = {"model": phoneme_model, "g2p_config": g2p_config}

        # შედეგების შენახვა
        with open(step_result_path, 'wb') as f:
            pickle.dump(results, f)

        self.results["phoneme_model"] = results
        return results

    def initialize_xtts_model(self):
        """
        XTTS-v2 მოდელის ინიციალიზაცია
        """
        step_result_path = Path(self.config["output_model_path"]) / "xtts_init_results.pkl"

        # შევამოწმოთ არსებობს თუ არა შედეგები
        if step_result_path.exists():
            logger.info(f"XTTS-v2 მოდელის ინიციალიზაციის შედეგები უკვე არსებობს: {step_result_path}")
            with open(step_result_path, 'rb') as f:
                results = pickle.load(f)
                self.results["xtts_model"] = results
                return results

        # თუ არ არსებობს, დავამუშავოთ
        xtts_model, xtts_config = self.xtts_initializer.process()
        results = {"model": xtts_model, "config": xtts_config}

        # შედეგების შენახვა
        with open(step_result_path, 'wb') as f:
            pickle.dump(results, f)

        self.results["xtts_model"] = results
        return results

    def combine_models(self):
        """
        სამივე მოდელის კომბინირება საბოლოო XTTS-v2 მოდელში
        """
        logger.info("მოდელების კომბინირება დაწყებულია")

        # დავრწმუნდეთ, რომ ყველა საჭირო მოდელი მომზადებულია
        if not all(k in self.results for k in ["language_model", "phoneme_model", "xtts_model"]):
            logger.error("ყველა საჭირო მოდელი არ არის მომზადებული!")
            return None

        # კომბინირება (აქ მხოლოდ მაგალითისთვის მარტივი კომბინირება)
        final_model = {
            "language_model": self.results["language_model"],
            "phoneme_model": self.results["phoneme_model"]["model"],
            "g2p_config": self.results["phoneme_model"]["g2p_config"],
            "xtts_model": self.results["xtts_model"]["model"],
            "xtts_config": self.results["xtts_model"]["config"]
        }

        logger.info("მოდელების კომბინირება დასრულებულია")
        return final_model

    def save_final_model(self, final_model):
        """
        საბოლოო მოდელის შენახვა
        """
        output_path = Path(self.config["output_model_path"]) / "xtts_v2_ka_combined.pkl"
        config_path = Path(self.config["output_model_path"]) / "xtts_v2_ka_config.json"

        # მოდელის შენახვა
        with open(output_path, 'wb') as f:
            pickle.dump(final_model, f)

        # კონფიგურაციის შენახვა
        save_config(self.config, config_path)

        logger.info(f"საბოლოო მოდელი შენახულია: {output_path}")
        logger.info(f"მოდელის კონფიგურაცია შენახულია: {config_path}")

    def run(self):
        """
        მთლიანი პრეტრეინინგის პროცესის გაშვება
        """
        logger.info("===== XTTS-v2 მოდელის პრეტრეინინგის დაწყება =====")
        start_time = time.time()

        try:
            # ეტაპი 1: ენის მოდელის მომზადება
            logger.info("----- ეტაპი 1: ენის მოდელის მომზადება -----")
            language_model = self.prepare_language_model()

            # ეტაპი 2: ფონემების მოდელის მომზადება
            logger.info("----- ეტაპი 2: ფონემების მოდელის მომზადება -----")
            phoneme_results = self.prepare_phoneme_model()

            # ეტაპი 3: XTTS-v2 მოდელის ინიციალიზაცია
            logger.info("----- ეტაპი 3: XTTS-v2 მოდელის ინიციალიზაცია -----")
            xtts_results = self.initialize_xtts_model()

            # ეტაპი 4: მოდელების კომბინირება და შენახვა
            logger.info("----- ეტაპი 4: მოდელების კომბინირება და შენახვა -----")
            final_model = self.combine_models()
            if final_model:
                self.save_final_model(final_model)

            elapsed_time = time.time() - start_time
            logger.info(f"===== XTTS-v2 მოდელის პრეტრეინინგი დასრულებულია =====")
            logger.info(f"მთლიანი დრო: {elapsed_time:.2f} წამი")

            return final_model

        except Exception as e:
            logger.error(f"შეცდომა პრეტრეინინგის პროცესში: {str(e)}", exc_info=True)
            return None


def main():
    """
    მთავარი ფუნქცია, რომელიც აკეთებს არგუმენტების პარსინგს და გაუშვებს პრეტრეინინგს
    """
    parser = argparse.ArgumentParser(description="XTTS-v2 მოდელის პრეტრეინინგი ქართული ენისთვის")
    parser.add_argument("--config", type=str, help="კონფიგურაციის ფაილის გზა")
    args = parser.parse_args()

    pipeline = PretrainPipeline(args.config)
    pipeline.run()


if __name__ == "__main__":
    main()