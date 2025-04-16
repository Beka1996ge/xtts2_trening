#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XTTS-v2 მოდელის ტრენინგის გამშვები სკრიპტი ქართული ენისთვის
"""

import os
import sys
import json
import argparse
import torch
from datetime import datetime

# ლოკალური მოდულების იმპორტი
from xtts_config import XTTSConfig
from train_xtts_model import XTTSTrainer
from lr_scheduler import NoamLRScheduler, WarmupCosineLR, CyclicLR


def setup_config(args):
    """
    კონფიგურაციის მომზადება ბრძანების ხაზის არგუმენტების მიხედვით

    Args:
        args: ბრძანების ხაზის არგუმენტები

    Returns:
        str, dict: განახლებული კონფიგურაციის ფაილის მისამართი და კონფიგურაციის მონაცემები
    """
    # კონფიგურაციის ჩატვირთვა
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # CLI არგუმენტების მიხედვით კონფიგურაციის განახლება
    if args.batch_size:
        config['batch_size'] = args.batch_size

    if args.learning_rate:
        config['learning_rate'] = args.learning_rate

    if args.epochs:
        config['epochs'] = args.epochs

    if args.grad_accum_steps:
        config['grad_accum_steps'] = args.grad_accum_steps

    # ექსპერიმენტის სახელის განსაზღვრა
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config['model_name']}_{timestamp}"

    # ახალი კონფიგურაციის ფაილის შექმნა
    os.makedirs("configs", exist_ok=True)
    output_config_path = os.path.join("configs", f"{experiment_name}.json")

    with open(output_config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return output_config_path, config


def validate_environment():
    """
    გარემოს ვალიდაცია ტრენინგის დაწყებამდე

    Returns:
        bool: არის თუ არა გარემო ვალიდური
    """
    # CUDA-ს ხელმისაწვდომობის შემოწმება
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"CUDA ხელმისაწვდომია. მოწყობილობა: {torch.cuda.get_device_name(0)}")
        print(f"CUDA ვერსია: {torch.version.cuda}")

        # მეხსიერების ინფორმაცია
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"GPU მეხსიერება: {gpu_mem:.2f} GB")

        # იკითხე თავისუფალი და დაკავებული მეხსიერება
        free_mem = torch.cuda.memory_reserved(0) / 1024 ** 3
        allocated_mem = torch.cuda.memory_allocated(0) / 1024 ** 3
        print(f"დაკავებული მეხსიერება: {allocated_mem:.2f} GB")
        print(f"რეზერვირებული მეხსიერება: {free_mem:.2f} GB")
    else:
        print("გაფრთხილება: CUDA არ არის ხელმისაწვდომი. ტრენინგი გაეშვება CPU-ზე, რაც მნიშვნელოვნად შეანელებს პროცესს.")

    # PyTorch-ის ვერსიის შემოწმება
    print(f"PyTorch ვერსია: {torch.__version__}")

    # დირექტორიების შემოწმება
    required_dirs = [
        "data/split/train",
        "data/split/val",
        "data/embeddings/embeddings"
    ]

    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"შეცდომა: დირექტორია {directory} ვერ მოიძებნა!")
            return False

    # მეტადატა ფაილების შემოწმება
    required_files = [
        "data/split/train/metadata.json",
        "data/split/val/metadata.json"
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"შეცდომა: ფაილი {file_path} ვერ მოიძებნა!")
            return False

    return True


def update_trainer_scheduler(trainer, args, config):
    """
    ტრენერის სასწავლო სიჩქარის დამგეგმავის განახლება

    Args:
        trainer: XTTSTrainer ობიექტი
        args: ბრძანების ხაზის არგუმენტები
        config: კონფიგურაციის მონაცემები

    Returns:
        XTTSTrainer: განახლებული ტრენერი
    """
    # სასწავლო სიჩქარე და ეპოქები კონფიგურაციიდან
    learning_rate = config.get("learning_rate", 2e-4)
    min_learning_rate = config.get("min_learning_rate", 1e-5)
    epochs = config.get("epochs", 100)

    # სასწავლო სიჩქარის დამგეგმავის შექმნა
    if args.scheduler == "noam":
        model_dim = config.get("model_params", {}).get("hidden_channels", 192)
        trainer.scheduler = NoamLRScheduler(
            optimizer=trainer.optimizer,
            model_dim=model_dim,
            warmup_steps=4000,
            min_lr=min_learning_rate
        )
        print(f"გამოიყენება Noam სასწავლო სიჩქარის დამგეგმავი, მოდელის განზომილება: {model_dim}")
    elif args.scheduler == "cosine":
        # დაახლოებითი ნაბიჯების რაოდენობა ეპოქაში
        steps_per_epoch = 1000  # ეს შეცვალეთ თქვენი მონაცემების ზომის მიხედვით
        max_steps = epochs * steps_per_epoch
        trainer.scheduler = WarmupCosineLR(
            optimizer=trainer.optimizer,
            max_steps=max_steps,
            warmup_steps=2000,
            min_lr=min_learning_rate
        )
        print(f"გამოიყენება Cosine სასწავლო სიჩქარის დამგეგმავი, სავარაუდო ნაბიჯები: {max_steps}")
    elif args.scheduler == "cyclic":
        trainer.scheduler = CyclicLR(
            optimizer=trainer.optimizer,
            base_lr=min_learning_rate,
            max_lr=learning_rate,
            step_size_up=2000,
            mode='triangular2'
        )
        print("გამოიყენება Cyclic სასწავლო სიჩქარის დამგეგმავი")
    else:
        print(f"გამოიყენება სტანდარტული სასწავლო სიჩქარის დამგეგმავი: {args.scheduler}")

    return trainer


def main():
    """მთავარი გამშვები ფუნქცია"""
    parser = argparse.ArgumentParser(description="XTTS-v2 მოდელის ტრენინგი ქართული ენისთვის")

    parser.add_argument("--config", type=str, required=True,
                        help="კონფიგურაციის ფაილის მისამართი")

    parser.add_argument("--resume", type=str,
                        help="ჩექფოინტი ტრენინგის გასაგრძელებლად")

    parser.add_argument("--batch_size", type=int,
                        help="ბატჩის ზომა")

    parser.add_argument("--learning_rate", type=float,
                        help="სასწავლო სიჩქარე")

    parser.add_argument("--epochs", type=int,
                        help="ეპოქების რაოდენობა")

    parser.add_argument("--grad_accum_steps", type=int,
                        help="გრადიენტის აკუმულაციის ნაბიჯები")

    parser.add_argument("--scheduler", type=str, choices=["cosine", "noam", "cyclic", "default"], default="default",
                        help="სასწავლო სიჩქარის დამგეგმავი")

    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None,
                        help="მოწყობილობა ტრენინგისთვის (cuda/cpu)")

    args = parser.parse_args()

    # გარემოს ვალიდაცია
    if not validate_environment():
        print("გარემოს ვალიდაცია ვერ მოხერხდა. ტრენინგი ჩერდება.")
        return

    # კონფიგურაციის მომზადება
    config_path, config = setup_config(args)

    print(f"ტრენინგი იწყება კონფიგურაციით: {config_path}")
    print(f"პარამეტრები:")
    print(f"  ბატჩის ზომა: {config['batch_size']}")
    print(f"  სასწავლო სიჩქარე: {config['learning_rate']}")
    print(f"  ეპოქების რაოდენობა: {config['epochs']}")
    print(f"  სასწავლო სიჩქარის დამგეგმავი: {args.scheduler}")

    # მოწყობილობის განსაზღვრა
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"ტრენინგი გაეშვება მოწყობილობაზე: {device}")

    # ტრენინგის დაწყება
    trainer = XTTSTrainer(
        config_path=config_path,
        resume_checkpoint=args.resume
    )

    # სასწავლო სიჩქარის დამგეგმავის შეცვლა (თუ მითითებულია CLI პარამეტრებში)
    if args.scheduler != "default":
        trainer = update_trainer_scheduler(trainer, args, config)

    # ტრენინგის გაშვება
    trainer.train()


if __name__ == "__main__":
    main()