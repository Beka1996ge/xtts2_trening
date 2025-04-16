#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XTTS-v2 მოდელის პრეტრეინინგის გამშვები სკრიპტი
ავტორი:
თარიღი: 2025-04-15
"""

import os
import sys
import json
import logging
import argparse
import torch
import time
from pathlib import Path
from datetime import datetime
from pretrain_xtts_model import PretrainPipeline
from data_loader1 import load_data_for_pretrain

# ლოგების კონფიგურაცია
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("run_xtts_pretrain")


def setup_logging(config):
    """
    ლოგების კონფიგურაცია, ყველა ლოგის ფაილში შენახვა
    """
    log_dir = Path(config["log_dir"])
    log_dir.mkdir(exist_ok=True, parents=True)

    log_file = log_dir / f"pretrain_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info(f"ლოგების ფაილი: {log_file}")
    logger.info(f"კონფიგურაცია: {config}")


def run_language_model_step(config_path):
    """
    ენის მოდელის ეტაპის გაშვება

    Args:
        config_path (str): კონფიგურაციის ფაილის გზა
    """
    logger.info("===== ენის მოდელის ეტაპის გაშვება =====")

    # კონფიგურაციის ჩატვირთვა
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # ლოგების კონფიგურაცია
    setup_logging(config)

    try:
        # პრეტრეინინგის კლასის შექმნა
        pipeline = PretrainPipeline(config_path)

        # ენის მოდელის მომზადება
        language_model = pipeline.prepare_language_model()

        logger.info("ენის მოდელის ეტაპი დასრულებულია")

    except Exception as e:
        logger.error(f"შეცდომა ენის მოდელის ეტაპზე: {str(e)}", exc_info=True)


def run_xtts_init_step(config_path):
    """
    XTTS-ის ინიციალიზაციის ეტაპის გაშვება

    Args:
        config_path (str): კონფიგურაციის ფაილის გზა
    """
    logger.info("===== XTTS-ის ინიციალიზაციის ეტაპის გაშვება =====")

    # კონფიგურაციის ჩატვირთვა
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # ლოგების კონფიგურაცია
    setup_logging(config)

    try:
        # პრეტრეინინგის კლასის შექმნა
        pipeline = PretrainPipeline(config_path)

        # XTTS მოდელის ინიციალიზაცია
        xtts_results = pipeline.initialize_xtts_model()

        logger.info("XTTS-ის ინიციალიზაციის ეტაპი დასრულებულია")

    except Exception as e:
        logger.error(f"შეცდომა XTTS-ის ინიციალიზაციის ეტაპზე: {str(e)}", exc_info=True)


def run_full_pretrain_step(config_path):
    """
    სრული პრეტრეინინგის ეტაპის გაშვება

    Args:
        config_path (str): კონფიგურაციის ფაილის გზა
    """
    logger.info("===== სრული პრეტრეინინგის ეტაპის გაშვება =====")

    # კონფიგურაციის ჩატვირთვა
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # ლოგების კონფიგურაცია
    setup_logging(config)

    try:
        # მონაცემების მომზადება
        logger.info("----- 1. მონაცემების მომზადება -----")
        try:
            data_manager, train_loader, val_loader, test_loader = load_data_for_pretrain(config_path)

            # მონაცემების სტატისტიკა
            stats = data_manager.get_metadata_stats()
            logger.info(f"მონაცემების სტატისტიკა: {stats}")
        except Exception as data_error:
            logger.warning(f"მონაცემების მომზადებისას პრობლემა: {str(data_error)}")
            logger.warning("გამოყენებული იქნება dummy მონაცემები ტესტირებისთვის")
            train_loader, val_loader, test_loader = None, None, None

        # პრეტრეინინგის კლასის შექმნა
        pipeline = PretrainPipeline(config_path)

        # ყველა ეტაპის გაშვება
        language_model = pipeline.prepare_language_model()
        phoneme_results = pipeline.prepare_phoneme_model()
        xtts_results = pipeline.initialize_xtts_model()

        # მოდელების კომბინირება
        final_model = pipeline.combine_models()
        if final_model:
            pipeline.save_final_model(final_model)

        logger.info("სრული პრეტრეინინგის ეტაპი დასრულებულია")

    except Exception as e:
        logger.error(f"შეცდომა სრული პრეტრეინინგის ეტაპზე: {str(e)}", exc_info=True)


def run_pretrain_process(config_path, step=None):
    """
    XTTS-v2 მოდელის პრეტრეინინგის პროცესის გაშვება

    Args:
        config_path (str): კონფიგურაციის ფაილის გზა
        step (str): რომელი ეტაპი უნდა გაეშვას (language_model, xtts_init, full_pretrain)
    """
    logger.info("===== XTTS-v2 მოდელის პრეტრეინინგის პროცესის დაწყება =====")
    start_time = time.time()

    # შესაბამისი ეტაპის გაშვება
    if step == "language_model":
        run_language_model_step(config_path)
    elif step == "xtts_init":
        run_xtts_init_step(config_path)
    elif step == "full_pretrain":
        run_full_pretrain_step(config_path)
    else:
        # თუ ეტაპი არ არის მითითებული, ყველა ეტაპი ერთად გაეშვება
        run_full_pretrain_step(config_path)

    elapsed_time = time.time() - start_time
    logger.info(f"===== XTTS-v2 მოდელის პრეტრეინინგის პროცესი დასრულებულია =====")
    logger.info(f"მთლიანი დრო: {elapsed_time:.2f} წამი")


def main():
    """
    მთავარი ფუნქცია
    """
    parser = argparse.ArgumentParser(description="XTTS-v2 მოდელის პრეტრეინინგის გაშვება")
    parser.add_argument("--config", type=str, default="xtts_pretrain_config.json",
                        help="კონფიგურაციის ფაილის გზა")
    parser.add_argument("--step", type=str, choices=["language_model", "xtts_init", "full_pretrain"],
                        help="რომელი ეტაპი უნდა გაეშვას")
    args = parser.parse_args()

    # პრეტრეინინგის პროცესის გაშვება
    run_pretrain_process(args.config, args.step)


if __name__ == "__main__":
    main()