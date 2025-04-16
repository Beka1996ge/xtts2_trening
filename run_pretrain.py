#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XTTS-v2 მოდელის პრეტრეინინგის მთავარი გამშვები სკრიპტი
ავტორი:
თარიღი: 2025-04-15
"""

import os
import sys
import json
import logging
import argparse
import time
import pickle
from pathlib import Path
from datetime import datetime

# ლოგების კონფიგურაცია
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("run_pretrain")


class PretrainRunner:
    """
    XTTS-v2 მოდელის პრეტრეინინგის მთავარი კლასი
    """

    def __init__(self, config_path, force=False):
        """
        ინიციალიზაცია

        Args:
            config_path (str): კონფიგურაციის ფაილის გზა
            force (bool): თუ True, ყველა ეტაპი ხელახლა გაეშვება
        """
        self.config_path = config_path
        self.force = force

        # კონფიგურაციის ჩატვირთვა
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # ლოგების კონფიგურაცია
        self.setup_logging()

        # საჭირო დირექტორიების შექმნა
        self.output_dir = Path(self.config["output_model_path"])
        self.checkpoint_dir = Path(self.config["checkpoint_dir"])
        self.log_dir = Path(self.config["log_dir"])

        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)

        # პროგრესის ფაილი
        self.progress_file = self.output_dir / "pretrain_progress.json"

        # პროგრესის ჩატვირთვა ან ინიციალიზაცია
        self.progress = self.load_progress()

    def setup_logging(self):
        """
        ლოგების კონფიგურაცია, ყველა ლოგის ფაილში შენახვა
        """
        log_dir = Path(self.config["log_dir"])
        log_dir.mkdir(exist_ok=True, parents=True)

        log_file = log_dir / f"pretrain_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        logger.info(f"ლოგების ფაილი: {log_file}")
        logger.info(f"კონფიგურაცია: {self.config}")

    def load_progress(self):
        """
        პროგრესის ჩატვირთვა

        Returns:
            dict: პროგრესის მდგომარეობა
        """
        if self.progress_file.exists() and not self.force:
            logger.info(f"პროგრესის ჩატვირთვა ფაილიდან: {self.progress_file}")
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                return progress
            except Exception as e:
                logger.error(f"შეცდომა პროგრესის ჩატვირთვისას: {str(e)}")

        # საწყისი პროგრესის მდგომარეობა
        progress = {
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "steps": {
                "prepare_embeddings": False,
                "language_model": False,
                "phoneme_model": False,
                "xtts_init": False,
                "full_pretrain": False
            },
            "checkpoints": {}
        }

        return progress

    def save_progress(self):
        """
        პროგრესის შენახვა
        """
        self.progress["last_updated"] = datetime.now().isoformat()

        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, ensure_ascii=False, indent=2)

        logger.info(f"პროგრესი შენახულია: {self.progress_file}")

    def run_command(self, command):
        """
        ბრძანების გაშვება

        Args:
            command (str): გასაშვები ბრძანება

        Returns:
            int: გასვლის კოდი
        """
        logger.info(f"ბრძანების გაშვება: {command}")
        return os.system(command)

    def run_embedding_adapter(self):
        """
        ემბედინგების ადაპტერის გაშვება

        Returns:
            bool: წარმატების ინდიკატორი
        """
        if self.progress["steps"]["prepare_embeddings"] and not self.force:
            logger.info("ემბედინგების ადაპტერი უკვე გაშვებულია, გამოტოვება")
            return True

        logger.info("ემბედინგების ადაპტერის გაშვება")

        force_flag = "--force" if self.force else ""
        command = f"python embedding_adapter.py --config {self.config_path} {force_flag}"
        exit_code = self.run_command(command)

        if exit_code == 0:
            self.progress["steps"]["prepare_embeddings"] = True
            self.save_progress()
            return True
        else:
            logger.error(f"ემბედინგების ადაპტერის გაშვებისას მოხდა შეცდომა, გასვლის კოდი: {exit_code}")
            return False

    def run_phoneme_processor(self):
        """
        ფონემების პროცესორის გაშვება

        Returns:
            bool: წარმატების ინდიკატორი
        """
        if self.progress["steps"]["phoneme_model"] and not self.force:
            logger.info("ფონემების პროცესორი უკვე გაშვებულია, გამოტოვება")
            return True

        logger.info("ფონემების პროცესორის გაშვება")

        command = f"python phoneme_processor.py --config {self.config_path}"
        exit_code = self.run_command(command)

        if exit_code == 0:
            self.progress["steps"]["phoneme_model"] = True
            self.save_progress()
            return True
        else:
            logger.error(f"ფონემების პროცესორის გაშვებისას მოხდა შეცდომა, გასვლის კოდი: {exit_code}")
            return False

    def run_pretrain_pipeline(self):
        """
        პრეტრეინინგის პროცესის გაშვება

        Returns:
            bool: წარმატების ინდიკატორი
        """
        # ენის მოდელის ეტაპი
        if not self.progress["steps"]["language_model"] or self.force:
            logger.info("ენის მოდელის ეტაპის გაშვება")

            # ამ კონტექსტში, run_xtts_pretrain.py გაუშვებს ენის მოდელის ეტაპს
            command = f"python run_xtts_pretrain.py --config {self.config_path} --step language_model"
            exit_code = self.run_command(command)

            if exit_code == 0:
                self.progress["steps"]["language_model"] = True
                self.save_progress()
            else:
                logger.error(f"ენის მოდელის ეტაპის გაშვებისას მოხდა შეცდომა, გასვლის კოდი: {exit_code}")
                return False
        else:
            logger.info("ენის მოდელის ეტაპი უკვე გაშვებულია, გამოტოვება")

        # XTTS-ის ინიციალიზაციის ეტაპი
        if not self.progress["steps"]["xtts_init"] or self.force:
            logger.info("XTTS-ის ინიციალიზაციის ეტაპის გაშვება")

            command = f"python run_xtts_pretrain.py --config {self.config_path} --step xtts_init"
            exit_code = self.run_command(command)

            if exit_code == 0:
                self.progress["steps"]["xtts_init"] = True
                self.save_progress()
            else:
                logger.error(f"XTTS-ის ინიციალიზაციის ეტაპის გაშვებისას მოხდა შეცდომა, გასვლის კოდი: {exit_code}")
                return False
        else:
            logger.info("XTTS-ის ინიციალიზაციის ეტაპი უკვე გაშვებულია, გამოტოვება")

        # სრული პრეტრეინინგის ეტაპი
        if not self.progress["steps"]["full_pretrain"] or self.force:
            logger.info("სრული პრეტრეინინგის ეტაპის გაშვება")

            command = f"python run_xtts_pretrain.py --config {self.config_path} --step full_pretrain"
            exit_code = self.run_command(command)

            if exit_code == 0:
                self.progress["steps"]["full_pretrain"] = True
                self.save_progress()
            else:
                logger.error(f"სრული პრეტრეინინგის ეტაპის გაშვებისას მოხდა შეცდომა, გასვლის კოდი: {exit_code}")
                return False
        else:
            logger.info("სრული პრეტრეინინგის ეტაპი უკვე გაშვებულია, გამოტოვება")

        return True

    def run(self):
        """
        მთლიანი პროცესის გაშვება

        Returns:
            bool: წარმატების ინდიკატორი
        """
        logger.info("===== XTTS-v2 მოდელის პრეტრეინინგის პროცესის დაწყება =====")
        start_time = time.time()

        try:
            # 1. ემბედინგების ადაპტერის გაშვება
            logger.info("----- 1. ემბედინგების ადაპტერის გაშვება -----")
            if not self.run_embedding_adapter():
                return False

            # 2. ფონემების პროცესორის გაშვება
            logger.info("----- 2. ფონემების პროცესორის გაშვება -----")
            if not self.run_phoneme_processor():
                return False

            # 3. პრეტრეინინგის პროცესის გაშვება
            logger.info("----- 3. პრეტრეინინგის პროცესის გაშვება -----")
            if not self.run_pretrain_pipeline():
                return False

            elapsed_time = time.time() - start_time
            logger.info(f"===== XTTS-v2 მოდელის პრეტრეინინგის პროცესი დასრულებულია =====")
            logger.info(f"მთლიანი დრო: {elapsed_time:.2f} წამი")

            return True

        except Exception as e:
            logger.error(f"შეცდომა პრეტრეინინგის პროცესში: {str(e)}", exc_info=True)
            return False


def main():
    """
    მთავარი ფუნქცია
    """
    parser = argparse.ArgumentParser(description="XTTS-v2 მოდელის პრეტრეინინგის გაშვება")
    parser.add_argument("--config", type=str, default="xtts_pretrain_config.json",
                        help="კონფიგურაციის ფაილის გზა")
    parser.add_argument("--force", action="store_true",
                        help="ყველა ეტაპის ხელახლა გაშვება")
    parser.add_argument("--only", type=str,
                        choices=["embeddings", "phonemes", "language", "xtts_init", "full_pretrain"],
                        help="მხოლოდ კონკრეტული ეტაპის გაშვება")
    args = parser.parse_args()

    runner = PretrainRunner(args.config, args.force)

    # თუ მხოლოდ კონკრეტული ეტაპი უნდა გაეშვას
    if args.only:
        if args.only == "embeddings":
            runner.run_embedding_adapter()
        elif args.only == "phonemes":
            runner.run_phoneme_processor()
        elif args.only == "language":
            runner.run_command(f"python run_xtts_pretrain.py --config {args.config} --step language_model")
            runner.progress["steps"]["language_model"] = True
            runner.save_progress()
        elif args.only == "xtts_init":
            runner.run_command(f"python run_xtts_pretrain.py --config {args.config} --step xtts_init")
            runner.progress["steps"]["xtts_init"] = True
            runner.save_progress()
        elif args.only == "full_pretrain":
            runner.run_command(f"python run_xtts_pretrain.py --config {args.config} --step full_pretrain")
            runner.progress["steps"]["full_pretrain"] = True
            runner.save_progress()
    else:
        # სრული პროცესის გაშვება
        runner.run()


if __name__ == "__main__":
    main()