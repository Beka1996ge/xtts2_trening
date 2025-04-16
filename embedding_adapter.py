#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ემბედინგების ადაპტერი XTTS-v2 მოდელისთვის
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
import time
from pathlib import Path
from tqdm import tqdm
import shutil

# ლოგების კონფიგურაცია
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("embedding_adapter")


class EmbeddingAdapter:
    """
    ემბედინგების ადაპტერი XTTS-v2 მოდელისთვის
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

        self.data_dir = Path(self.config["data_dir"])
        self.embeddings_dir = self.data_dir.parent / "embeddings" / "embeddings"
        self.output_dir = self.data_dir / "xtts_embeddings"

        # საჭირო დირექტორიების შექმნა
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def get_embeddings_list(self):
        """
        არსებული ემბედინგების სიის მიღება

        Returns:
            list: ემბედინგების ფაილების სია
        """
        if not self.embeddings_dir.exists():
            logger.error(f"ემბედინგების დირექტორია არ არსებობს: {self.embeddings_dir}")
            return []

        embeddings = list(self.embeddings_dir.glob("*.npy"))
        logger.info(f"ნაპოვნია {len(embeddings)} ემბედინგი")

        return embeddings

    def get_metadata_files(self):
        """
        მეტადატას ფაილების მოძებნა ყველა სპლიტისთვის

        Returns:
            dict: მეტადატას ფაილების გზები სპლიტების მიხედვით
        """
        metadata_files = {}

        for split in ["train", "val", "test"]:
            metadata_path = self.data_dir / "split" / split / f"{split}_metadata.csv"
            if metadata_path.exists():
                metadata_files[split] = metadata_path

        return metadata_files

    def adapt_embeddings(self, force=False):
        """
        ემბედინგების ადაპტირება XTTS-v2 ფორმატისთვის

        Args:
            force (bool): თუ True, არსებული ემბედინგების გადაწერა მოხდება

        Returns:
            int: დამუშავებული ემბედინგების რაოდენობა
        """
        logger.info("ემბედინგების ადაპტირების დაწყება")

        # ემბედინგების სიის მიღება
        embeddings = self.get_embeddings_list()

        if not embeddings:
            logger.error("ემბედინგები ვერ მოიძებნა")
            return 0

        # თუ შედეგების დირექტორია ცარიელი არ არის და force არის False
        if not force and any(self.output_dir.iterdir()):
            logger.info("XTTS-v2 ემბედინგები უკვე არსებობს, გამოიყენეთ force=True გადასაწერად")
            return 0

        # ემბედინგების დამუშავება
        processed_count = 0

        for embed_path in tqdm(embeddings, desc="ემბედინგების ადაპტირება"):
            file_id = embed_path.stem
            output_path = self.output_dir / f"{file_id}.npy"

            # თუ ფაილი უკვე არსებობს და force არის False, გამოვტოვოთ
            if not force and output_path.exists():
                continue

            try:
                # ემბედინგის ჩატვირთვა
                embedding = np.load(embed_path)

                # XTTS-v2-სთვის ფორმატის შეცვლა
                # აქ შეგიძლიათ დაამატოთ ნებისმიერი საჭირო ტრანსფორმაცია
                # მაგალითად, გაზომვა, ნორმალიზაცია და ა.შ.

                # ხელოვნური მაგალითისთვის უბრალოდ ვაკოპირებთ
                adapted_embedding = embedding.copy()

                # შედეგების შენახვა
                np.save(output_path, adapted_embedding)
                processed_count += 1

            except Exception as e:
                logger.error(f"შეცდომა ემბედინგის დამუშავებისას {file_id}: {str(e)}")

        logger.info(f"დამუშავებულია {processed_count} ემბედინგი")
        return processed_count

    def update_metadata(self, force=False):
        """
        მეტადატას განახლება ახალი ემბედინგების გზებით

        Args:
            force (bool): თუ True, არსებული მეტადატას გადაწერა მოხდება

        Returns:
            int: განახლებული მეტადატა ფაილების რაოდენობა
        """
        logger.info("მეტადატას განახლების დაწყება")

        # მეტადატას ფაილების მოძებნა
        metadata_files = self.get_metadata_files()

        if not metadata_files:
            logger.error("მეტადატას ფაილები ვერ მოიძებნა")
            return 0

        # განახლებული მეტადატა ფაილების რაოდენობა
        updated_count = 0

        for split, metadata_path in metadata_files.items():
            try:
                # არსებული მეტადატას ჩატვირთვა
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # სათაურის გამოყოფა
                header = lines[0].strip()

                # თუ ემბედინგის სვეტი უკვე არსებობს, შევამოწმოთ force
                if "embedding_path" in header and not force:
                    logger.info(f"მეტადატა ფაილი {split} უკვე შეიცავს ემბედინგის ინფორმაციას")
                    continue

                # ახალი სათაურის შექმნა
                if "embedding_path" not in header:
                    new_header = header + ",embedding_path"
                else:
                    new_header = header

                # ახალი მონაცემების შექმნა
                new_lines = [new_header + "\n"]

                # დანარჩენი ხაზების დამუშავება
                for line in lines[1:]:
                    parts = line.strip().split(",")
                    file_id = parts[0]  # დავუშვათ, file_id პირველ სვეტშია

                    # ახალი ემბედინგის გზის დამატება
                    embedding_path = str(self.output_dir / f"{file_id}.npy")

                    if "embedding_path" in header:
                        # ემბედინგის სვეტის ინდექსის პოვნა
                        emb_idx = header.split(",").index("embedding_path")
                        # სვეტის განახლება
                        parts[emb_idx] = embedding_path
                    else:
                        # ახალი სვეტის დამატება
                        parts.append(embedding_path)

                    new_lines.append(",".join(parts) + "\n")

                # შედეგების ფაილის გზა
                output_path = metadata_path.parent / f"{split}_metadata_updated.csv"

                # შედეგების შენახვა
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)

                # ძველი ფაილის შენახვა და ახალი გადარქმევა
                backup_path = metadata_path.parent / f"{split}_metadata_backup.csv"
                shutil.copy2(metadata_path, backup_path)
                shutil.move(output_path, metadata_path)

                logger.info(f"მეტადატა ფაილი განახლებულია: {metadata_path}")
                updated_count += 1

            except Exception as e:
                logger.error(f"შეცდომა მეტადატას განახლებისას {split}: {str(e)}")

        logger.info(f"განახლებულია {updated_count} მეტადატა ფაილი")
        return updated_count

    def run(self, force=False):
        """
        სრული ადაპტაციის პროცესის გაშვება

        Args:
            force (bool): თუ True, არსებული ფაილების გადაწერა მოხდება

        Returns:
            tuple: (დამუშავებული ემბედინგების რაოდენობა, განახლებული მეტადატა ფაილების რაოდენობა)
        """
        logger.info("ემბედინგების ადაპტაციის პროცესის დაწყება")
        start_time = time.time()

        # ემბედინგების ადაპტირება
        processed_embeddings = self.adapt_embeddings(force)

        # მეტადატას განახლება
        updated_metadata = self.update_metadata(force)

        elapsed_time = time.time() - start_time
        logger.info(f"ემბედინგების ადაპტაციის პროცესი დასრულებულია: "
                    f"{processed_embeddings} ემბედინგი, {updated_metadata} მეტადატა ფაილი")
        logger.info(f"მთლიანი დრო: {elapsed_time:.2f} წამი")

        return processed_embeddings, updated_metadata


def main():
    """
    მთავარი ფუნქცია
    """
    parser = argparse.ArgumentParser(description="ემბედინგების ადაპტირება XTTS-v2 მოდელისთვის")
    parser.add_argument("--config", type=str, default="xtts_pretrain_config.json",
                        help="კონფიგურაციის ფაილის გზა")
    parser.add_argument("--force", action="store_true",
                        help="არსებული ფაილების გადაწერა")
    args = parser.parse_args()

    # ადაპტაციის პროცესის გაშვება
    adapter = EmbeddingAdapter(args.config)
    adapter.run(args.force)


if __name__ == "__main__":
    main()