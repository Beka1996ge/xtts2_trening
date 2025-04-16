#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XTTS-v2 მოდელისთვის მეტამონაცემების ფაილების შექმნა
ავტორი:
თარიღი: 2025-04-15
"""

import os
import csv
import argparse
from pathlib import Path
import logging
import glob

# ლოგების კონფიგურაცია
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("create_metadata")


def ensure_dir(path):
    """დირექტორიების არსებობის შემოწმება და შექმნა"""
    os.makedirs(path, exist_ok=True)
    return path


def find_embeddings(embeddings_dir):
    """
    ემბედინგების ფაილების მოძებნა

    Args:
        embeddings_dir (str): ემბედინგების დირექტორიის გზა

    Returns:
        list: ემბედინგების file_id-ების სია
    """
    embedding_files = glob.glob(os.path.join(embeddings_dir, "*.npy"))
    file_ids = [os.path.splitext(os.path.basename(f))[0] for f in embedding_files]
    return file_ids


def create_basic_metadata(file_ids, output_dir, split="train", ratio=(0.8, 0.1, 0.1)):
    """
    მარტივი მეტამონაცემების CSV ფაილის შექმნა

    Args:
        file_ids (list): file_id-ების სია
        output_dir (str): შედეგების დირექტორიის გზა
        split (str): ნაკრების ტიპი (train, val, test)
        ratio (tuple): ნაკრებების დაყოფის თანაფარდობა

    Returns:
        str: შექმნილი ფაილის გზა
    """
    # ნაკრებების მიხედვით file_id-ების დაყოფა
    train_size = int(len(file_ids) * ratio[0])
    val_size = int(len(file_ids) * ratio[1])

    if split == "train":
        selected_ids = file_ids[:train_size]
    elif split == "val":
        selected_ids = file_ids[train_size:train_size + val_size]
    elif split == "test":
        selected_ids = file_ids[train_size + val_size:]
    else:
        selected_ids = file_ids

    # შედეგების დირექტორიის მომზადება
    split_dir = ensure_dir(os.path.join(output_dir, "split", split))

    # მეტამონაცემების ფაილის შექმნა
    metadata_path = os.path.join(split_dir, f"{split}_metadata.csv")

    with open(metadata_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file_id", "text", "duration"])

        # მარტივი მაგალითები
        for file_id in selected_ids:
            # არარსებული ტექსტი და ხანგრძლივობა ტესტირებისთვის
            sample_text = f"ეს არის ტექსტის მაგალითი {file_id} ფაილისთვის."
            duration = 3.0  # ხანგრძლივობა წამებში
            writer.writerow([file_id, sample_text, duration])

    logger.info(f"შექმნილია მეტამონაცემების ფაილი {split} ნაკრებისთვის: {metadata_path}")
    logger.info(f"დამატებულია {len(selected_ids)} ჩანაწერი")

    return metadata_path


def main():
    parser = argparse.ArgumentParser(description="XTTS-v2 მოდელისთვის მეტამონაცემების ფაილების შექმნა")
    parser.add_argument("--embeddings_dir", type=str, default="data/embeddings/embeddings",
                        help="ემბედინგების დირექტორიის გზა")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="შედეგების დირექტორიის გზა")
    args = parser.parse_args()

    # ემბედინგების მოძებნა
    file_ids = find_embeddings(args.embeddings_dir)

    if not file_ids:
        logger.warning(f"ემბედინგების ფაილები ვერ მოიძებნა დირექტორიაში {args.embeddings_dir}")
        # შევქმნათ დემო file_id-ები
        file_ids = [f"sample_{i:03d}" for i in range(100)]

    logger.info(f"ნაპოვნია {len(file_ids)} ემბედინგი/ფაილი")

    # მეტამონაცემების ფაილების შექმნა თითოეული ნაკრებისთვის
    create_basic_metadata(file_ids, args.output_dir, "train")
    create_basic_metadata(file_ids, args.output_dir, "val")
    create_basic_metadata(file_ids, args.output_dir, "test")


if __name__ == "__main__":
    main()