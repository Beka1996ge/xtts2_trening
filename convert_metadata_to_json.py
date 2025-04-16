#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV მეტადატა ფაილების გარდაქმნა JSON ფორმატში
"""

import os
import csv
import json
import argparse


def convert_csv_to_json(csv_file, json_file):
    """
    CSV ფაილის გარდაქმნა JSON ფორმატში.

    Args:
        csv_file (str): CSV ფაილის მისამართი
        json_file (str): გამომავალი JSON ფაილის მისამართი
    """
    data = []

    # CSV ფაილის წაკითხვა
    with open(csv_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            data.append(row)

    # JSON ფაილის შექმნა
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"გარდაქმნილია {len(data)} ჩანაწერი: {csv_file} -> {json_file}")


def main():
    parser = argparse.ArgumentParser(description="CSV მეტადატა ფაილების გარდაქმნა JSON ფორმატში")
    parser.add_argument("--data_dir", type=str, default="data/split",
                        help="მონაცემთა დირექტორია")

    args = parser.parse_args()

    # დირექტორიების ჩამონათვალი
    dirs = ["train", "val", "test"]

    for dir_name in dirs:
        dir_path = os.path.join(args.data_dir, dir_name)
        if os.path.exists(dir_path):
            csv_file = os.path.join(dir_path, "metadata.csv")
            json_file = os.path.join(dir_path, "metadata.json")

            if os.path.exists(csv_file):
                convert_csv_to_json(csv_file, json_file)
            else:
                print(f"CSV ფაილი ვერ მოიძებნა: {csv_file}")


if __name__ == "__main__":
    main()