#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
import sys
import time
import shutil
import pandas as pd
from tqdm import tqdm


def install_dependencies():
    """საჭირო პითონის პაკეტების ინსტალაცია"""
    print("საჭირო პაკეტების ინსტალაცია...")

    packages = [
        "librosa",
        "soundfile",
        "pydub",
        "pandas",
        "numpy",
        "scikit-learn",
        "tqdm"
    ]

    for package in packages:
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], check=True)
            print(f"{package} წარმატებით დაინსტალირდა")
        except subprocess.CalledProcessError as e:
            print(f"შეცდომა {package}-ის ინსტალაციისას: {str(e)}")
            print("გაფრთხილება: შეიძლება ყველა ფუნქციონალი არ იმუშაოს სრულად")

    print("დამოკიდებულებების ინსტალაცია დასრულებულია!")


def generate_metadata(cv_corpus_dir, force=False):
    """მეტადატას გენერაცია თუ არ არსებობს ან ცარიელია"""
    clips_dir = os.path.join(cv_corpus_dir, "clips")
    validated_tsv = os.path.join(cv_corpus_dir, "validated.tsv")
    generated_metadata = os.path.join(cv_corpus_dir, "generated_metadata.csv")
    sentences_file = os.path.join(cv_corpus_dir, "validated_sentences.tsv")

    if not os.path.exists(sentences_file):
        sentences_file = None

    # შევამოწმოთ არსებობს თუ არა ვალიდური მეტადატა
    valid_metadata_exists = False
    if os.path.exists(validated_tsv) and not force:
        try:
            df = pd.read_csv(validated_tsv, sep='\t')
            if len(df) > 0 and 'path' in df.columns and 'sentence' in df.columns:
                valid_metadata_exists = True
                print(f"არსებული მეტადატა ფაილი ვალიდურია და შეიცავს {len(df)} ჩანაწერს")
        except Exception as e:
            print(f"არსებული მეტადატა ფაილი არავალიდურია: {str(e)}")

    # თუ არ არსებობს ვალიდური მეტადატა, დავაგენერიროთ
    if not valid_metadata_exists or force:
        print("მეტადატა ფაილი არ არსებობს ან არავალიდურია. ვაგენერირებთ ახალს...")

        from create_metadata import generate_metadata_for_clips
        metadata_df = generate_metadata_for_clips(clips_dir, generated_metadata, sentences_file)

        # თუ წარმატებით დაგენერირდა, შევქმნათ/განვაახლოთ validated.tsv
        if len(metadata_df) > 0:
            metadata_df.to_csv(validated_tsv, sep='\t', index=False)
            print(f"განახლებულია validated.tsv {len(metadata_df)} ჩანაწერით")
            return validated_tsv
        else:
            print("გაფრთხილება: დაგენერირებული მეტადატა ცარიელია")
            return None

    return validated_tsv


def process_audio_files(audio_dir, clips_tsv, output_dir):
    """აუდიო ფაილების დამუშავება"""
    from audio_processing import process_all_files

    print("აუდიო ფაილების დამუშავება...")
    os.makedirs(output_dir, exist_ok=True)

    processed_df = process_all_files(audio_dir, output_dir, clips_tsv)
    print(f"დამუშავებული ფაილების რაოდენობა: {len(processed_df)}")

    return processed_df


def segment_audio_files(processed_dir, processed_metadata, output_dir):
    """აუდიო ფაილების სეგმენტაცია"""
    from audio_segmentation import segment_all_files

    print("აუდიო ფაილების სეგმენტაცია...")
    os.makedirs(output_dir, exist_ok=True)

    segments_df = segment_all_files(processed_dir, processed_metadata, output_dir)
    print(f"სეგმენტების რაოდენობა: {len(segments_df)}")

    return segments_df


def split_dataset(data_dir, clips_tsv, segments_dir, segments_metadata, output_dir):
    """მონაცემთა ნაკრების დაყოფა"""
    from data_splitting import load_speaker_info, split_data_balanced, copy_files_to_split_directories

    print("მონაცემთა ნაკრების დაყოფა...")
    os.makedirs(output_dir, exist_ok=True)

    # მონაცემების ჩატვირთვა
    segments_df = pd.read_csv(segments_metadata)
    speaker_info = load_speaker_info(clips_tsv)

    # მონაცემების დაყოფა
    train_df, val_df, test_df = split_data_balanced(segments_df, speaker_info)

    # ფაილების კოპირება შესაბამის დირექტორიებში
    copy_files_to_split_directories(train_df, val_df, test_df, segments_dir, output_dir)

    print("მონაცემთა დაყოფა დასრულებულია!")


def create_xtts_config(output_dir):
    """XTTS-ისთვის საჭირო კონფიგურაციის ფაილის შექმნა"""
    print("XTTS კონფიგურაციის ფაილის შექმნა...")

    config = {
        "output_path": "xtts_output",
        "model": {
            "language": "ka",
            "run_name": "xtts_ka"
        },
        "audio": {
            "sample_rate": 16000
        },
        "datasets": [
            {
                "name": "common_voice_ka",
                "path": os.path.join(output_dir, "train"),
                "meta_file_train": "metadata_xtts.csv",
                "meta_file_val": "../val/metadata_xtts.csv"
            }
        ],
        "trainer": {
            "max_epochs": 1000,
            "batch_size": 16,
            "eval_batch_size": 8,
            "gradient_clip": 5.0,
            "iterations_per_checkpoint": 1000,
            "target_loss": "mse",
            "grad_accum": 1
        }
    }

    import json
    config_path = os.path.join(output_dir, "xtts_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"კონფიგურაციის ფაილი შეიქმნა: {config_path}")


def main():
    """მთავარი ფუნქცია მთლიანი პროცესის გასაშვებად"""
    parser = argparse.ArgumentParser(description="XTTS-v2 ქართული ენის მონაცემების მომზადება")
    parser.add_argument("--base_dir", default="data", help="საბაზისო დირექტორია მონაცემებისთვის")
    parser.add_argument("--cv_corpus_dir", default="cv-corpus/ka", help="Common Voice ლოკალური დირექტორია")
    parser.add_argument("--regenerate_metadata", action="store_true", help="ხელახლა დააგენერირე მეტადატა ფაილი")
    parser.add_argument("--skip_processing", action="store_true", help="გამოტოვე აუდიოს დამუშავების ეტაპი")
    parser.add_argument("--skip_segmentation", action="store_true", help="გამოტოვე სეგმენტაციის ეტაპი")
    parser.add_argument("--skip_splitting", action="store_true", help="გამოტოვე დაყოფის ეტაპი")

    args = parser.parse_args()

    # დირექტორიების შექმნა
    base_dir = args.base_dir
    cv_corpus_dir = os.path.join(base_dir, args.cv_corpus_dir)
    processed_dir = os.path.join(base_dir, "processed/audio")
    segments_dir = os.path.join(base_dir, "segmented")
    split_dir = os.path.join(base_dir, "split")

    start_time = time.time()

    # საჭირო პაკეტების ინსტალაცია
    install_dependencies()

    # ლოკალური მონაცემების შემოწმება
    audio_dir = os.path.join(cv_corpus_dir, "clips")

    if not os.path.exists(audio_dir):
        print(f"შეცდომა: აუდიო დირექტორია {audio_dir} ვერ მოიძებნა.")
        sys.exit(1)

    # მეტადატა ფაილების გენერაცია თუ საჭიროა
    clips_tsv = generate_metadata(cv_corpus_dir, force=args.regenerate_metadata)

    if not clips_tsv:
        print("ვერ მოხერხდა ვალიდური მეტადატას მოძიება ან გენერაცია. პროცესი წყდება.")
        sys.exit(1)

    print(f"ნაპოვნია Common Voice მონაცემები: {cv_corpus_dir}")
    print(f"გამოყენებული მეტადატა ფაილი: {clips_tsv}")
    print(f"აუდიო ფაილების დირექტორია: {audio_dir}")

    # ეტაპი 1: აუდიო ფაილების დამუშავება
    if not args.skip_processing:
        print("\n===== ეტაპი 1: აუდიო ფაილების დამუშავება =====")
        processed_metadata = os.path.join(processed_dir, "processed_clips.csv")
        process_audio_files(audio_dir, clips_tsv, processed_dir)
    else:
        print("\n===== ეტაპი 1: აუდიოს დამუშავება გამოტოვებულია =====")
        processed_metadata = os.path.join(processed_dir, "processed_clips.csv")

    # შევამოწმოთ შეიქმნა თუ არა მეტადატა ფაილი
    if not os.path.exists(processed_metadata):
        print(f"შეცდომა: დამუშავებული მეტადატა ფაილი {processed_metadata} ვერ მოიძებნა.")
        print("დარწმუნდით რომ აუდიო ფაილების დამუშავების ეტაპი წარმატებით დასრულდა.")
        sys.exit(1)

    # ეტაპი 2: აუდიო ფაილების სეგმენტაცია
    if not args.skip_segmentation:
        print("\n===== ეტაპი 2: აუდიო ფაილების სეგმენტაცია =====")
        segments_metadata = os.path.join(segments_dir, "segments_metadata.csv")
        segment_audio_files(processed_dir, processed_metadata, segments_dir)
    else:
        print("\n===== ეტაპი 2: სეგმენტაცია გამოტოვებულია =====")
        segments_metadata = os.path.join(segments_dir, "segments_metadata.csv")

    # შევამოწმოთ შეიქმნა თუ არა სეგმენტების მეტადატა ფაილი
    if not os.path.exists(segments_metadata):
        print(f"შეცდომა: სეგმენტების მეტადატა ფაილი {segments_metadata} ვერ მოიძებნა.")
        print("დარწმუნდით რომ აუდიო ფაილების სეგმენტაციის ეტაპი წარმატებით დასრულდა.")
        sys.exit(1)

    # ეტაპი 3: მონაცემთა ნაკრებების დაყოფა
    if not args.skip_splitting:
        print("\n===== ეტაპი 3: მონაცემთა ნაკრებების დაყოფა =====")
        split_dataset(cv_corpus_dir, clips_tsv, segments_dir, segments_metadata, split_dir)
        create_xtts_config(split_dir)
    else:
        print("\n===== ეტაპი 3: დაყოფა გამოტოვებულია =====")

    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\n===== მონაცემთა მომზადება დასრულებულია! =====")
    print(f"სულ დახარჯული დრო: {int(hours)}სთ {int(minutes)}წთ {int(seconds)}წმ")
    print(f"\nმომზადებული მონაცემები ხელმისაწვდომია შემდეგ დირექტორიებში:")
    print(f"- სატრენინგო: {os.path.join(split_dir, 'train')}")
    print(f"- ვალიდაციის: {os.path.join(split_dir, 'val')}")
    print(f"- სატესტო: {os.path.join(split_dir, 'test')}")
    print(f"- XTTS კონფიგურაცია: {os.path.join(split_dir, 'xtts_config.json')}")


if __name__ == "__main__":
    main()