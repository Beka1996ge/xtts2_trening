#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import argparse
import librosa
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor


def get_audio_duration(audio_file):
    """ფაილის ხანგრძლივობის დადგენა"""
    try:
        audio, sr = librosa.load(audio_file, sr=None)
        duration = librosa.get_duration(y=audio, sr=sr)
        return duration
    except Exception as e:
        print(f"შეცდომა {audio_file} ფაილის ხანგრძლივობის დადგენისას: {str(e)}")
        return None


def generate_metadata_for_clips(clips_dir, output_file, sentences_file=None):
    """მეტადატას გენერაცია clips დირექტორიაში არსებული მპ3 ფაილებისთვის"""
    print(f"ვეძებთ MP3 ფაილებს დირექტორიაში: {clips_dir}")
    mp3_files = glob.glob(os.path.join(clips_dir, "*.mp3"))
    print(f"ნაპოვნია {len(mp3_files)} MP3 ფაილი")

    # შევეცადოთ წავიკითხოთ წინადადებების ფაილი, თუ მითითებულია
    sentences_dict = {}
    if sentences_file and os.path.exists(sentences_file):
        try:
            print(f"ვკითხულობთ წინადადებების ფაილს: {sentences_file}")
            sentences_df = pd.read_csv(sentences_file, sep='\t')
            # შევამოწმოთ სვეტები
            if 'client_id' in sentences_df.columns and 'sentence' in sentences_df.columns:
                for _, row in sentences_df.iterrows():
                    sentences_dict[row['client_id']] = row['sentence']
                print(f"წავიკითხეთ {len(sentences_dict)} წინადადება")
            else:
                print(
                    f"გაფრთხილება: წინადადებების ფაილში არ არის საჭირო სვეტები. არსებული სვეტები: {sentences_df.columns.tolist()}")
        except Exception as e:
            print(f"შეცდომა წინადადებების ფაილის წაკითხვისას: {str(e)}")

    # პარალელურად დავადგინოთ აუდიო ფაილების ხანგრძლივობა
    print("ვადგენთ აუდიო ფაილების ხანგრძლივობას...")
    durations = {}

    num_processes = max(1, multiprocessing.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {executor.submit(get_audio_duration, mp3_file): mp3_file for mp3_file in mp3_files}

        for future in tqdm(futures, total=len(mp3_files), desc="აუდიო ხანგრძლივობის დადგენა"):
            mp3_file = futures[future]
            try:
                duration = future.result()
                if duration is not None:
                    durations[mp3_file] = duration
            except Exception as e:
                print(f"შეცდომა {mp3_file} ფაილის დამუშავებისას: {str(e)}")

    # შევქმნათ მეტადატა
    metadata = []
    for mp3_file in mp3_files:
        basename = os.path.basename(mp3_file)
        file_id = basename.replace('.mp3', '')

        # შევეცადოთ წინადადების პოვნა ID-ით
        sentence = ""
        if file_id in sentences_dict:
            sentence = sentences_dict[file_id]
        else:
            # შევეცადოთ client_id-ის ექსტრაქცია ფაილის სახელიდან
            parts = file_id.split('_')
            if len(parts) >= 3:
                potential_id = parts[2]  # common_voice_ka_42114137.mp3 -> 42114137
                if potential_id in sentences_dict:
                    sentence = sentences_dict[potential_id]

        # თუ ვერ ვიპოვეთ წინადადება, დავტოვოთ ცარიელს
        duration = durations.get(mp3_file, 0)

        metadata.append({
            'path': basename,
            'sentence': sentence,
            'duration': duration
        })

    # შევინახოთ მეტადატა
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(output_file, index=False)
    print(f"მეტადატა ფაილი შექმნილია: {output_file}")
    print(f"შექმნილია {len(metadata_df)} ჩანაწერი")

    # შევქმნათ ვალიდაციის TSV ფაილიც
    tsv_output = output_file.replace('.csv', '.tsv')
    metadata_df.to_csv(tsv_output, sep='\t', index=False)
    print(f"TSV მეტადატა ფაილი შექმნილია: {tsv_output}")

    return metadata_df


def main():
    parser = argparse.ArgumentParser(description="MP3 ფაილებისთვის მეტადატას გენერაცია")
    parser.add_argument("--clips_dir", default="data/cv-corpus/ka/clips", help="MP3 ფაილების დირექტორია")
    parser.add_argument("--output", default="data/cv-corpus/ka/generated_metadata.csv",
                        help="გამომავალი მეტადატა ფაილი")
    parser.add_argument("--sentences", default=None, help="წინადადებების ფაილი (არასავალდებულო)")

    args = parser.parse_args()

    generate_metadata_for_clips(args.clips_dir, args.output, args.sentences)


if __name__ == "__main__":
    main()