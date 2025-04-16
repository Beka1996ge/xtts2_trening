#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm


def load_speaker_info(cv_tsv_path):
    """Common Voice მონაცემთა ბაზიდან სპიკერების შესახებ ინფორმაციის ჩატვირთვა"""
    try:
        # თუ არსებობს, ჩატვირთე demographics.tsv
        demographics_path = os.path.join(os.path.dirname(cv_tsv_path), "demographics.tsv")
        if os.path.exists(demographics_path):
            demographics_df = pd.read_csv(demographics_path, sep='\t')
            return demographics_df
        else:
            # თუ demographics არ არსებობს, შევეცადოთ სპიკერის ინფო ამოვიღოთ clips.tsv-დან
            clips_df = pd.read_csv(cv_tsv_path, sep='\t')
            if 'client_id' in clips_df.columns:
                speaker_info = clips_df[['client_id']].drop_duplicates()
                speaker_info['gender'] = 'unknown'
                speaker_info['age'] = 'unknown'
                speaker_info['accent'] = 'unknown'
                return speaker_info
    except Exception as e:
        print(f"ვერ მოხერხდა სპიკერების ინფორმაციის ჩატვირთვა: {str(e)}")

    # თუ ვერც ერთი მეთოდი არ იმუშავა, დავაბრუნოთ ცარიელი DataFrame
    return pd.DataFrame(columns=['client_id', 'gender', 'age', 'accent'])


def split_data_balanced(metadata_df, speaker_info, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    """მონაცემების დაბალანსებული დაყოფა"""
    # სპიკერების მონაცემების მიერთება
    if 'client_id' in metadata_df.columns and not speaker_info.empty:
        merged_df = pd.merge(metadata_df, speaker_info, on='client_id', how='left')
    else:
        # თუ სპიკერების ინფორმაცია არ არის, უბრალოდ გამოვიყენოთ არსებული მეტადატა
        merged_df = metadata_df.copy()
        # დავამატოთ დროებითი კლიენტის ID, თუ არ არსებობს
        if 'client_id' not in merged_df.columns:
            file_prefixes = merged_df['path'].apply(lambda x: x.split('_')[0] if isinstance(x, str) else 'unknown')
            merged_df['client_id'] = file_prefixes

    # უნიკალური სპიკერების მიღება
    speakers = merged_df['client_id'].unique()

    # სპიკერების დაყოფა ან პირდაპირ აუდიო ფაილების დაყოფა
    if len(speakers) <= 1:
        print(f"მხოლოდ {len(speakers)} სპიკერი ნაპოვნია. პირდაპირ ვყოფთ აუდიო ფაილებს.")

        # პირდაპირ დავყოთ აუდიო ფაილები
        all_indices = np.arange(len(merged_df))
        np.random.seed(random_state)
        np.random.shuffle(all_indices)

        train_end = int(len(merged_df) * train_size)
        val_end = int(len(merged_df) * (train_size + val_size))

        train_indices = all_indices[:train_end]
        val_indices = all_indices[train_end:val_end]
        test_indices = all_indices[val_end:]

        train_df = merged_df.iloc[train_indices].copy()
        val_df = merged_df.iloc[val_indices].copy()
        test_df = merged_df.iloc[test_indices].copy()
    else:
        # სპიკერების დაყოფა
        train_speakers, temp_speakers = train_test_split(
            speakers,
            train_size=train_size,
            random_state=random_state
        )

        # დარჩენილი სპიკერების დაყოფა ვალიდაციის და ტესტის ნაკრებებად
        val_ratio = val_size / (val_size + test_size)
        val_speakers, test_speakers = train_test_split(
            temp_speakers,
            train_size=val_ratio,
            random_state=random_state
        )

        # მონაცემთა ნაკრებების შექმნა
        train_df = merged_df[merged_df['client_id'].isin(train_speakers)]
        val_df = merged_df[merged_df['client_id'].isin(val_speakers)]
        test_df = merged_df[merged_df['client_id'].isin(test_speakers)]

    # სტატისტიკის გამოთვლა და ჩვენება
    print(f"სატრენინგო ნაკრები: {len(train_df)} ფაილი")
    print(f"ვალიდაციის ნაკრები: {len(val_df)} ფაილი")
    print(f"სატესტო ნაკრები: {len(test_df)} ფაილი")

    # გენდერის განაწილების შემოწმება, თუ მონაცემები არსებობს
    if 'gender' in merged_df.columns and merged_df['gender'].nunique() > 1:
        for name, df in [('სატრენინგო', train_df), ('ვალიდაციის', val_df), ('სატესტო', test_df)]:
            gender_dist = df['gender'].value_counts(normalize=True)
            print(f"{name} ნაკრებში გენდერის განაწილება: {gender_dist.to_dict()}")

    return train_df, val_df, test_df


def copy_files_to_split_directories(train_df, val_df, test_df, audio_dir, output_base_dir):
    """მონაცემების კოპირება შესაბამის დირექტორიებში"""
    # დირექტორიების შექმნა
    train_dir = os.path.join(output_base_dir, "train")
    val_dir = os.path.join(output_base_dir, "val")
    test_dir = os.path.join(output_base_dir, "test")

    for directory in [train_dir, val_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)

    # ფაილების კოპირება და მეტადატის შექმნა
    for name, df, directory in [
        ('სატრენინგო', train_df, train_dir),
        ('ვალიდაციის', val_df, val_dir),
        ('სატესტო', test_df, test_dir)
    ]:
        copied_files = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{name} ფაილების კოპირება"):
            src_path = os.path.join(audio_dir, row['path'])
            dst_path = os.path.join(directory, row['path'])

            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                copied_files.append(row)

        # მხოლოდ წარმატებით დაკოპირებული ფაილების მეტადატის შენახვა
        if copied_files:
            subset_df = pd.DataFrame(copied_files)
            metadata_path = os.path.join(directory, "metadata.csv")
            subset_df.to_csv(metadata_path, index=False)

            # შევქმნათ ფორმატირებული ფაილი XTTS-თვის
            xtts_format = subset_df[['path', 'sentence']].copy()
            xtts_format.columns = ['audio_file', 'text']
            xtts_format['audio_file'] = xtts_format['audio_file'].apply(lambda x: os.path.basename(x))
            xtts_metadata_path = os.path.join(directory, "metadata_xtts.csv")
            xtts_format.to_csv(xtts_metadata_path, index=False)

            print(f"{name} ნაკრებში დაკოპირდა {len(copied_files)} ფაილი")


def main():
    """მთავარი ფუნქცია"""
    # საჭირო დირექტორიების და ფაილების მისამართები
    data_dir = "data/raw/cv-corpus-latest/ka"
    clips_tsv = os.path.join(data_dir, "clips.tsv")
    segments_dir = "data/segmented"
    segments_metadata = os.path.join(segments_dir, "segments_metadata.csv")
    output_dir = "data/split"

    # მონაცემების ჩატვირთვა
    segments_df = pd.read_csv(segments_metadata)
    speaker_info = load_speaker_info(clips_tsv)

    # მონაცემების დაყოფა
    train_df, val_df, test_df = split_data_balanced(segments_df, speaker_info)

    # ფაილების კოპირება შესაბამის დირექტორიებში
    copy_files_to_split_directories(train_df, val_df, test_df, segments_dir, output_dir)

    print("მონაცემთა დაყოფა დასრულებულია!")


if __name__ == "__main__":
    main()