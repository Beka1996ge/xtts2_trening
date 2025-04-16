#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import glob
from scipy.io import wavfile
from pydub import AudioSegment
import multiprocessing
from concurrent.futures import ProcessPoolExecutor


def convert_to_wav(mp3_path, wav_path):
    """MP3 ფაილის WAV ფორმატში გადაყვანა 16kHz გაწყვეტით"""
    try:
        audio = AudioSegment.from_mp3(mp3_path)
        audio = audio.set_frame_rate(16000)
        audio = audio.set_channels(1)
        audio.export(wav_path, format="wav")
        return True
    except Exception as e:
        print(f"შეცდომა ფაილის დამუშავებისას {mp3_path}: {str(e)}")
        return False


def normalize_audio(wav_path):
    """აუდიოს ნორმალიზება"""
    try:
        # აუდიოს ჩატვირთვა
        audio, sr = librosa.load(wav_path, sr=16000)

        # ხმაურის შემცირება - მარტივი მეთოდი
        # უფრო რთული მეთოდებისთვის spectral gating ან Wiener filtering გამოიყენება
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)

        # ხმის დონის ნორმალიზება
        audio_normalized = librosa.util.normalize(audio_trimmed)

        # შენახვა
        sf.write(wav_path, audio_normalized, sr)
        return True
    except Exception as e:
        print(f"შეცდომა ნორმალიზებისას {wav_path}: {str(e)}")
        return False


def is_good_quality(wav_path, min_duration=1.0, max_duration=20.0, min_rms=0.01):
    """ამოწმებს აუდიო ფაილის ხარისხს"""
    try:
        audio, sr = librosa.load(wav_path, sr=16000)

        # ხანგრძლივობის შემოწმება
        duration = librosa.get_duration(y=audio, sr=sr)
        if duration < min_duration or duration > max_duration:
            return False

        # RMS ენერგიის შემოწმება (ძალიან ჩუმი აუდიოების გასაფილტრად)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < min_rms:
            return False

        return True
    except Exception as e:
        print(f"შეცდომა ხარისხის შემოწმებისას {wav_path}: {str(e)}")
        return False


def process_audio_file(mp3_file, output_dir):
    """ერთი აუდიო ფაილის სრული დამუშავება"""
    base_name = os.path.basename(mp3_file).replace('.mp3', '')
    wav_path = os.path.join(output_dir, f"{base_name}.wav")

    # 1. გადაყვანა WAV ფორმატში
    success = convert_to_wav(mp3_file, wav_path)
    if not success:
        return None

    # 2. ნორმალიზება
    success = normalize_audio(wav_path)
    if not success:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return None

    # 3. ხარისხის შემოწმება
    if not is_good_quality(wav_path):
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return None

    return wav_path


def process_all_files(input_dir, output_dir, clips_tsv_path):
    """ყველა აუდიო ფაილის დამუშავება პარალელურად"""
    os.makedirs(output_dir, exist_ok=True)

    # ტრანსკრიფციების ჩატვირთვა
    print(f"მეტადატას ჩატვირთვა: {clips_tsv_path}")

    # TSV ფაილის პარსინგი
    try:
        clips_df = pd.read_csv(clips_tsv_path, sep='\t')
        print(f"წარმატებით ჩაიტვირთა {len(clips_df)} ჩანაწერი TSV ფაილიდან")
    except Exception as e:
        print(f"შეცდომა TSV ფაილის ჩატვირთვისას: {str(e)}")
        print("ვცდილობთ სხვა დელიმიტერით...")
        try:
            clips_df = pd.read_csv(clips_tsv_path, sep=',')
            print(f"წარმატებით ჩაიტვირთა {len(clips_df)} ჩანაწერი CSV ფაილიდან")
        except Exception as e2:
            print(f"შეცდომა CSV ფაილის ჩატვირთვისას: {str(e2)}")
            return pd.DataFrame()

    # შევამოწმოთ აუცილებელი სვეტები
    required_columns = ['path', 'sentence']

    # შევეცადოთ სვეტების სახელების დადგენას Common Voice-ის სხვადასხვა ვერსიებისთვის
    column_mapping = {}

    if 'path' not in clips_df.columns:
        if 'filename' in clips_df.columns:
            column_mapping['filename'] = 'path'
        elif 'clip_name' in clips_df.columns:
            column_mapping['clip_name'] = 'path'

    if 'sentence' not in clips_df.columns:
        if 'text' in clips_df.columns:
            column_mapping['text'] = 'sentence'
        elif 'transcript' in clips_df.columns:
            column_mapping['transcript'] = 'sentence'

    # სვეტების სახელების გადარქმევა
    if column_mapping:
        clips_df = clips_df.rename(columns=column_mapping)

    # შევამოწმოთ ისევ აუცილებელი სვეტები
    missing_columns = [col for col in required_columns if col not in clips_df.columns]
    if missing_columns:
        print(f"შეცდომა: მეტადატა ფაილში ვერ მოიძებნა აუცილებელი სვეტები: {missing_columns}")
        print(f"არსებული სვეტები: {clips_df.columns.tolist()}")
        return pd.DataFrame()

    # MP3 ფაილების მოძიება
    mp3_files = glob.glob(os.path.join(input_dir, "*.mp3"))
    print(f"ნაპოვნია {len(mp3_files)} MP3 ფაილი დასამუშავებლად")

    # გავფილტროთ მხოლოდ ის MP3 ფაილები, რომლებიც მეტადატაშია
    mp3_basenames = [os.path.basename(f) for f in mp3_files]

    # შევამოწმოთ არის თუ არა .mp3 გაფართოება path სვეტში
    has_extension = clips_df['path'].str.endswith('.mp3').any()

    if has_extension:
        valid_mp3s = [f for f in mp3_files if os.path.basename(f) in clips_df['path'].values]
    else:
        valid_mp3s = [f for f in mp3_files if os.path.basename(f).replace('.mp3', '') in clips_df['path'].values]

    print(f"მეტადატაში მოძიებულია {len(valid_mp3s)} MP3 ფაილი")

    if len(valid_mp3s) == 0:
        print("გაფრთხილება: მეტადატაში ვერ მოიძებნა არცერთი MP3 ფაილი")
        print("შევამოწმოთ შესაძლო გაუგებრობა path სვეტში:")

        # დავბეჭდოთ პირველი რამდენიმე მნიშვნელობა path სვეტიდან
        print(f"path სვეტის პირველი 5 მნიშვნელობა: {clips_df['path'].head(5).tolist()}")
        print(f"MP3 ფაილების პირველი 5 სახელი: {mp3_basenames[:5] if len(mp3_basenames) >= 5 else mp3_basenames}")

        # შევეცადოთ პრეფიქსით მოძიება
        matching_files = []
        for file_path in mp3_files:
            basename = os.path.basename(file_path)
            for path_value in clips_df['path']:
                if path_value in basename or basename.replace('.mp3', '') in path_value:
                    matching_files.append(file_path)
                    break

        if matching_files:
            valid_mp3s = matching_files
            print(f"პრეფიქსული შედარებით ნაპოვნია {len(valid_mp3s)} MP3 ფაილი")
        else:
            # თუ ვერაფერი მოვძებნეთ, დავამუშაოთ ყველა მოძიებული MP3
            valid_mp3s = mp3_files
            print("ვერ მოიძებნა შესაბამისობა. ვამუშავებთ ყველა MP3 ფაილს.")

    # საპროცესო ბირთვების რაოდენობა
    num_processes = max(1, multiprocessing.cpu_count() - 1)

    processed_files = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {executor.submit(process_audio_file, mp3_file, output_dir): mp3_file for mp3_file in valid_mp3s}

        for future in tqdm(futures, desc="აუდიო ფაილების დამუშავება"):
            mp3_file = futures[future]
            try:
                result = future.result()
                if result:
                    processed_files.append(os.path.basename(mp3_file).replace('.mp3', ''))
            except Exception as e:
                print(f"შეცდომა ფაილის დამუშავებისას {mp3_file}: {str(e)}")

    # ავაგოთ ახალი დამუშავებული მეტადატა
    processed_metadata = []

    for processed_file in processed_files:
        # მოვძებნოთ შესაბამისი ტრანსკრიფცია
        if has_extension:
            matching_rows = clips_df[clips_df['path'] == f"{processed_file}.mp3"]
        else:
            matching_rows = clips_df[clips_df['path'] == processed_file]

        if len(matching_rows) > 0:
            for _, row in matching_rows.iterrows():
                processed_metadata.append({
                    'path': f"{processed_file}.wav",
                    'sentence': row['sentence']
                })
        else:
            # თუ ვერ ვიპოვეთ პირდაპირი შესაბამისობა, ვცადოთ ნაწილობრივი
            for _, row in clips_df.iterrows():
                if processed_file in row['path'] or row['path'] in processed_file:
                    processed_metadata.append({
                        'path': f"{processed_file}.wav",
                        'sentence': row['sentence']
                    })
                    break

    # შევინახოთ დამუშავებული მეტადატა
    processed_df = pd.DataFrame(processed_metadata)

    if not processed_df.empty:
        processed_df.to_csv(os.path.join(output_dir, "processed_clips.csv"), index=False)
        print(f"წარმატებით დამუშავდა {len(processed_df)} ფაილი {len(valid_mp3s)}-დან")
    else:
        print("შეცდომა: ვერ მოხერხდა დამუშავებული მეტადატის შექმნა")

    return processed_df


if __name__ == "__main__":
    data_dir = "data/cv-corpus/ka"
    clips_tsv = os.path.join(data_dir, "validated.tsv")
    audio_dir = os.path.join(data_dir, "clips")
    output_dir = "data/processed/audio"

    processed_df = process_all_files(audio_dir, output_dir, clips_tsv)
    print(f"დამუშავებული ფაილების რაოდენობა: {len(processed_df)}")