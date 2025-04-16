#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import numpy as np
from pydub import AudioSegment
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import json


def segment_audio(audio_path, transcript, output_dir, segment_id, min_duration=3.0, max_duration=15.0):
    """აუდიო ფაილის სეგმენტაცია ოპტიმალური ზომის ნაწილებად"""
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        duration = librosa.get_duration(y=audio, sr=sr)

        # თუ ფაილი უკვე ოპტიმალური ზომისაა
        if min_duration <= duration <= max_duration:
            output_path = os.path.join(output_dir, f"{segment_id}_0.wav")
            sf.write(output_path, audio, sr)
            return [(output_path, transcript)]

        # თუ ფაილი მეტისმეტად მოკლეა
        if duration < min_duration:
            return []

        # ხმოვანი ნაწილების აღმოჩენა (VAD - Voice Activity Detection)
        # მარტივი მიდგომით ენერგიაზე დაფუძნებული
        frame_length = 1024
        hop_length = 512

        # აუდიოს ენერგიის გამოთვლა
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

        # პაუზების აღმოჩენა (ენერგიის დაბალი მაჩვენებელი)
        threshold = 0.05 * np.mean(energy)
        silence_frames = np.where(energy < threshold)[0]

        # პაუზების გარდაქმნა წამებში
        silence_times = librosa.frames_to_time(silence_frames, sr=sr, hop_length=hop_length)

        # პოტენციური დაყოფის წერტილები
        split_points = []
        target_duration = (min_duration + max_duration) / 2

        current_pos = 0
        while current_pos < duration - min_duration:
            # მიზნობრივი პოზიციის მოძებნა
            target_pos = current_pos + target_duration

            # პაუზების მოძებნა მიზნობრივი პოზიციის ახლოს
            potential_splits = silence_times[(silence_times > current_pos + min_duration) &
                                             (silence_times < min(current_pos + max_duration, duration))]

            if len(potential_splits) > 0:
                # უახლოესი პაუზის მოძებნა
                split_point = potential_splits[np.argmin(np.abs(potential_splits - target_pos))]
                split_points.append(split_point)
                current_pos = split_point
            else:
                # თუ პაუზა ვერ მოიძებნა, მაშინ target_duration-ის შემდეგ გავჭრათ
                if current_pos + target_duration < duration - min_duration:
                    split_point = current_pos + target_duration
                    split_points.append(split_point)
                    current_pos = split_point
                else:
                    break

        # სეგმენტებად დაყოფა
        segments = []
        start_time = 0

        for i, split_point in enumerate(split_points):
            end_time = split_point
            segment_audio = audio[int(start_time * sr):int(end_time * sr)]

            # სეგმენტის შენახვა
            output_path = os.path.join(output_dir, f"{segment_id}_{i}.wav")
            sf.write(output_path, segment_audio, sr)

            segments.append((output_path, transcript))
            start_time = end_time

        # ბოლო სეგმენტი
        if start_time < duration and duration - start_time >= min_duration:
            segment_audio = audio[int(start_time * sr):]
            output_path = os.path.join(output_dir, f"{segment_id}_{len(split_points)}.wav")
            sf.write(output_path, segment_audio, sr)
            segments.append((output_path, transcript))

        return segments

    except Exception as e:
        print(f"შეცდომა ფაილის სეგმენტაციისას {audio_path}: {str(e)}")
        return []


def process_file_for_segmentation(args):
    """პარალელური დამუშავებისთვის ერთი ფაილის დამუშავება"""
    audio_path, transcript, output_dir, file_id = args
    return segment_audio(audio_path, transcript, output_dir, file_id)


def segment_all_files(input_dir, metadata_path, output_dir):
    """ყველა აუდიო ფაილის სეგმენტაცია"""
    os.makedirs(output_dir, exist_ok=True)

    # ტრანსკრიფციების ჩატვირთვა
    metadata_df = pd.read_csv(metadata_path)

    tasks = []
    for index, row in metadata_df.iterrows():
        file_id = os.path.basename(row['path']).replace('.wav', '')
        audio_path = os.path.join(input_dir, row['path'])
        if os.path.exists(audio_path):
            tasks.append((audio_path, row['sentence'], output_dir, file_id))

    # პარალელური დამუშავება
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    all_segments = []

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(tqdm(executor.map(process_file_for_segmentation, tasks),
                            total=len(tasks),
                            desc="აუდიო ფაილების სეგმენტაცია"))

    # შედეგების გაერთიანება
    for segments in results:
        all_segments.extend(segments)

    # მეტამონაცემების შექმნა სეგმენტებისთვის
    segments_metadata = []
    for i, (audio_path, transcript) in enumerate(all_segments):
        segments_metadata.append({
            'id': f"segment_{i:06d}",
            'path': os.path.basename(audio_path),
            'sentence': transcript
        })

    # მეტამონაცემების შენახვა
    segments_df = pd.DataFrame(segments_metadata)
    segments_df.to_csv(os.path.join(output_dir, "segments_metadata.csv"), index=False)

    print(f"სულ {len(segments_metadata)} სეგმენტი შეიქმნა")
    return segments_df


if __name__ == "__main__":
    processed_dir = "data/processed/audio"
    processed_metadata = os.path.join(processed_dir, "processed_clips.csv")
    segments_dir = "data/segmented"

    segments_df = segment_all_files(processed_dir, processed_metadata, segments_dir)
    print(f"სეგმენტების რაოდენობა: {len(segments_df)}")