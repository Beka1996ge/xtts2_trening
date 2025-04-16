#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XTTS-v2 მოდელის გამოყენებით მეტყველების ფაილების გენერაცია
"""

import os
import sys
import json
import torch
import argparse
import numpy as np
import soundfile as sf
from tqdm import tqdm

# ლოკალური მოდულების იმპორტი
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xtts2_training.phoneme_processor import KaPhonemeProcessor

# XTTS ბიბლიოთეკები
from TTS.tts.configs.xtts_config import XTTSConfig
from TTS.tts.models.xtts import Xtts


class XTTSSpeechGenerator:
    """
    XTTS-v2 მოდელის გამოყენებით მეტყველების ფაილების გენერაციის კლასი
    """

    def __init__(
            self,
            model_path: str,
            config_path: str,
            output_dir: str = "generated_speech",
            device: str = None
    ):
        """
        Args:
            model_path: მოდელის ფაილის მისამართი
            config_path: კონფიგურაციის ფაილის მისამართი
            output_dir: გამომავალი ფაილების დირექტორია
            device: მოწყობილობა (cuda/cpu)
        """
        self.model_path = model_path
        self.config_path = config_path
        self.output_dir = output_dir

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # შეიქმნას გამომავალი დირექტორია
        os.makedirs(output_dir, exist_ok=True)

        # მოდელის ჩატვირთვა
        self.model, self.config = self._load_model()

        print(f"მოდელი ჩატვირთულია მოწყობილობაზე: {self.device}")

    def _load_model(self):
        """
        მოდელის და კონფიგურაციის ჩატვირთვა

        Returns:
            Tuple: მოდელი და კონფიგურაცია
        """
        # კონფიგურაციის ჩატვირთვა
        config = XTTSConfig()
        config.load_json(self.config_path)

        # მოდელის ინიციალიზაცია
        model = Xtts(config)

        # ჩექფოინტის ჩატვირთვა
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model"])

        # ფონემების პროცესორის ინიციალიზაცია
        model.phoneme_processor = KaPhonemeProcessor()

        # მოდელის გადატანა მოწყობილობაზე
        model = model.to(self.device)
        model.eval()

        return model, config

    def generate_from_text(
            self,
            text: str,
            speaker_embedding: torch.Tensor,
            output_filename: str = None,
            return_waveform: bool = False
    ):
        """
        მეტყველების გენერაცია ტექსტიდან

        Args:
            text: შემავალი ტექსტი
            speaker_embedding: ხმოვანი ემბედინგი
            output_filename: გამომავალი ფაილის სახელი
            return_waveform: დააბრუნოს თუ არა ხმოვანი ტალღა

        Returns:
            str: გამომავალი ფაილის მისამართი
            np.ndarray (optional): ხმოვანი ტალღა (თუ return_waveform=True)
        """
        # ემბედინგის გადატანა მოწყობილობაზე
        speaker_embedding = speaker_embedding.to(self.device)

        # ფონემების გარდაქმნა (თუ იყენებს მოდელი)
        if hasattr(self.model, 'phoneme_processor') and self.model.phoneme_processor is not None:
            phonemes = self.model.phoneme_processor.text_to_phonemes(text)
            input_data = {'text': text, 'phonemes': phonemes}
        else:
            input_data = {'text': text}

        input_data['speaker_embedding'] = speaker_embedding

        with torch.no_grad():
            # ინფერენსის გაშვება
            outputs = self.model.inference(input_data)

        # ხმოვანი ტალღის გამოღება
        waveform = outputs['waveform'].cpu().numpy()

        # ფაილის სახელის განსაზღვრა
        if output_filename is None:
            output_filename = f"generated_{len(os.listdir(self.output_dir))}.wav"

        # ფაილის მისამართის მომზადება
        output_path = os.path.join(self.output_dir, output_filename)

        # ხმოვანი ფაილის შენახვა
        sf.write(output_path, waveform, self.config.audio.sample_rate)

        print(f"აუდიო ფაილი შენახულია: {output_path}")

        if return_waveform:
            return output_path, waveform
        else:
            return output_path

    def generate_from_file(
            self,
            input_file: str,
            speaker_embedding_file: str,
            output_dir: str = None
    ):
        """
        მეტყველების გენერაცია ფაილიდან

        Args:
            input_file: შემავალი ტექსტური ფაილი, სადაც ყოველ ხაზზე ერთი წინადადებაა
            speaker_embedding_file: ხმოვანი ემბედინგის ფაილი
            output_dir: გამომავალი დირექტორია

        Returns:
            List[str]: გენერირებული ფაილების მისამართები
        """
        if output_dir is None:
            output_dir = self.output_dir

        # გამომავალი დირექტორიის შექმნა
        os.makedirs(output_dir, exist_ok=True)

        # ემბედინგის ჩატვირთვა
        speaker_embedding = np.load(speaker_embedding_file)
        speaker_embedding = torch.FloatTensor(speaker_embedding)

        # ტექსტების წაკითხვა
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = f.read().splitlines()

        # მეტყველების ფაილების გენერაცია
        output_files = []

        for i, text in enumerate(tqdm(texts, desc="აუდიოს გენერაცია")):
            # ცარიელი ტექსტების გამოტოვება
            if not text.strip():
                continue

            # გამომავალი ფაილის სახელის განსაზღვრა
            output_filename = f"generated_{i:04d}.wav"

            # მეტყველების გენერაცია
            output_path = self.generate_from_text(
                text=text,
                speaker_embedding=speaker_embedding,
                output_filename=output_filename
            )

            output_files.append(output_path)

        return output_files

    def generate_from_json(
            self,
            input_json: str,
            embedding_dir: str,
            output_dir: str = None
    ):
        """
        მეტყველების გენერაცია JSON ფაილიდან

        Args:
            input_json: JSON ფაილი, რომელიც შეიცავს ტექსტებს და სპიკერების ინფორმაციას
            embedding_dir: ხმოვანი ემბედინგების დირექტორია
            output_dir: გამომავალი დირექტორია

        Returns:
            Dict: გენერირებული ფაილების მისამართები, დაჯგუფებული სპიკერების მიხედვით
        """
        if output_dir is None:
            output_dir = self.output_dir

        # გამომავალი დირექტორიის შექმნა
        os.makedirs(output_dir, exist_ok=True)

        # JSON ფაილის ჩატვირთვა
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # რეზულტატების შენახვისთვის
        results = {}

        # გავიაროთ ყველა ჩანაწერი
        for item in tqdm(data, desc="აუდიოს გენერაცია"):
            text = item.get('text', '')
            speaker_id = item.get('speaker_id', 'unknown')
            embedding_file = item.get('embedding_file', None)

            # ცარიელი ტექსტების გამოტოვება
            if not text.strip():
                continue

            # ემბედინგის ფაილის განსაზღვრა
            if embedding_file is None:
                embedding_file = f"{speaker_id}.npy"

            # ემბედინგის ფაილის მისამართი
            embedding_path = os.path.join(embedding_dir, embedding_file)

            # ემბედინგის არსებობის შემოწმება
            if not os.path.exists(embedding_path):
                print(f"გაფრთხილება: ემბედინგის ფაილი ვერ მოიძებნა: {embedding_path}")
                continue

            # მოცემული სპიკერის დირექტორია
            speaker_dir = os.path.join(output_dir, speaker_id)
            os.makedirs(speaker_dir, exist_ok=True)

            # ემბედინგის ჩატვირთვა
            speaker_embedding = np.load(embedding_path)
            speaker_embedding = torch.FloatTensor(speaker_embedding)

            # გამომავალი ფაილის სახელის განსაზღვრა
            output_filename = f"{item.get('id', len(os.listdir(speaker_dir)))}.wav"

            # მეტყველების გენერაცია
            output_path = self.generate_from_text(
                text=text,
                speaker_embedding=speaker_embedding,
                output_filename=os.path.join(speaker_id, output_filename)
            )

            # შედეგის შენახვა
            if speaker_id not in results:
                results[speaker_id] = []

            results[speaker_id].append({
                'text': text,
                'audio_file': output_path,
                'id': item.get('id', None)
            })

        # შედეგების ჩაწერა JSON ფაილში
        results_path = os.path.join(output_dir, 'generation_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"გენერაციის შედეგები შენახულია: {results_path}")

        return results


def main():
    """მთავარი გამშვები ფუნქცია"""
    parser = argparse.ArgumentParser(description="XTTS-v2 მოდელით მეტყველების გენერაცია")

    parser.add_argument("--model", type=str, required=True,
                        help="მოდელის ფაილის მისამართი")

    parser.add_argument("--config", type=str, required=True,
                        help="კონფიგურაციის ფაილის მისამართი")

    parser.add_argument("--output_dir", type=str, default="generated_speech",
                        help="გამომავალი დირექტორია")

    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None,
                        help="მოწყობილობა (cuda/cpu)")

    # ტექსტიდან გენერაციისთვის
    parser.add_argument("--text", type=str,
                        help="ტექსტი მეტყველების გენერაციისთვის")

    parser.add_argument("--embedding", type=str,
                        help="ხმოვანი ემბედინგის ფაილის მისამართი")

    # ფაილიდან გენერაციისთვის
    parser.add_argument("--input_file", type=str,
                        help="შემავალი ტექსტური ფაილი")

    # JSON-დან გენერაციისთვის
    parser.add_argument("--input_json", type=str,
                        help="შემავალი JSON ფაილი")

    parser.add_argument("--embedding_dir", type=str,
                        help="ხმოვანი ემბედინგების დირექტორია")

    args = parser.parse_args()

    # გენერატორის ინიციალიზაცია
    generator = XTTSSpeechGenerator(
        model_path=args.model,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device
    )

    # გენერაციის რეჟიმის განსაზღვრა
    if args.text and args.embedding:
        # ერთი ტექსტიდან გენერაცია
        generator.generate_from_text(
            text=args.text,
            speaker_embedding=torch.FloatTensor(np.load(args.embedding)),
            output_filename="generated.wav"
        )
    elif args.input_file and args.embedding:
        # ფაილიდან გენერაცია
        generator.generate_from_file(
            input_file=args.input_file,
            speaker_embedding_file=args.embedding,
            output_dir=args.output_dir
        )
    elif args.input_json and args.embedding_dir:
        # JSON-დან გენერაცია
        generator.generate_from_json(
            input_json=args.input_json,
            embedding_dir=args.embedding_dir,
            output_dir=args.output_dir
        )
    else:
        parser.print_help()
        print("\nგთხოვთ მიუთითოთ რომელიმე გენერაციის რეჟიმი:")
        print("1. ერთი ტექსტიდან გენერაცია: --text და --embedding")
        print("2. ფაილიდან გენერაცია: --input_file და --embedding")
        print("3. JSON-დან გენერაცია: --input_json და --embedding_dir")


if __name__ == "__main__":
    main()