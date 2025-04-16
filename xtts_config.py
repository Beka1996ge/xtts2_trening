#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XTTS-v2 მოდელის კონფიგურაციის კლასი
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class XTTSConfig:
    """
    XTTS-v2 მოდელის კონფიგურაციის კლასი
    """
    # მოდელის პარამეტრები
    model_name: str = "xtts_v2_ka"
    run_name: str = "xtts_v2_georgian_training"

    # მონაცემების პარამეტრები
    data_path: str = "data/split"
    output_path: str = "models/xtts_v2_ka"

    # ტრენინგის პარამეტრები
    batch_size: int = 16
    eval_batch_size: int = 8
    num_workers: int = 4
    num_eval_workers: int = 2
    epochs: int = 100
    save_step: int = 5000
    print_step: int = 100
    eval_step: int = 1000

    # ოპტიმიზაციის პარამეტრები
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-6
    grad_clip_threshold: float = 1.0

    # გრადიენტების აკუმულაცია
    grad_accum_steps: int = 2

    # აუდიოს პარამეტრები
    audio: Dict = field(default_factory=lambda: {
        "sample_rate": 16000,
        "win_length": 1024,
        "hop_length": 256,
        "num_mels": 80,
        "mel_fmin": 0,
        "mel_fmax": 8000
    })

    # ჩექფოინტის პარამეტრები
    checkpoint_path: str = "checkpoints/xtts_v2_ka"
    dashboard_logger: str = "tensorboard"

    # მოდელის სპეციფიკური პარამეტრები
    num_chars: int = 100  # დაემატა: სიმბოლოების რაოდენობა
    use_d_vector_file: bool = True
    d_vector_file: str = "data/embeddings/embeddings"
    d_vector_dim: int = 512

    # ენის პარამეტრები
    language_id: str = "ka"
    use_language_embedding: bool = True

    # ფონემების პარამეტრები
    use_phonemes: bool = True
    phoneme_language: str = "ka"
    text_cleaner: str = "basic_cleaners"

    # მოდელის არქიტექტურა
    model_args: Dict = field(default_factory=lambda: {
        "hidden_channels": 192,
        "speaker_embedding_dim": 512,
        "use_speaker_embedding": True,
        "use_d_vector_file": True,
        "d_vector_dim": 512,
        "num_chars": 100
    })

    def load_json(self, json_file_path):
        """
        კონფიგურაციის ჩატვირთვა JSON ფაილიდან
        """
        import json
        with open(json_file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        # კონფიგურაციის პარამეტრების განახლება
        for key, value in config_dict.items():
            if not key.startswith("# "):  # კომენტარების გამოტოვება
                if hasattr(self, key):
                    setattr(self, key, value)

        # მოდელის არგუმენტები
        if "model_params" in config_dict:
            self.model_args = config_dict["model_params"]
            if "d_vector_dim" in config_dict["model_params"]:
                self.d_vector_dim = config_dict["model_params"]["d_vector_dim"]
            if "use_d_vector_file" in config_dict["model_params"]:
                self.use_d_vector_file = config_dict["model_params"]["use_d_vector_file"]
            if "speaker_embedding_dim" in config_dict["model_params"]:
                self.model_args["speaker_embedding_dim"] = config_dict["model_params"]["speaker_embedding_dim"]
            if "num_chars" not in self.model_args:
                self.model_args["num_chars"] = self.num_chars

        # ფონემების პარამეტრები
        if "phoneme_params" in config_dict:
            if "use_phonemes" in config_dict["phoneme_params"]:
                self.use_phonemes = config_dict["phoneme_params"]["use_phonemes"]
            if "phoneme_language" in config_dict["phoneme_params"]:
                self.phoneme_language = config_dict["phoneme_params"]["phoneme_language"]
            if "text_cleaner" in config_dict["phoneme_params"]:
                self.text_cleaner = config_dict["phoneme_params"]["text_cleaner"]

    def to_dict(self):
        """
        კონფიგურაციის გადაყვანა ლექსიკონში
        """
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                result[key] = value
        return result