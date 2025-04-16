#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XTTS-v2 მოდელის ვალიდაციის მეტრიკები და შეფასების მექანიზმები
"""

import os
import json
import torch
import numpy as np
import soundfile as sf
from typing import Dict, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from pesq import pesq
from pystoi import stoi


class XTTSValidation:
    """
    XTTS-v2 მოდელის ვალიდაციის კლასი, რომელიც ითვლის სხვადასხვა მეტრიკებს
    """

    def __init__(
            self,
            model,
            val_loader,
            device,
            output_dir: str = "validation_outputs",
            sample_rate: int = 16000
    ):
        """
        Args:
            model: XTTS მოდელი
            val_loader: ვალიდაციის მონაცემთა ჩამტვირთველი
            device: მოწყობილობა (cuda/cpu)
            output_dir: შედეგების დირექტორია
            sample_rate: ხმის გაციფრების სიხშირე
        """
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        self.sample_rate = sample_rate

        # შედეგების დირექტორიის შექმნა
        os.makedirs(output_dir, exist_ok=True)

        # მეტრიკები
        self.metrics = {}

    def _compute_pesq(self, reference: np.ndarray, degraded: np.ndarray) -> float:
        """
        PESQ (Perceptual Evaluation of Speech Quality) მეტრიკის გამოთვლა

        Args:
            reference: ორიგინალი აუდიო
            degraded: დაგენერირებული აუდიო

        Returns:
            float: PESQ მნიშვნელობა
        """
        try:
            score = pesq(self.sample_rate, reference, degraded, 'wb')
            return score
        except Exception as e:
            print(f"PESQ გამოთვლის შეცდომა: {e}")
            return 0.0

    def _compute_stoi(self, reference: np.ndarray, degraded: np.ndarray) -> float:
        """
        STOI (Short-Time Objective Intelligibility) მეტრიკის გამოთვლა

        Args:
            reference: ორიგინალი აუდიო
            degraded: დაგენერირებული აუდიო

        Returns:
            float: STOI მნიშვნელობა
        """
        try:
            score = stoi(reference, degraded, self.sample_rate, extended=False)
            return score
        except Exception as e:
            print(f"STOI გამოთვლის შეცდომა: {e}")
            return 0.0

    def _compute_mse(self, reference: np.ndarray, degraded: np.ndarray) -> float:
        """
        MSE (Mean Squared Error) მეტრიკის გამოთვლა

        Args:
            reference: ორიგინალი მელ-სპექტროგრამა
            degraded: დაგენერირებული მელ-სპექტროგრამა

        Returns:
            float: MSE მნიშვნელობა
        """
        # ვლაპარაკობთ მელ-სპექტროგრამებზე ან ხმოვან ტალღებზე
        return mean_squared_error(reference, degraded)

    def _save_audio_sample(
            self,
            waveform: np.ndarray,
            filename: str,
            directory: str = None
    ) -> str:
        """
        აუდიო სემპლის შენახვა

        Args:
            waveform: ხმოვანი ტალღა
            filename: ფაილის სახელი
            directory: დირექტორია (თუ None, გამოიყენება output_dir)

        Returns:
            str: შენახული ფაილის მისამართი
        """
        if directory is None:
            directory = self.output_dir

        os.makedirs(directory, exist_ok=True)

        output_path = os.path.join(directory, filename)
        sf.write(output_path, waveform, self.sample_rate)

        return output_path

    def _plot_spectrograms(
            self,
            reference_mel: np.ndarray,
            generated_mel: np.ndarray,
            filename: str,
            directory: str = None
    ) -> str:
        """
        სპექტროგრამების შედარების ვიზუალიზაცია

        Args:
            reference_mel: ორიგინალი მელ-სპექტროგრამა
            generated_mel: დაგენერირებული მელ-სპექტროგრამა
            filename: ფაილის სახელი
            directory: დირექტორია (თუ None, გამოიყენება output_dir)

        Returns:
            str: შენახული ფაილის მისამართი
        """
        if directory is None:
            directory = self.output_dir

        os.makedirs(directory, exist_ok=True)

        plt.figure(figsize=(12, 8))

        # ორიგინალი სპექტროგრამა
        plt.subplot(2, 1, 1)
        plt.imshow(reference_mel.T, aspect='auto', origin='lower')
        plt.title('ორიგინალი მელ-სპექტროგრამა')
        plt.colorbar()

        # დაგენერირებული სპექტროგრამა
        plt.subplot(2, 1, 2)
        plt.imshow(generated_mel.T, aspect='auto', origin='lower')
        plt.title('დაგენერირებული მელ-სპექტროგრამა')
        plt.colorbar()

        plt.tight_layout()

        output_path = os.path.join(directory, filename)
        plt.savefig(output_path)
        plt.close()

        return output_path

    def run_inference(self, input_text: str, speaker_embedding: torch.Tensor) -> Dict:
        """
        ინფერენსის გაშვება მოდელზე

        Args:
            input_text: შემავალი ტექსტი
            speaker_embedding: ხმოვანი ემბედინგი

        Returns:
            Dict: მოდელის გამომავალი
        """
        self.model.eval()

        # ხმოვანი ემბედინგის გადატანა მოწყობილობაზე
        speaker_embedding = speaker_embedding.to(self.device)

        # ფონემების გარდაქმნა (თუ იყენებს მოდელი)
        if hasattr(self.model, 'phoneme_processor') and self.model.phoneme_processor is not None:
            phonemes = self.model.phoneme_processor.text_to_phonemes(input_text)
            input_data = {'text': input_text, 'phonemes': phonemes}
        else:
            input_data = {'text': input_text}

        input_data['speaker_embedding'] = speaker_embedding

        with torch.no_grad():
            # ინფერენსის გაშვება
            outputs = self.model.inference(input_data)

        return outputs

    def evaluate(self, num_samples: int = None) -> Dict:
        """
        მოდელის ვალიდაცია და მეტრიკების გამოთვლა

        Args:
            num_samples: შესაფასებელი სემპლების რაოდენობა (None ნიშნავს ყველას)

        Returns:
            Dict: შეფასების მეტრიკები
        """
        self.model.eval()

        pesq_scores = []
        stoi_scores = []
        mse_scores = []

        eval_samples = []

        # ასაღები სემპლების რაოდენობის განსაზღვრა
        total_samples = len(self.val_loader.dataset)
        if num_samples is None or num_samples > total_samples:
            num_samples = total_samples

        print(f"მოდელის შეფასება {num_samples} სემპლზე...")

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_loader)):
                # ბატჩის გადატანა მოწყობილობაზე
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                # ორიგინალი ხმოვანი ტალღები
                original_waveforms = batch.get('waveforms')

                # ტექსტი და ემბედინგები
                texts = batch.get('texts', [])
                speaker_embeddings = batch.get('speaker_embeddings', [])

                # ყოველი ელემენტისთვის ბატჩში
                for j in range(len(texts)):
                    # სემპლების შეზღუდვა
                    if len(eval_samples) >= num_samples:
                        break

                    # ინფერენსის გაშვება
                    outputs = self.run_inference(texts[j], speaker_embeddings[j])

                    # დაგენერირებული ხმოვანი ტალღა
                    generated_waveform = outputs.get('waveform', torch.zeros(1)).cpu().numpy()

                    # ორიგინალი ხმოვანი ტალღა
                    original_waveform = original_waveforms[j].cpu().numpy()

                    # ზომების გათანაბრება (უმცირესზე)
                    min_length = min(len(original_waveform), len(generated_waveform))
                    original_waveform = original_waveform[:min_length]
                    generated_waveform = generated_waveform[:min_length]

                    # მეტრიკების გამოთვლა
                    pesq_score = self._compute_pesq(original_waveform, generated_waveform)
                    stoi_score = self._compute_stoi(original_waveform, generated_waveform)

                    # სპექტროგრამები (თუ არის ხელმისაწვდომი)
                    if 'mel_spectrogram' in outputs and 'original_mel' in batch:
                        original_mel = batch['original_mel'][j].cpu().numpy()
                        generated_mel = outputs['mel_spectrogram'].cpu().numpy()

                        # ზომების გათანაბრება (უმცირესზე)
                        min_frames = min(original_mel.shape[0], generated_mel.shape[0])
                        original_mel = original_mel[:min_frames]
                        generated_mel = generated_mel[:min_frames]

                        # MSE გამოთვლა
                        mse_score = self._compute_mse(original_mel, generated_mel)
                        mse_scores.append(mse_score)

                        # სპექტროგრამების ვიზუალიზაცია
                        spec_plot_path = self._plot_spectrograms(
                            original_mel,
                            generated_mel,
                            f"sample_{len(eval_samples)}_spectrogram.png"
                        )
                    else:
                        mse_score = None
                        spec_plot_path = None

                    # სემპლის შენახვა
                    original_path = self._save_audio_sample(
                        original_waveform,
                        f"sample_{len(eval_samples)}_original.wav"
                    )

                    generated_path = self._save_audio_sample(
                        generated_waveform,
                        f"sample_{len(eval_samples)}_generated.wav"
                    )

                    # მეტრიკების შენახვა
                    pesq_scores.append(pesq_score)
                    stoi_scores.append(stoi_score)

                    # შეფასების სემპლის შენახვა
                    eval_samples.append({
                        'text': texts[j],
                        'original_audio': original_path,
                        'generated_audio': generated_path,
                        'spectrogram_plot': spec_plot_path,
                        'metrics': {
                            'pesq': pesq_score,
                            'stoi': stoi_score,
                            'mse': mse_score
                        }
                    })

                # სემპლების შეზღუდვა
                if len(eval_samples) >= num_samples:
                    break

        # საშუალო მეტრიკების გამოთვლა
        avg_pesq = np.mean(pesq_scores) if pesq_scores else 0.0
        avg_stoi = np.mean(stoi_scores) if stoi_scores else 0.0
        avg_mse = np.mean(mse_scores) if mse_scores else 0.0

        # მეტრიკების შეჯამება
        metrics = {
            'pesq': {
                'mean': float(avg_pesq),
                'std': float(np.std(pesq_scores)) if pesq_scores else 0.0,
                'min': float(np.min(pesq_scores)) if pesq_scores else 0.0,
                'max': float(np.max(pesq_scores)) if pesq_scores else 0.0,
            },
            'stoi': {
                'mean': float(avg_stoi),
                'std': float(np.std(stoi_scores)) if stoi_scores else 0.0,
                'min': float(np.min(stoi_scores)) if stoi_scores else 0.0,
                'max': float(np.max(stoi_scores)) if stoi_scores else 0.0,
            },
            'mse': {
                'mean': float(avg_mse),
                'std': float(np.std(mse_scores)) if mse_scores else 0.0,
                'min': float(np.min(mse_scores)) if mse_scores else 0.0,
                'max': float(np.max(mse_scores)) if mse_scores else 0.0,
            }
        }

        # მეტრიკების შენახვა
        self.metrics = metrics

        # ვიზუალიზაციების შექმნა
        self._create_visualizations(pesq_scores, stoi_scores, mse_scores)

        # შედეგების შენახვა
        results = {
            'metrics': metrics,
            'samples': eval_samples
        }

        # შედეგების ჩაწერა ფაილში
        with open(os.path.join(self.output_dir, 'validation_results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"შეფასების შედეგები შენახულია: {self.output_dir}/validation_results.json")
        print(f"საშუალო PESQ: {avg_pesq:.4f}")
        print(f"საშუალო STOI: {avg_stoi:.4f}")
        if mse_scores:
            print(f"საშუალო MSE: {avg_mse:.4f}")

        return metrics

    def _create_visualizations(self, pesq_scores, stoi_scores, mse_scores):
        """
        შეფასების მეტრიკების ვიზუალიზაცია

        Args:
            pesq_scores: PESQ მნიშვნელობები
            stoi_scores: STOI მნიშვნელობები
            mse_scores: MSE მნიშვნელობები
        """
        # ჰისტოგრამები
        plt.figure(figsize=(15, 5))

        # PESQ ჰისტოგრამა
        plt.subplot(1, 3, 1)
        plt.hist(pesq_scores, bins=10, color='blue', alpha=0.7)
        plt.axvline(np.mean(pesq_scores), color='red', linestyle='dashed', linewidth=2)
        plt.title(f'PESQ განაწილება (საშუალო: {np.mean(pesq_scores):.4f})')
        plt.xlabel('PESQ მნიშვნელობა')
        plt.ylabel('სიხშირე')

        # STOI ჰისტოგრამა
        plt.subplot(1, 3, 2)
        plt.hist(stoi_scores, bins=10, color='green', alpha=0.7)
        plt.axvline(np.mean(stoi_scores), color='red', linestyle='dashed', linewidth=2)
        plt.title(f'STOI განაწილება (საშუალო: {np.mean(stoi_scores):.4f})')
        plt.xlabel('STOI მნიშვნელობა')
        plt.ylabel('სიხშირე')

        # MSE ჰისტოგრამა (თუ არის)
        if mse_scores:
            plt.subplot(1, 3, 3)
            plt.hist(mse_scores, bins=10, color='purple', alpha=0.7)
            plt.axvline(np.mean(mse_scores), color='red', linestyle='dashed', linewidth=2)
            plt.title(f'MSE განაწილება (საშუალო: {np.mean(mse_scores):.4f})')
            plt.xlabel('MSE მნიშვნელობა')
            plt.ylabel('სიხშირე')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metrics_distribution.png'))
        plt.close()