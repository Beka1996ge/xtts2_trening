"""
ხმის ემბედინგის გენერირების მოდული XTTS-v2 დატრეინინგებისთვის.
ეს მოდული იყენებს WavLM მოდელს სპიკერის ხმოვანი ემბედინგების შესაქმნელად.
"""

import os
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm
import logging
from typing import List, Optional, Tuple, Dict, Union
from pathlib import Path
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
import librosa

# ლოგირების კონფიგურაცია
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VoiceEmbeddingGenerator:
    """
    ხმოვანი ემბედინგების გენერატორი, რომელიც იყენებს WavLM მოდელს
    სპიკერის ხმოვანი მახასიათებლების წარმოსადგენად ვექტორულ სივრცეში.
    """

    def __init__(
            self,
            model_name: str = "microsoft/wavlm-base-plus",
            device: Optional[str] = None,
            sample_rate: int = 16000,
            embedding_layer: int = -1,  # ბოლო ფენა
            pooling_method: str = "mean"
    ):
        """
        ინიციალიზაცია ხმოვანი ემბედინგის გენერატორისთვის.

        Args:
            model_name: WavLM მოდელის სახელი ("microsoft/wavlm-base-plus" ან "microsoft/wavlm-large")
            device: გამოსაყენებელი მოწყობილობა (None ავტომატურად აირჩევს GPU/CPU)
            sample_rate: სემპლირების სიხშირე (WavLM-ისთვის უნდა იყოს 16კჰც)
            embedding_layer: მოდელის რომელი ფენიდან უნდა ავიღოთ ემბედინგები (-1 ნიშნავს ბოლო ფენას)
            pooling_method: ემბედინგის პულინგის მეთოდი ("mean", "max" ან "attention")
        """
        # device დაყენება
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"იყენებს {self.device} მოწყობილობას")
        logger.info(f"იტვირთება WavLM მოდელი: {model_name}")

        # მოდელისა და ფიჩა ექსტრაქტორის ჩატვირთვა
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name)

        # მოდელის გადატანა მითითებულ მოწყობილობაზე
        self.model = self.model.to(self.device)
        self.model.eval()  # შეფასების რეჟიმის დაყენება

        self.sample_rate = sample_rate
        self.embedding_layer = embedding_layer
        self.pooling_method = pooling_method

        logger.info("ხმოვანი ემბედინგის გენერატორი წარმატებით ინიციალიზირებულია.")

    def preprocess_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        აუდიო ფაილის წინასწარი დამუშავება ემბედინგის გენერირებისთვის.

        Args:
            audio_path: აუდიო ფაილის გზა

        Returns:
            Tuple დამუშავებული აუდიო მონაცემებით და სემპლირების სიხშირით
        """
        try:
            # აუდიოს წაკითხვა
            speech_array, orig_sr = sf.read(audio_path)

            # სტერეოდან მონოზე გადაყვანა (თუ საჭიროა)
            if len(speech_array.shape) > 1:
                speech_array = speech_array.mean(axis=1)

            # resampling 16kHz-ზე (თუ საჭიროა)
            if orig_sr != self.sample_rate:
                speech_array = librosa.resample(
                    speech_array,
                    orig_sr=orig_sr,
                    target_sr=self.sample_rate
                )

            return speech_array, self.sample_rate

        except Exception as e:
            logger.error(f"შეცდომა აუდიოს დამუშავებისას {audio_path}: {str(e)}")
            raise

    def compute_embedding(self, audio_tensor: torch.Tensor) -> np.ndarray:
        """
        ხმოვანი ემბედინგის გამოთვლა აუდიო ტენზორიდან.

        Args:
            audio_tensor: წინასწარ დამუშავებული აუდიო ტენზორი

        Returns:
            numpy მასივი ხმოვანი ემბედინგით
        """
        try:
            # გადაყვანა ტენზორში
            inputs = {k: v.to(self.device) for k, v in audio_tensor.items()}

            # ემბედინგის გამოთვლა
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            # საჭირო ფენის ამოღება
            hidden_states = outputs.hidden_states[self.embedding_layer]

            # პულინგის მეთოდის არჩევა
            if self.pooling_method == "mean":
                embeddings = hidden_states.mean(dim=1)
            elif self.pooling_method == "max":
                embeddings = hidden_states.max(dim=1)[0]
            else:  # default to mean
                embeddings = hidden_states.mean(dim=1)

            # გადაყვანა CPU-ზე და numpy მასივში
            return embeddings.cpu().numpy()

        except Exception as e:
            logger.error(f"შეცდომა ემბედინგის გამოთვლისას: {str(e)}")
            raise

    def generate_embedding_from_file(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        ხმოვანი ემბედინგის გენერირება აუდიო ფაილიდან.

        Args:
            audio_path: აუდიო ფაილის გზა

        Returns:
            numpy მასივი ხმოვანი ემბედინგით
        """
        try:
            # აუდიოს წინასწარი დამუშავება
            speech_array, sr = self.preprocess_audio(audio_path)

            # აუდიოს გადაყვანა მოდელის საჭირო ფორმატში
            inputs = self.feature_extractor(
                speech_array,
                sampling_rate=sr,
                return_tensors="pt"
            )

            # ემბედინგის გამოთვლა
            embedding = self.compute_embedding(inputs)

            return embedding

        except Exception as e:
            logger.error(f"შეცდომა ემბედინგის გენერირებისას ფაილიდან {audio_path}: {str(e)}")
            raise

    def process_long_audio(
            self,
            audio_path: Union[str, Path],
            segment_length: float = 5.0,
            segment_overlap: float = 1.0
    ) -> np.ndarray:
        """
        გრძელი აუდიოს დამუშავება სეგმენტებად და ემბედინგების გასაშუალოება.

        Args:
            audio_path: აუდიო ფაილის გზა
            segment_length: სეგმენტის სიგრძე წამებში
            segment_overlap: სეგმენტების გადაფარვა წამებში

        Returns:
            numpy მასივი საბოლოო ხმოვანი ემბედინგით
        """
        try:
            # აუდიოს წინასწარი დამუშავება
            speech_array, sr = self.preprocess_audio(audio_path)

            # აუდიოს სიგრძე წამებში
            audio_length = len(speech_array) / sr

            # სეგმენტის ზომა და ნაბიჯი სემპლებში
            segment_samples = int(segment_length * sr)
            step_samples = int((segment_length - segment_overlap) * sr)

            # შევამოწმოთ აუდიოს სიგრძე
            if len(speech_array) < segment_samples:
                logger.warning(f"აუდიო ფაილი {audio_path} ძალიან მოკლეა. გამოიყენება მთლიანი ფაილი.")
                inputs = self.feature_extractor(
                    speech_array,
                    sampling_rate=sr,
                    return_tensors="pt"
                )
                return self.compute_embedding(inputs)

            # ემბედინგების სია
            all_embeddings = []

            # სეგმენტებად დაყოფა და ემბედინგების გამოთვლა
            for start_sample in range(0, len(speech_array) - segment_samples + 1, step_samples):
                end_sample = start_sample + segment_samples

                # სეგმენტის ამოჭრა
                segment = speech_array[start_sample:end_sample]

                # ემბედინგის გამოთვლა
                inputs = self.feature_extractor(
                    segment,
                    sampling_rate=sr,
                    return_tensors="pt"
                )

                embedding = self.compute_embedding(inputs)
                all_embeddings.append(embedding)

            # ემბედინგების გასაშუალოება
            if all_embeddings:
                final_embedding = np.mean(all_embeddings, axis=0)
                return final_embedding
            else:
                logger.error(f"ვერ მოხერხდა ემბედინგების გამოთვლა ფაილისთვის {audio_path}")
                raise ValueError("ვერ მოხერხდა ემბედინგების გამოთვლა")

        except Exception as e:
            logger.error(f"შეცდომა გრძელი აუდიოს დამუშავებისას {audio_path}: {str(e)}")
            raise

    def generate_embeddings_batch(
            self,
            audio_folder: Union[str, Path],
            output_folder: Union[str, Path],
            process_long_files: bool = True,
            segment_length: float = 5.0,
            segment_overlap: float = 1.0,
            file_extensions: List[str] = ['.wav', '.mp3', '.flac'],
            overwrite: bool = False
    ) -> Dict[str, str]:
        """
        ხმოვანი ემბედინგების გენერირება მრავალი აუდიო ფაილისთვის.

        Args:
            audio_folder: აუდიო ფაილების საქაღალდე
            output_folder: ემბედინგების შესანახი საქაღალდე
            process_long_files: გრძელი ფაილების სპეციალური დამუშავება
            segment_length: სეგმენტის სიგრძე წამებში
            segment_overlap: სეგმენტების გადაფარვა წამებში
            file_extensions: აუდიო ფაილების გაფართოებები
            overwrite: არსებული ემბედინგების გადაწერა

        Returns:
            Dict შენახული ემბედინგების გზებით
        """
        try:
            # საქაღალდის შექმნა თუ არ არსებობს
            os.makedirs(output_folder, exist_ok=True)

            # აუდიო ფაილების სია
            audio_files = []
            for ext in file_extensions:
                audio_files.extend(list(Path(audio_folder).glob(f"*{ext}")))

            if not audio_files:
                logger.warning(f"ვერ მოიძებნა აუდიო ფაილები {audio_folder} საქაღალდეში")
                return {}

            logger.info(f"ნაპოვნია {len(audio_files)} აუდიო ფაილი დასამუშავებლად")

            # შედეგების ლექსიკონი
            results = {}

            # ემბედინგების გენერირება თითოეული ფაილისთვის
            for audio_file in tqdm(audio_files, desc="ემბედინგების გენერირება"):
                try:
                    # შედეგის ფაილის სახელი
                    output_file = Path(output_folder) / f"{audio_file.stem}.npy"

                    # შევამოწმოთ არსებობს თუ არა უკვე
                    if output_file.exists() and not overwrite:
                        logger.debug(f"ემბედინგი უკვე არსებობს: {output_file}")
                        results[str(audio_file)] = str(output_file)
                        continue

                    # აუდიოს ხანგრძლივობის შემოწმება
                    duration = librosa.get_duration(path=str(audio_file))

                    # შესაბამისი მეთოდის არჩევა
                    if process_long_files and duration > segment_length * 1.5:
                        embedding = self.process_long_audio(
                            audio_file,
                            segment_length=segment_length,
                            segment_overlap=segment_overlap
                        )
                    else:
                        embedding = self.generate_embedding_from_file(audio_file)

                    # ემბედინგის შენახვა
                    np.save(output_file, embedding)

                    results[str(audio_file)] = str(output_file)

                except Exception as e:
                    logger.error(f"შეცდომა ფაილის დამუშავებისას {audio_file}: {str(e)}")
                    continue

            logger.info(f"წარმატებით დამუშავდა {len(results)} ფაილი {len(audio_files)}-დან")
            return results

        except Exception as e:
            logger.error(f"შეცდომა ბეჩის დამუშავებისას: {str(e)}")
            raise


# მთავარი ფუნქცია სკრიპტის გაშვებისთვის
def main():
    import argparse

    parser = argparse.ArgumentParser(description="ხმოვანი ემბედინგების გენერირება XTTS-v2-ისთვის")
    parser.add_argument("--audio_folder", type=str, required=True, help="აუდიო ფაილების საქაღალდე")
    parser.add_argument("--output_folder", type=str, required=True, help="ემბედინგების შესანახი საქაღალდე")
    parser.add_argument("--model", type=str, default="microsoft/wavlm-base-plus", help="WavLM მოდელის სახელი")
    parser.add_argument("--device", type=str, default=None, help="გამოსაყენებელი მოწყობილობა (cuda ან cpu)")
    parser.add_argument("--segment_length", type=float, default=5.0, help="სეგმენტის სიგრძე წამებში")
    parser.add_argument("--segment_overlap", type=float, default=1.0, help="სეგმენტების გადაფარვა წამებში")
    parser.add_argument("--overwrite", action="store_true", help="არსებული ემბედინგების გადაწერა")

    args = parser.parse_args()

    try:
        # ემბედინგის გენერატორის შექმნა
        embedding_generator = VoiceEmbeddingGenerator(
            model_name=args.model,
            device=args.device
        )

        # ემბედინგების გენერირება
        results = embedding_generator.generate_embeddings_batch(
            audio_folder=args.audio_folder,
            output_folder=args.output_folder,
            segment_length=args.segment_length,
            segment_overlap=args.segment_overlap,
            overwrite=args.overwrite
        )

        logger.info(f"წარმატებით შეიქმნა {len(results)} ემბედინგი")

    except Exception as e:
        logger.error(f"შეცდომა პროგრამის მუშაობისას: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())