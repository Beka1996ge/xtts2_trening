"""
ეს სკრიპტი წარმოადგენს დამოუკიდებელ პროცესს ხმოვანი ემბედინგების გენერირებისთვის XTTS-v2 მოდელისთვის.
ის იღებს სეგმენტირებულ აუდიო ფაილებს და ქმნის მათთვის ხმოვან ემბედინგებს.
"""

import os
import argparse
import logging
import time
from pathlib import Path

# იმპორტები ხმოვანი ემბედინგების მოდულებიდან
from voice_embedding import VoiceEmbeddingGenerator
import prepare_embeddings
import embedding_visualization

# ლოგერის კონფიგურაცია
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("embeddings.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="ხმოვანი ემბედინგების გენერირება XTTS-v2 მოდელისთვის")

    # მთავარი არგუმენტები
    parser.add_argument("--segmented_dir", type=str, default="data/segmented",
                        help="სეგმენტირებული აუდიო ფაილების დირექტორია")
    parser.add_argument("--output_dir", type=str, default="data/embeddings",
                        help="ემბედინგების შედეგების დირექტორია")
    parser.add_argument("--metadata", type=str, default="data/metadata.csv",
                        help="მეტადატას ფაილის გზა (არასავალდებულო)")

    # ემბედინგის პარამეტრები
    parser.add_argument("--model", type=str, default="microsoft/wavlm-base-plus",
                        help="WavLM მოდელის სახელი (microsoft/wavlm-base-plus ან microsoft/wavlm-large)")
    parser.add_argument("--device", type=str, default=None,
                        help="გამოსაყენებელი მოწყობილობა (cuda ან cpu)")
    parser.add_argument("--segment_length", type=float, default=5.0,
                        help="სეგმენტის სიგრძე წამებში ემბედინგებისთვის")
    parser.add_argument("--segment_overlap", type=float, default=1.0,
                        help="სეგმენტების გადაფარვა წამებში ემბედინგებისთვის")

    # დამატებითი პარამეტრები
    parser.add_argument("--no_visualize", action="store_true",
                        help="ემბედინგების ვიზუალიზაციის გამორთვა")
    parser.add_argument("--overwrite", action="store_true",
                        help="არსებული ემბედინგების გადაწერა")
    parser.add_argument("--copy_to_split", type=str, default=None,
                        help="ემბედინგების კოპირება split საქაღალდეში (მაგ., data/split)")

    args = parser.parse_args()

    start_time = time.time()

    try:
        logger.info(f"იწყება ხმოვანი ემბედინგების გენერირება...")
        logger.info(f"სეგმენტირებული აუდიო დირექტორია: {args.segmented_dir}")
        logger.info(f"შედეგების დირექტორია: {args.output_dir}")

        # შევქმნათ საქაღალდეები თუ არ არსებობს
        os.makedirs(args.output_dir, exist_ok=True)

        # ემბედინგების გენერირება
        results = prepare_embeddings.prepare_embeddings(
            audio_folder=args.segmented_dir,
            output_folder=args.output_dir,
            metadata_path=args.metadata if os.path.exists(args.metadata) else None,
            model_name=args.model,
            device=args.device,
            segment_length=args.segment_length,
            segment_overlap=args.segment_overlap,
            visualize=not args.no_visualize,
            overwrite=args.overwrite
        )

        logger.info(f"შეიქმნა {results['embeddings_count']} ემბედინგი.")

        # ემბედინგების კოპირება split საქაღალდეში (თუ მითითებულია)
        if args.copy_to_split and os.path.exists(args.copy_to_split):
            logger.info(f"ემბედინგების კოპირება split საქაღალდეში: {args.copy_to_split}")
            prepare_embeddings.copy_embeddings_to_split_folders(
                embeddings_folder=os.path.join(args.output_dir, "embeddings"),
                split_folder=args.copy_to_split
            )

        end_time = time.time()
        processing_time = end_time - start_time

        logger.info(f"ემბედინგების გენერირება დასრულებულია. დრო: {processing_time:.2f} წამი")
        logger.info(f"ემბედინგები შენახულია: {os.path.join(args.output_dir, 'embeddings')}")

        if not args.no_visualize:
            logger.info(f"ვიზუალიზაცია შენახულია: {os.path.join(args.output_dir, 'visualizations')}")

        return 0

    except Exception as e:
        logger.error(f"შეცდომა პროცესის დროს: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())