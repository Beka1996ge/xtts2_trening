"""
ხმოვანი ემბედინგების მომზადების სკრიპტი XTTS-v2 დატრეინინგებისთვის.
ეს სკრიპტი გამოიყენება ჯამში ყველა სეგმენტირებული აუდიო ფაილისთვის
ხმოვანი ემბედინგების შესაქმნელად და XTTS ტრეინინგისთვის მოსამზადებლად.
"""

import os
import shutil
import json
import glob
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

# ლოკალური მოდულების იმპორტი
from voice_embedding import VoiceEmbeddingGenerator
import embedding_visualization

# ლოგირების კონფიგურაცია
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_embeddings(
        audio_folder: Union[str, Path],
        output_folder: Union[str, Path],
        metadata_path: Optional[Union[str, Path]] = None,
        model_name: str = "microsoft/wavlm-base-plus",
        device: Optional[str] = None,
        segment_length: float = 5.0,
        segment_overlap: float = 1.0,
        visualize: bool = True,
        overwrite: bool = False
) -> Dict[str, str]:
    """
    ხმოვანი ემბედინგების მომზადება XTTS-v2 დატრეინინგებისთვის.

    Args:
        audio_folder: აუდიო ფაილების საქაღალდე
        output_folder: ემბედინგების შესანახი საქაღალდე
        metadata_path: მეტადატას ფაილის გზა
        model_name: WavLM მოდელის სახელი
        device: გამოსაყენებელი მოწყობილობა
        segment_length: სეგმენტის სიგრძე წამებში
        segment_overlap: სეგმენტების გადაფარვა წამებში
        visualize: ვიზუალიზაციის შექმნა
        overwrite: არსებული ემბედინგების გადაწერა

    Returns:
        Dict ემბედინგების გზებით და სხვა ინფორმაციით
    """
    try:
        # საქაღალდეების მომზადება
        embeddings_folder = os.path.join(output_folder, "embeddings")
        os.makedirs(embeddings_folder, exist_ok=True)

        # ვიზუალიზაციის საქაღალდე
        if visualize:
            viz_folder = os.path.join(output_folder, "visualizations")
            os.makedirs(viz_folder, exist_ok=True)

        # მეტადატას ჩატვირთვა თუ არსებობს
        metadata = None
        if metadata_path and os.path.exists(metadata_path):
            try:
                metadata = pd.read_csv(metadata_path)
                logger.info(f"მეტადატა ჩატვირთულია: {len(metadata)} ჩანაწერი")
            except Exception as e:
                logger.warning(f"მეტადატას ჩატვირთვის შეცდომა: {str(e)}")

        # ემბედინგის გენერატორის შექმნა
        embedding_generator = VoiceEmbeddingGenerator(
            model_name=model_name,
            device=device
        )

        # ემბედინგების გენერირება
        results = embedding_generator.generate_embeddings_batch(
            audio_folder=audio_folder,
            output_folder=embeddings_folder,
            process_long_files=True,
            segment_length=segment_length,
            segment_overlap=segment_overlap,
            overwrite=overwrite
        )

        # ემბედინგების ვიზუალიზაცია
        if visualize and results:
            try:
                # ვიზუალიზატორის შექმნა
                visualizer = embedding_visualization.EmbeddingVisualizer()

                # ემბედინგების ჩატვირთვა
                embeddings, file_names = visualizer.load_embeddings(embeddings_folder)

                # ჯგუფების ამოღება ფაილის სახელებიდან
                groups = visualizer.extract_groups_from_filenames(file_names)

                # t-SNE ვიზუალიზაცია
                embeddings_tsne = visualizer.apply_tsne(embeddings)

                # შევინახოთ t-SNE ვიზუალიზაცია
                tsne_png_path = os.path.join(viz_folder, "embeddings_tsne.png")
                visualizer.visualize_embeddings(
                    embeddings_tsne,
                    file_names,
                    groups,
                    title="ხმოვანი ემბედინგების t-SNE ვიზუალიზაცია",
                    output_path=tsne_png_path
                )

                # ინტერაქტიული ვიზუალიზაცია
                try:
                    tsne_html_path = os.path.join(viz_folder, "embeddings_tsne_interactive.html")
                    visualizer.visualize_embeddings_interactive(
                        embeddings_tsne,
                        file_names,
                        groups,
                        title="ხმოვანი ემბედინგების ინტერაქტიული ვიზუალიზაცია (t-SNE)",
                        output_path=tsne_html_path
                    )
                except Exception as e:
                    logger.warning(f"ინტერაქტიული ვიზუალიზაციის შეცდომა: {str(e)}")

                # UMAP ვიზუალიზაცია (თუ ხელმისაწვდომია)
                try:
                    embeddings_umap = visualizer.apply_umap(embeddings)

                    # შევინახოთ UMAP ვიზუალიზაცია
                    umap_png_path = os.path.join(viz_folder, "embeddings_umap.png")
                    visualizer.visualize_embeddings(
                        embeddings_umap,
                        file_names,
                        groups,
                        title="ხმოვანი ემბედინგების UMAP ვიზუალიზაცია",
                        output_path=umap_png_path
                    )

                    # ინტერაქტიული UMAP ვიზუალიზაცია
                    try:
                        umap_html_path = os.path.join(viz_folder, "embeddings_umap_interactive.html")
                        visualizer.visualize_embeddings_interactive(
                            embeddings_umap,
                            file_names,
                            groups,
                            title="ხმოვანი ემბედინგების ინტერაქტიული ვიზუალიზაცია (UMAP)",
                            output_path=umap_html_path
                        )
                    except Exception as e:
                        logger.warning(f"ინტერაქტიული UMAP ვიზუალიზაციის შეცდომა: {str(e)}")

                except Exception as e:
                    logger.warning(f"UMAP ვიზუალიზაციის შეცდომა: {str(e)}")

            except Exception as e:
                logger.warning(f"ვიზუალიზაციის შეცდომა: {str(e)}")

        # მეტადატას განახლება ემბედინგების გზებით, თუ არსებობს
        if metadata is not None:
            # ფაილების სახელების და ემბედინგების გზების ლექსიკონის შექმნა
            embedding_paths = {}
            for audio_path, emb_path in results.items():
                audio_filename = os.path.basename(audio_path)
                audio_name = os.path.splitext(audio_filename)[0]
                embedding_paths[audio_name] = emb_path

            # მეტადატაში ემბედინგების სვეტის დამატება
            try:
                # ფაილის სახელების სვეტის მოძებნა
                filename_columns = [col for col in metadata.columns if 'file' in col.lower() or 'name' in col.lower()]

                if filename_columns:
                    filename_col = filename_columns[0]

                    # ემბედინგის გზების დამატება
                    metadata['embedding_path'] = metadata[filename_col].apply(
                        lambda x: embedding_paths.get(os.path.splitext(x)[0], '')
                    )

                    # შენახვა
                    metadata_output_path = os.path.join(output_folder, "metadata_with_embeddings.csv")
                    metadata.to_csv(metadata_output_path, index=False)
                    logger.info(f"განახლებული მეტადატა შენახულია: {metadata_output_path}")
                else:
                    logger.warning("მეტადატაში ვერ მოიძებნა ფაილის სახელის სვეტი")
            except Exception as e:
                logger.warning(f"მეტადატას განახლების შეცდომა: {str(e)}")

        # XTTS კონფიგურაციის განახლება ემბედინგების ინფორმაციით
        update_xtts_config(
            output_folder,
            model_name=model_name,
            embedding_dimension=list(map(lambda x: np.load(x).shape, list(results.values())))[0] if results else None
        )

        return {
            "embeddings_folder": embeddings_folder,
            "embeddings_count": len(results),
            "config_updated": True
        }

    except Exception as e:
        logger.error(f"შეცდომა ემბედინგების მომზადებისას: {str(e)}")
        raise


def update_xtts_config(
        output_folder: Union[str, Path],
        model_name: str = "microsoft/wavlm-base-plus",
        embedding_dimension: Optional[Tuple[int, ...]] = None
):
    """
    XTTS კონფიგურაციის ფაილის განახლება ემბედინგების ინფორმაციით.

    Args:
        output_folder: შედეგების საქაღალდე
        model_name: გამოყენებული მოდელის სახელი
        embedding_dimension: ემბედინგის განზომილება
    """
    try:
        config_path = os.path.join(output_folder, "xtts_config.json")

        # არსებული კონფიგურაციის ჩატვირთვა თუ არსებობს
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

        # ემბედინგების ინფორმაციის დამატება
        if "model" not in config:
            config["model"] = {}

        if "speaker_encoder" not in config["model"]:
            config["model"]["speaker_encoder"] = {}

        # ემბედინგის ინფორმაციის განახლება
        config["model"]["speaker_encoder"]["model_name"] = model_name
        config["model"]["speaker_encoder"]["type"] = "wavlm"

        if embedding_dimension:
            flat_dim = np.prod(embedding_dimension)
            config["model"]["speaker_encoder"]["embedding_dim"] = int(flat_dim)

        # კონფიგურაციის შენახვა
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"XTTS კონფიგურაცია განახლებულია: {config_path}")

    except Exception as e:
        logger.error(f"კონფიგურაციის განახლების შეცდომა: {str(e)}")


def copy_embeddings_to_split_folders(
        embeddings_folder: Union[str, Path],
        split_folder: Union[str, Path]
):
    """
    ემბედინგების კოპირება split საქაღალდეებში.

    Args:
        embeddings_folder: ემბედინგების საქაღალდე
        split_folder: დაყოფილი მონაცემების საქაღალდე
    """
    try:
        # დაყოფილი მონაცემების ქვესაქაღალდეების მოძიება
        split_subfolders = ['train', 'val', 'test']

        for split in split_subfolders:
            split_path = os.path.join(split_folder, split)

            if not os.path.exists(split_path):
                logger.warning(f"საქაღალდე არ არსებობს: {split_path}")
                continue

            # აუდიო საქაღალდის მოძიება
            audio_folder = os.path.join(split_path, "audio")
            if not os.path.exists(audio_folder):
                logger.warning(f"აუდიო საქაღალდე არ არსებობს: {audio_folder}")
                continue

            # ემბედინგების საქაღალდის შექმნა
            split_embeddings_folder = os.path.join(split_path, "embeddings")
            os.makedirs(split_embeddings_folder, exist_ok=True)

            # აუდიო ფაილების სია
            audio_files = glob.glob(os.path.join(audio_folder, "*.wav"))

            if not audio_files:
                logger.warning(f"აუდიო ფაილები არ არსებობს: {audio_folder}")
                continue

            # აუდიო ფაილების შესაბამისი ემბედინგების კოპირება
            copied_count = 0
            for audio_file in tqdm(audio_files, desc=f"{split} ემბედინგების კოპირება"):
                audio_name = os.path.splitext(os.path.basename(audio_file))[0]
                embedding_file = os.path.join(embeddings_folder, f"{audio_name}.npy")

                if os.path.exists(embedding_file):
                    # ემბედინგის კოპირება
                    dest_path = os.path.join(split_embeddings_folder, f"{audio_name}.npy")
                    shutil.copy2(embedding_file, dest_path)
                    copied_count += 1

            logger.info(f"დასრულდა {split} ემბედინგების კოპირება: {copied_count}/{len(audio_files)}")

    except Exception as e:
        logger.error(f"ემბედინგების კოპირების შეცდომა: {str(e)}")


# მთავარი ფუნქცია სკრიპტის გაშვებისთვის
def main():
    parser = argparse.ArgumentParser(description="ხმოვანი ემბედინგების მომზადება XTTS-v2 დატრეინინგებისთვის")
    parser.add_argument("--audio_folder", type=str, required=True, help="აუდიო ფაილების საქაღალდე")
    parser.add_argument("--output_folder", type=str, required=True, help="შედეგების საქაღალდე")
    parser.add_argument("--metadata", type=str, help="მეტადატას ფაილის გზა (CSV)")
    parser.add_argument("--model", type=str, default="microsoft/wavlm-base-plus", help="WavLM მოდელის სახელი")
    parser.add_argument("--device", type=str, help="გამოსაყენებელი მოწყობილობა (cuda ან cpu)")
    parser.add_argument("--segment_length", type=float, default=5.0, help="სეგმენტის სიგრძე წამებში")
    parser.add_argument("--segment_overlap", type=float, default=1.0, help="სეგმენტების გადაფარვა წამებში")
    parser.add_argument("--no_visualize", action="store_true", help="ვიზუალიზაციის გამორთვა")
    parser.add_argument("--overwrite", action="store_true", help="არსებული ემბედინგების გადაწერა")
    parser.add_argument("--copy_to_split", type=str, help="ემბედინგების კოპირება split საქაღალდეებში")

    args = parser.parse_args()

    try:
        # ხმოვანი ემბედინგების მომზადება
        results = prepare_embeddings(
            audio_folder=args.audio_folder,
            output_folder=args.output_folder,
            metadata_path=args.metadata,
            model_name=args.model,
            device=args.device,
            segment_length=args.segment_length,
            segment_overlap=args.segment_overlap,
            visualize=not args.no_visualize,
            overwrite=args.overwrite
        )

        logger.info(f"შეიქმნა {results['embeddings_count']} ემბედინგი")

        # ემბედინგების კოპირება split საქაღალდეებში თუ მითითებულია
        if args.copy_to_split:
            copy_embeddings_to_split_folders(
                embeddings_folder=os.path.join(args.output_folder, "embeddings"),
                split_folder=args.copy_to_split
            )

    except Exception as e:
        logger.error(f"შეცდომა პროგრამის მუშაობისას: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())