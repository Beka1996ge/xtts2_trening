"""
ხმოვანი ემბედინგების ვიზუალიზაციის სკრიპტი XTTS-v2 დატრეინინგებისთვის.
ეს სკრიპტი ახდენს ემბედინგების ვიზუალიზაციას t-SNE ან UMAP მეთოდების გამოყენებით,
რათა გამოავლინოს კლასტერები და შეაფასოს ემბედინგების ხარისხი.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging
from typing import List, Dict, Optional, Union, Tuple
import pandas as pd
import glob
from sklearn.manifold import TSNE

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# ლოგირების კონფიგურაცია
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingVisualizer:
    """
    ხმოვანი ემბედინგების ვიზუალიზატორი t-SNE და UMAP მეთოდების გამოყენებით.
    """

    def __init__(self,
                 random_state: int = 42,
                 figsize: Tuple[int, int] = (16, 10),
                 dpi: int = 150):
        """
        ინიციალიზაცია ემბედინგების ვიზუალიზატორისთვის.

        Args:
            random_state: შემთხვევითი გენერატორის მდგომარეობა
            figsize: გრაფიკის ზომა
            dpi: გრაფიკის გარჩევადობა
        """
        self.random_state = random_state
        self.figsize = figsize
        self.dpi = dpi

        # UMAP ხელმისაწვდომობის შემოწმება
        if not UMAP_AVAILABLE:
            logger.warning("UMAP ბიბლიოთეკა არ არის დაინსტალირებული. მხოლოდ t-SNE იქნება ხელმისაწვდომი.")

    def load_embeddings(self, embeddings_folder: Union[str, Path]) -> Tuple[np.ndarray, List[str]]:
        """
        ემბედინგების ჩატვირთვა ფოლდერიდან.

        Args:
            embeddings_folder: ემბედინგების ფოლდერის გზა

        Returns:
            Tuple ემბედინგების მასივით და ფაილების სახელებით
        """
        try:
            # ემბედინგების ფაილების მოძიება
            embeddings_files = list(Path(embeddings_folder).glob("*.npy"))

            if not embeddings_files:
                logger.error(f"ემბედინგები ვერ მოიძებნა საქაღალდეში {embeddings_folder}")
                raise FileNotFoundError(f"ემბედინგები ვერ მოიძებნა საქაღალდეში {embeddings_folder}")

            logger.info(f"ნაპოვნია {len(embeddings_files)} ემბედინგი.")

            # ემბედინგების და ფაილის სახელების მასივები
            embeddings_list = []
            file_names = []

            # ემბედინგების ჩატვირთვა
            for emb_file in embeddings_files:
                try:
                    embedding = np.load(emb_file)

                    # თუ ემბედინგი 2D-ზე მეტია, გავასწოროთ
                    if embedding.ndim > 2:
                        embedding = embedding.reshape(embedding.shape[0], -1)

                    # თუ ემბედინგი არის ბეჩი (მაგ., [1, 768]), ავიღოთ პირველი ვექტორი
                    if embedding.ndim == 2 and embedding.shape[0] == 1:
                        embedding = embedding.squeeze(0)

                    embeddings_list.append(embedding)
                    file_names.append(emb_file.stem)
                except Exception as e:
                    logger.warning(f"ფაილის ჩატვირთვის შეცდომა {emb_file}: {str(e)}")
                    continue

            if not embeddings_list:
                logger.error("ვერ მოხერხდა ემბედინგების ჩატვირთვა")
                raise ValueError("ვერ მოხერხდა ემბედინგების ჩატვირთვა")

            # ემბედინგების დამრგვალება
            try:
                embeddings_array = np.vstack(embeddings_list)
                logger.info(f"ჩატვირთულია {embeddings_array.shape} ფორმის ემბედინგები.")
                return embeddings_array, file_names
            except ValueError:
                # ემბედინგების ზომები შეიძლება განსხვავდებოდეს
                logger.error("ემბედინგები განსხვავებული ზომებისაა, შეუძლებელია მათი დამრგვალება")
                for i, emb in enumerate(embeddings_list):
                    logger.error(f"ემბედინგი {i}, ფორმა: {emb.shape}")
                raise ValueError("ემბედინგები განსხვავებული ზომებისაა")

        except Exception as e:
            logger.error(f"შეცდომა ემბედინგების ჩატვირთვისას: {str(e)}")
            raise

    def apply_tsne(self, embeddings: np.ndarray, n_components: int = 2, perplexity: float = 30.0) -> np.ndarray:
        """
        t-SNE ალგორითმის გამოყენება ემბედინგების განზომილების შესამცირებლად.

        Args:
            embeddings: ემბედინგების მასივი
            n_components: შედეგის განზომილება
            perplexity: t-SNE პერპლექსია

        Returns:
            შემცირებული განზომილების ემბედინგების მასივი
        """
        try:
            logger.info(f"t-SNE გამოიყენება {embeddings.shape[0]} ემბედინგზე...")

            tsne = TSNE(
                n_components=n_components,
                perplexity=min(perplexity, embeddings.shape[0] - 1),
                random_state=self.random_state,
                n_jobs=-1
            )

            embeddings_reduced = tsne.fit_transform(embeddings)
            logger.info(f"t-SNE დასრულდა, შედეგის ფორმა: {embeddings_reduced.shape}")

            return embeddings_reduced

        except Exception as e:
            logger.error(f"შეცდომა t-SNE-ს გამოყენებისას: {str(e)}")
            raise

    def apply_umap(self, embeddings: np.ndarray, n_components: int = 2, n_neighbors: int = 15) -> np.ndarray:
        """
        UMAP ალგორითმის გამოყენება ემბედინგების განზომილების შესამცირებლად.

        Args:
            embeddings: ემბედინგების მასივი
            n_components: შედეგის განზომილება
            n_neighbors: მეზობლების რაოდენობა UMAP-ისთვის

        Returns:
            შემცირებული განზომილების ემბედინგების მასივი
        """
        if not UMAP_AVAILABLE:
            logger.error("UMAP ბიბლიოთეკა არ არის დაინსტალირებული")
            raise ImportError("UMAP ბიბლიოთეკა არ არის დაინსტალირებული")

        try:
            logger.info(f"UMAP გამოიყენება {embeddings.shape[0]} ემბედინგზე...")

            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=min(n_neighbors, embeddings.shape[0] - 1),
                random_state=self.random_state
            )

            embeddings_reduced = reducer.fit_transform(embeddings)
            logger.info(f"UMAP დასრულდა, შედეგის ფორმა: {embeddings_reduced.shape}")

            return embeddings_reduced

        except Exception as e:
            logger.error(f"შეცდომა UMAP-ის გამოყენებისას: {str(e)}")
            raise

    def visualize_embeddings(
            self,
            embeddings_reduced: np.ndarray,
            file_names: List[str],
            groups: Optional[List[str]] = None,
            title: str = "ხმოვანი ემბედინგების ვიზუალიზაცია",
            output_path: Optional[Union[str, Path]] = None,
            show_labels: bool = True,
            label_fontsize: int = 8
    ):
        """
        ემბედინგების ვიზუალიზაცია scatter plot-ის გამოყენებით.

        Args:
            embeddings_reduced: შემცირებული განზომილების ემბედინგები
            file_names: ფაილების სახელები
            groups: ჯგუფების სახელები (სპიკერები, ენები და ა.შ.)
            title: გრაფიკის სათაური
            output_path: შედეგის ფაილის გზა
            show_labels: წერტილებზე ლეიბლების ჩვენება
            label_fontsize: ლეიბლების ფონტის ზომა
        """
        try:
            plt.figure(figsize=self.figsize, dpi=self.dpi)

            # ფერების პალიტრის შერჩევა
            if groups is not None:
                unique_groups = list(set(groups))
                colors = sns.color_palette("husl", len(unique_groups))
                color_map = {group: color for group, color in zip(unique_groups, colors)}
                point_colors = [color_map[group] for group in groups]

                # ლეგენდის ელემენტების შექმნა
                handles = [plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=color_map[group], markersize=8, label=group)
                           for group in unique_groups]

                # წერტილების დახაზვა
                scatter = plt.scatter(
                    embeddings_reduced[:, 0],
                    embeddings_reduced[:, 1],
                    c=point_colors,
                    alpha=0.7,
                    s=50
                )

                # ლეგენდის დამატება
                plt.legend(handles=handles, title="ჯგუფები", loc="best")

            else:
                # ჯგუფების გარეშე ვიზუალიზაცია
                scatter = plt.scatter(
                    embeddings_reduced[:, 0],
                    embeddings_reduced[:, 1],
                    alpha=0.7,
                    s=50
                )

            # ლეიბლების დამატება
            if show_labels and file_names:
                for i, file_name in enumerate(file_names):
                    plt.annotate(
                        file_name,
                        (embeddings_reduced[i, 0], embeddings_reduced[i, 1]),
                        fontsize=label_fontsize,
                        alpha=0.7
                    )

            # გრაფიკის გაფორმება
            plt.title(title, fontsize=14)
            plt.xlabel('კომპონენტი 1', fontsize=12)
            plt.ylabel('კომპონენტი 2', fontsize=12)
            plt.tight_layout()

            # შენახვა თუ მითითებულია გზა
            if output_path:
                plt.savefig(output_path, bbox_inches='tight')
                logger.info(f"გრაფიკი შენახულია: {output_path}")

            plt.show()

        except Exception as e:
            logger.error(f"შეცდომა ვიზუალიზაციის დროს: {str(e)}")
            raise

    def extract_groups_from_filenames(self, file_names: List[str], pattern: str = r"(\w+)_") -> List[str]:
        """
        ჯგუფების ამოღება ფაილის სახელებიდან რეგულარული გამოსახულების გამოყენებით.

        Args:
            file_names: ფაილის სახელების სია
            pattern: რეგულარული გამოსახულება ჯგუფის ამოსაღებად

        Returns:
            ჯგუფების სია
        """
        import re
        groups = []

        for name in file_names:
            match = re.search(pattern, name)
            if match:
                groups.append(match.group(1))
            else:
                groups.append("unknown")

        return groups

    def visualize_embeddings_interactive(
            self,
            embeddings_reduced: np.ndarray,
            file_names: List[str],
            groups: Optional[List[str]] = None,
            title: str = "ინტერაქტიული ხმოვანი ემბედინგების ვიზუალიზაცია",
            output_path: Optional[Union[str, Path]] = None
    ):
        """
        ემბედინგების ინტერაქტიული ვიზუალიზაცია plotly-ის გამოყენებით.

        Args:
            embeddings_reduced: შემცირებული განზომილების ემბედინგები
            file_names: ფაილების სახელები
            groups: ჯგუფების სახელები (სპიკერები, ენები და ა.შ.)
            title: გრაფიკის სათაური
            output_path: შედეგის HTML ფაილის გზა
        """
        try:
            # plotly-ის იმპორტი
            import plotly.express as px
            import plotly.graph_objects as go

            # მონაცემების მომზადება
            data = {
                'x': embeddings_reduced[:, 0],
                'y': embeddings_reduced[:, 1],
                'filename': file_names
            }

            if groups:
                data['group'] = groups

            df = pd.DataFrame(data)

            # გრაფიკის შექმნა
            if groups:
                fig = px.scatter(
                    df, x='x', y='y', color='group',
                    hover_name='filename', title=title
                )
            else:
                fig = px.scatter(
                    df, x='x', y='y',
                    hover_name='filename', title=title
                )

            # გრაფიკის გაფორმება
            fig.update_traces(marker=dict(size=10))
            fig.update_layout(
                title=dict(text=title, font=dict(size=16)),
                xaxis=dict(title='კომპონენტი 1'),
                yaxis=dict(title='კომპონენტი 2'),
                hovermode='closest'
            )

            # შენახვა თუ მითითებულია გზა
            if output_path:
                fig.write_html(output_path)
                logger.info(f"ინტერაქტიული გრაფიკი შენახულია: {output_path}")

            return fig

        except ImportError:
            logger.warning("plotly არ არის დაინსტალირებული, ინტერაქტიული ვიზუალიზაცია ვერ შესრულდება")
            self.visualize_embeddings(
                embeddings_reduced, file_names, groups, title, output_path
            )
        except Exception as e:
            logger.error(f"შეცდომა ინტერაქტიული ვიზუალიზაციის დროს: {str(e)}")
            raise


# მთავარი ფუნქცია
def main():
    parser = argparse.ArgumentParser(description="ხმოვანი ემბედინგების ვიზუალიზაცია XTTS-v2 დატრეინინგებისთვის.")
    parser.add_argument("--embeddings_folder", type=str, required=True, help="ემბედინგების შემცველი საქაღალდე")
    parser.add_argument("--output", type=str, default="embeddings_visualization.png", help="შედეგის ფაილის გზა")
    parser.add_argument("--method", type=str, choices=["tsne", "umap"], default="tsne",
                        help="განზომილების შემცირების მეთოდი")
    parser.add_argument("--interactive", action="store_true", help="ინტერაქტიული გრაფიკის შექმნა (plotly საჭიროა)")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE პერპლექსია")
    parser.add_argument("--n_neighbors", type=int, default=15, help="UMAP მეზობლების რაოდენობა")
    parser.add_argument("--group_pattern", type=str, default=r"(\w+)_",
                        help="რეგულარული გამოსახულება ჯგუფების ამოსაღებად")
    parser.add_argument("--show_labels", action="store_true", help="ლეიბლების ჩვენება წერტილებზე")

    args = parser.parse_args()

    try:
        # ვიზუალიზატორის შექმნა
        visualizer = EmbeddingVisualizer()

        # ემბედინგების ჩატვირთვა
        embeddings, file_names = visualizer.load_embeddings(args.embeddings_folder)

        # ჯგუფების ამოღება
        groups = visualizer.extract_groups_from_filenames(file_names, args.group_pattern)

        # განზომილების შემცირება
        if args.method == "tsne":
            embeddings_reduced = visualizer.apply_tsne(embeddings, perplexity=args.perplexity)
        else:  # umap
            embeddings_reduced = visualizer.apply_umap(embeddings, n_neighbors=args.n_neighbors)

        # ვიზუალიზაცია
        if args.interactive:
            output_path = args.output.replace('.png', '.html')
            visualizer.visualize_embeddings_interactive(
                embeddings_reduced, file_names, groups,
                title=f"ხმოვანი ემბედინგების ვიზუალიზაცია ({args.method.upper()})",
                output_path=output_path
            )
        else:
            visualizer.visualize_embeddings(
                embeddings_reduced, file_names, groups,
                title=f"ხმოვანი ემბედინგების ვიზუალიზაცია ({args.method.upper()})",
                output_path=args.output,
                show_labels=args.show_labels
            )

        logger.info("ვიზუალიზაცია წარმატებით დასრულდა")

    except Exception as e:
        logger.error(f"შეცდომა: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())