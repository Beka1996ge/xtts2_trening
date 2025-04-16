#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XTTS-v2 მოდელის პრეტრეინინგის Checkpoint-ების მონიტორინგი
ავტორი:
თარიღი: 2025-04-15
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ლოგების კონფიგურაცია
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("checkpoint_monitor")


class CheckpointMonitor:
    """
    Checkpoint-ების მონიტორინგის კლასი
    """

    def __init__(self, config_path):
        """
        ინიციალიზაცია

        Args:
            config_path (str): კონფიგურაციის ფაილის გზა
        """
        # კონფიგურაციის ჩატვირთვა
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.checkpoint_dir = Path(self.config["checkpoint_dir"])
        self.output_dir = Path(self.config["output_model_path"])

        # პროგრესის ფაილი
        self.progress_file = self.output_dir / "pretrain_progress.json"

    def get_checkpoints(self):
        """
        ყველა Checkpoint-ის მოძებნა

        Returns:
            list: Checkpoint-ების გზები დალაგებული დროის მიხედვით
        """
        if not self.checkpoint_dir.exists():
            logger.error(f"Checkpoint-ების დირექტორია არ არსებობს: {self.checkpoint_dir}")
            return []

        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        checkpoints.sort(key=lambda x: int(x.stem.split("_")[1]))

        logger.info(f"ნაპოვნია {len(checkpoints)} Checkpoint")
        return checkpoints

    def get_progress(self):
        """
        პროგრესის მდგომარეობის მიღება

        Returns:
            dict: პროგრესის მდგომარეობა
        """
        if not self.progress_file.exists():
            logger.error(f"პროგრესის ფაილი არ არსებობს: {self.progress_file}")
            return {}

        with open(self.progress_file, 'r', encoding='utf-8') as f:
            progress = json.load(f)

        return progress

    def load_checkpoint_metadata(self, checkpoint_path):
        """
        Checkpoint-ის მეტადატას ჩატვირთვა

        Args:
            checkpoint_path (Path): Checkpoint-ის გზა

        Returns:
            dict: Checkpoint-ის მეტადატა
        """
        try:
            # პირდაპირ მთლიანი Checkpoint-ის ჩატვირთვის ნაცვლად, მხოლოდ საჭირო ინფორმაციას ვიღებთ
            metadata = torch.load(checkpoint_path, map_location="cpu")

            # თუ ეს არის model_state_dict, ვიღებთ მხოლოდ საჭირო ინფორმაციას
            if isinstance(metadata, dict) and "model_state_dict" in metadata:
                return {
                    "step": metadata.get("step", 0),
                    "loss": metadata.get("loss", None),
                    "timestamp": metadata.get("timestamp", None),
                    "model_parameters": sum(p.numel() for p in metadata["model_state_dict"].values())
                }

            return {"error": "არავალიდური Checkpoint ფორმატი"}

        except Exception as e:
            logger.error(f"შეცდომა Checkpoint-ის ჩატვირთვისას {checkpoint_path}: {str(e)}")
            return {"error": str(e)}

    def get_checkpoint_stats(self):
        """
        ყველა Checkpoint-ის სტატისტიკის მიღება

        Returns:
            pd.DataFrame: Checkpoint-ების სტატისტიკა
        """
        checkpoints = self.get_checkpoints()
        if not checkpoints:
            return pd.DataFrame()

        stats = []

        for checkpoint in checkpoints:
            try:
                step = int(checkpoint.stem.split("_")[1])

                # ფაილის ზომა და შექმნის დრო
                size_mb = checkpoint.stat().st_size / (1024 * 1024)
                modified_time = datetime.fromtimestamp(checkpoint.stat().st_mtime)

                # მეტადატას ჩატვირთვა
                metadata = self.load_checkpoint_metadata(checkpoint)

                checkpoint_info = {
                    "step": step,
                    "path": str(checkpoint),
                    "size_mb": size_mb,
                    "modified_time": modified_time,
                }

                # მეტადატას დამატება
                checkpoint_info.update({k: v for k, v in metadata.items() if k != "step"})

                stats.append(checkpoint_info)

            except Exception as e:
                logger.error(f"შეცდომა Checkpoint-ის დამუშავებისას {checkpoint}: {str(e)}")

        df = pd.DataFrame(stats)

        # შევამოწმოთ, გვაქვს თუ არა timestamp-ები
        if "modified_time" in df.columns and df["modified_time"].notna().any():
            df = df.sort_values("modified_time")

            # დროის ინტერვალების გამოთვლა
            if len(df) > 1:
                df["time_diff"] = df["modified_time"].diff().apply(
                    lambda x: x.total_seconds() / 60 if pd.notna(x) else None)

                # ნაბიჯების ინტერვალების გამოთვლა
                df["step_diff"] = df["step"].diff()

                # ნაბიჯების თითოეულ წუთზე (steps/minute)
                df["steps_per_minute"] = df["step_diff"] / df["time_diff"]

        return df

    def plot_training_progress(self, stats_df, output_path=None):
        """
        ტრენინგის პროგრესის ვიზუალიზაცია

        Args:
            stats_df (pd.DataFrame): Checkpoint-ების სტატისტიკა
            output_path (str): შედეგების შესანახი გზა
        """
        if stats_df.empty:
            logger.error("ვიზუალიზაციისთვის მონაცემები არ არის")
            return

        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        # 1. Loss-ის გრაფიკი
        if "loss" in stats_df.columns and stats_df["loss"].notna().any():
            axs[0].plot(stats_df["step"], stats_df["loss"])
            axs[0].set_xlabel("ნაბიჯი")
            axs[0].set_ylabel("Loss")
            axs[0].set_title("ტრენინგის Loss")
            axs[0].grid(True)

        # 2. ნაბიჯები/წუთში გრაფიკი
        if "steps_per_minute" in stats_df.columns and stats_df["steps_per_minute"].notna().any():
            axs[1].plot(stats_df["step"], stats_df["steps_per_minute"])
            axs[1].set_xlabel("ნაბიჯი")
            axs[1].set_ylabel("ნაბიჯები/წუთში")
            axs[1].set_title("ტრენინგის სიჩქარე")
            axs[1].grid(True)

        # 3. Checkpoint-ების ზომა
        if "size_mb" in stats_df.columns:
            axs[2].plot(stats_df["step"], stats_df["size_mb"])
            axs[2].set_xlabel("ნაბიჯი")
            axs[2].set_ylabel("ზომა (MB)")
            axs[2].set_title("Checkpoint-ების ზომა")
            axs[2].grid(True)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
            logger.info(f"გრაფიკები შენახულია: {output_path}")
        else:
            plt.show()

    def estimate_completion(self, stats_df, total_steps):
        """
        ტრენინგის დასრულების სავარაუდო დროის გამოთვლა

        Args:
            stats_df (pd.DataFrame): Checkpoint-ების სტატისტიკა
            total_steps (int): ჯამური ნაბიჯების რაოდენობა

        Returns:
            dict: სავარაუდო დასრულების ინფორმაცია
        """
        if stats_df.empty or "steps_per_minute" not in stats_df.columns:
            return {"error": "არასაკმარისი მონაცემები სავარაუდო დროის გამოსათვლელად"}

        current_step = stats_df["step"].max()
        remaining_steps = total_steps - current_step

        if remaining_steps <= 0:
            return {"status": "completed", "completion_percentage": 100}

        # ბოლო 5 Checkpoint-ის საშუალო სიჩქარე
        recent_speed = stats_df["steps_per_minute"].tail(5).mean()

        if pd.isna(recent_speed) or recent_speed <= 0:
            return {"error": "არავალიდური ტრენინგის სიჩქარე"}

        estimated_minutes = remaining_steps / recent_speed
        estimated_completion = datetime.now() + timedelta(minutes=estimated_minutes)

        completion_percentage = (current_step / total_steps) * 100

        return {
            "current_step": current_step,
            "total_steps": total_steps,
            "remaining_steps": remaining_steps,
            "completion_percentage": completion_percentage,
            "recent_speed_steps_per_minute": recent_speed,
            "estimated_minutes_remaining": estimated_minutes,
            "estimated_completion_time": estimated_completion.strftime("%Y-%m-%d %H:%M:%S")
        }

    def run(self, plot=False, watch=False, interval=60):
        """
        მონიტორინგის გაშვება

        Args:
            plot (bool): თუ True, პროგრესის გრაფიკის გენერაცია მოხდება
            watch (bool): თუ True, განახლებების მუდმივი მონიტორინგი მოხდება
            interval (int): მონიტორინგის ინტერვალი წამებში

        Returns:
            pd.DataFrame: Checkpoint-ების სტატისტიკა
        """
        total_steps = self.config.get("max_steps", self.config.get("epochs", 100) * 1000)

        try:
            if watch:
                logger.info(f"Checkpoint-ების მონიტორინგის დაწყება, ინტერვალი: {interval} წამი")

                while True:
                    stats_df = self.get_checkpoint_stats()

                    if not stats_df.empty:
                        # პროგრესის შეფასება
                        completion_info = self.estimate_completion(stats_df, total_steps)

                        # ინფორმაციის ჩვენება
                        print("\n" + "=" * 50)
                        print(f"დრო: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                        if "error" in completion_info:
                            print(f"შეცდომა: {completion_info['error']}")
                        else:
                            print(f"ნაბიჯი: {completion_info['current_step']} / {completion_info['total_steps']} "
                                  f"({completion_info['completion_percentage']:.2f}%)")
                            print(f"დარჩენილი ნაბიჯები: {completion_info['remaining_steps']}")
                            print(f"სიჩქარე: {completion_info['recent_speed_steps_per_minute']:.2f} ნაბიჯი/წუთში")
                            print(
                                f"სავარაუდო დრო დასრულებამდე: {completion_info['estimated_minutes_remaining']:.2f} წუთი")
                            print(f"სავარაუდო დასრულების დრო: {completion_info['estimated_completion_time']}")

                            # ბოლო Checkpoint-ის ინფორმაცია
                            last_checkpoint = stats_df.iloc[-1]
                            print(f"\nბოლო Checkpoint:")
                            print(f"  გზა: {last_checkpoint['path']}")
                            print(f"  ზომა: {last_checkpoint['size_mb']:.2f} MB")
                            print(f"  შექმნის დრო: {last_checkpoint['modified_time']}")

                            if "loss" in last_checkpoint and pd.notna(last_checkpoint["loss"]):
                                print(f"  Loss: {last_checkpoint['loss']:.6f}")

                        print("=" * 50)

                        # გრაფიკების გენერაცია თუ საჭიროა
                        if plot:
                            plot_path = self.output_dir / "checkpoint_progress.png"
                            self.plot_training_progress(stats_df, plot_path)

                    else:
                        print("\nვერ მოიძებნა Checkpoint-ები")

                    # დაცდა შემდეგი განახლებამდე
                    time.sleep(interval)

            else:
                # ერთჯერადი მონიტორინგი
                stats_df = self.get_checkpoint_stats()

                if not stats_df.empty:
                    # პროგრესის შეფასება
                    completion_info = self.estimate_completion(stats_df, total_steps)

                    # ინფორმაციის ჩვენება
                    print("\n" + "=" * 50)

                    if "error" in completion_info:
                        print(f"შეცდომა: {completion_info['error']}")
                    else:
                        print(f"ნაბიჯი: {completion_info['current_step']} / {completion_info['total_steps']} "
                              f"({completion_info['completion_percentage']:.2f}%)")
                        print(f"დარჩენილი ნაბიჯები: {completion_info['remaining_steps']}")
                        print(f"სიჩქარე: {completion_info['recent_speed_steps_per_minute']:.2f} ნაბიჯი/წუთში")
                        print(f"სავარაუდო დრო დასრულებამდე: {completion_info['estimated_minutes_remaining']:.2f} წუთი")
                        print(f"სავარაუდო დასრულების დრო: {completion_info['estimated_completion_time']}")

                    print("=" * 50)

                    # Checkpoint-ების სტატისტიკის ჩვენება
                    print("\nCheckpoint-ების სტატისტიკა:")
                    print(f"სულ Checkpoint-ები: {len(stats_df)}")

                    if "size_mb" in stats_df.columns:
                        print(f"საშუალო ზომა: {stats_df['size_mb'].mean():.2f} MB")

                    if "steps_per_minute" in stats_df.columns and stats_df["steps_per_minute"].notna().any():
                        print(f"საშუალო სიჩქარე: {stats_df['steps_per_minute'].mean():.2f} ნაბიჯი/წუთში")

                    # გრაფიკების გენერაცია თუ საჭიროა
                    if plot:
                        plot_path = self.output_dir / "checkpoint_progress.png"
                        self.plot_training_progress(stats_df, plot_path)

                else:
                    print("\nვერ მოიძებნა Checkpoint-ები")

                return stats_df

        except KeyboardInterrupt:
            logger.info("მონიტორინგი შეწყვეტილია მომხმარებლის მიერ")
            return None

    def cleanup_old_checkpoints(self, keep_last=5, keep_every=10):
        """
        ძველი Checkpoint-ების გასუფთავება, რომ შევინახოთ მხოლოდ საჭირო ფაილები

        Args:
            keep_last (int): ბოლო რამდენი Checkpoint შევინახოთ
            keep_every (int): ყოველი რამდენიმე Checkpoint შევინახოთ

        Returns:
            int: წაშლილი Checkpoint-ების რაოდენობა
        """
        checkpoints = self.get_checkpoints()
        if not checkpoints or len(checkpoints) <= keep_last:
            return 0

        # Checkpoint-ები მივიღოთ ნაბიჯების მიხედვით
        steps = [int(cp.stem.split("_")[1]) for cp in checkpoints]

        to_keep = set()

        # ბოლო keep_last ნაბიჯები
        to_keep.update(steps[-keep_last:])

        # ყოველი მე-keep_every ნაბიჯი
        for i, step in enumerate(steps):
            if i % keep_every == 0:
                to_keep.add(step)

        # წაშლა
        deleted_count = 0
        for checkpoint, step in zip(checkpoints, steps):
            if step not in to_keep:
                try:
                    checkpoint.unlink()
                    deleted_count += 1
                    logger.info(f"წაშლილია Checkpoint: {checkpoint}")
                except Exception as e:
                    logger.error(f"შეცდომა Checkpoint-ის წაშლისას {checkpoint}: {str(e)}")

        logger.info(f"გასუფთავებულია {deleted_count} Checkpoint, დარჩენილია {len(checkpoints) - deleted_count}")
        return deleted_count

    def analyze_loss_trends(self, window_size=10):
        """
        Loss-ის ტრენდების ანალიზი

        Args:
            window_size (int): მცოცავი საშუალოს ფანჯრის ზომა

        Returns:
            dict: ანალიზის შედეგები
        """
        stats_df = self.get_checkpoint_stats()

        if stats_df.empty or "loss" not in stats_df.columns or not stats_df["loss"].notna().any():
            return {"error": "Loss-ის მონაცემები არ არის ხელმისაწვდომი"}

        # მცოცავი საშუალო
        stats_df["loss_ma"] = stats_df["loss"].rolling(window=window_size).mean()

        # ბოლო window_size ნაბიჯის ტრენდი
        recent_loss = stats_df["loss"].tail(window_size)
        recent_trend = np.polyfit(range(len(recent_loss)), recent_loss, 1)[0]

        # საუკეთესო loss
        best_loss = stats_df["loss"].min()
        best_step = stats_df.loc[stats_df["loss"].idxmin(), "step"]

        # ბოლო loss
        last_loss = stats_df["loss"].iloc[-1]
        last_step = stats_df["step"].iloc[-1]

        # სტაბილიზაციის ანალიზი
        stability = stats_df["loss"].tail(window_size).std() / stats_df["loss"].tail(window_size).mean()

        results = {
            "best_loss": best_loss,
            "best_step": best_step,
            "last_loss": last_loss,
            "last_step": last_step,
            "recent_trend": recent_trend,
            "stability": stability,
            "converging": recent_trend < 0,
            "plateau": abs(recent_trend) < 0.0001,
            "diverging": recent_trend > 0.0001,
            "stable": stability < 0.05
        }

        return results

    def export_stats(self, output_path=None):
        """
        სტატისტიკის ექსპორტი CSV ფაილში

        Args:
            output_path (str): შედეგების შესანახი გზა

        Returns:
            str: შენახული ფაილის გზა
        """
        stats_df = self.get_checkpoint_stats()

        if stats_df.empty:
            logger.error("ექსპორტისთვის მონაცემები არ არის")
            return None

        if output_path is None:
            output_path = self.output_dir / "checkpoint_stats.csv"

        stats_df.to_csv(output_path, index=False)
        logger.info(f"სტატისტიკა ექსპორტირებულია: {output_path}")

        return output_path


def main():
    """
    მთავარი ფუნქცია
    """
    parser = argparse.ArgumentParser(description="XTTS-v2 Checkpoint-ების მონიტორინგი")
    parser.add_argument("--config", type=str, default="xtts_pretrain_config.json",
                        help="კონფიგურაციის ფაილის გზა")
    parser.add_argument("--plot", action="store_true",
                        help="პროგრესის გრაფიკის გენერაცია")
    parser.add_argument("--watch", action="store_true",
                        help="განახლებების მუდმივი მონიტორინგი")
    parser.add_argument("--interval", type=int, default=60,
                        help="მონიტორინგის ინტერვალი წამებში (მხოლოდ --watch რეჟიმში)")
    parser.add_argument("--cleanup", action="store_true",
                        help="ძველი Checkpoint-ების გასუფთავება")
    parser.add_argument("--keep-last", type=int, default=5,
                        help="გასუფთავებისას შენახული ბოლო Checkpoint-ების რაოდენობა")
    parser.add_argument("--keep-every", type=int, default=10,
                        help="გასუფთავებისას შენახული Checkpoint-ების ინტერვალი")
    parser.add_argument("--analyze", action="store_true",
                        help="Loss-ის ტრენდების ანალიზი")
    parser.add_argument("--export", action="store_true",
                        help="სტატისტიკის ექსპორტი CSV ფაილში")
    args = parser.parse_args()

    # მონიტორის ინიციალიზაცია
    monitor = CheckpointMonitor(args.config)

    # ძველი Checkpoint-ების გასუფთავება
    if args.cleanup:
        monitor.cleanup_old_checkpoints(args.keep_last, args.keep_every)
        return

    # Loss-ის ტრენდების ანალიზი
    if args.analyze:
        results = monitor.analyze_loss_trends()

        print("\n" + "=" * 50)
        print("Loss-ის ანალიზის შედეგები:")

        if "error" in results:
            print(f"შეცდომა: {results['error']}")
        else:
            print(f"საუკეთესო Loss: {results['best_loss']:.6f} (ნაბიჯი: {results['best_step']})")
            print(f"ბოლო Loss: {results['last_loss']:.6f} (ნაბიჯი: {results['last_step']})")
            print(f"ბოლო ტრენდი: {results['recent_trend']:.6f}")
            print(f"სტაბილურობის ინდექსი: {results['stability']:.6f}")

            status = []
            if results["converging"]:
                status.append("კონვერგენცია")
            if results["plateau"]:
                status.append("პლატო")
            if results["diverging"]:
                status.append("დივერგენცია")
            if results["stable"]:
                status.append("სტაბილური")

            print(f"სტატუსი: {', '.join(status)}")

        print("=" * 50)
        return

    # სტატისტიკის ექსპორტი
    if args.export:
        monitor.export_stats()
        return

    # მონიტორინგის გაშვება
    monitor.run(args.plot, args.watch, args.interval)


if __name__ == "__main__":
    main()