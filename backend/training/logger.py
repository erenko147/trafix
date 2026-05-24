"""
Training metrics logger — writes CSV + JSON logs and configures Python logging.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class TrainingLogger:
    """Logs per-episode metrics to CSV and JSON; configures the root logger."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path  = self.output_dir / "training_log.csv"
        self.json_path = self.output_dir / "training_history.json"
        self.history: List[Dict] = []

        with open(self.csv_path, "w") as f:
            f.write(
                "episode,reward_mean,reward_std,policy_loss,value_loss,"
                "entropy,avg_speed,avg_waiting,total_vehicles,lr,"
                "entropy_coef,timestamp\n"
            )

        log_path = self.output_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_path, encoding="utf-8"),
                logging.StreamHandler(sys.stdout),
            ],
        )

    def log_episode(self, episode: int, data: Dict):
        self.history.append({"episode": episode, **data})

        with open(self.csv_path, "a") as f:
            f.write(
                f"{episode},{data.get('reward_mean', 0):.6f},"
                f"{data.get('reward_std', 0):.6f},"
                f"{data.get('policy_loss', 0):.6f},"
                f"{data.get('value_loss', 0):.6f},"
                f"{data.get('entropy', 0):.6f},"
                f"{data.get('avg_speed', 0):.4f},"
                f"{data.get('avg_waiting', 0):.4f},"
                f"{data.get('total_vehicles', 0)},"
                f"{data.get('lr', 0):.8f},"
                f"{data.get('entropy_coef', 0):.6f},"
                f"{datetime.now().isoformat()}\n"
            )

    def save_history(self):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
