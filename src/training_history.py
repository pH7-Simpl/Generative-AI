from dataclasses import dataclass, field
import json
import os
from typing import Dict, List
import matplotlib.pyplot as plt

@dataclass
class TrainingHistory:

    epochs: List[Dict[str, float]] = field(default_factory=list)

    def add(self, metrics: Dict[str, float]):
        self.epochs.append(metrics)

    def latest(self) -> Dict[str, float]:
        if len(self.epochs) == 0:
            return {}

        return self.epochs[-1]

    def get(self, key: str) -> List[float]:
        return [
            epoch[key]
            for epoch in self.epochs
            if key in epoch
        ]

    def keys(self) -> List[str]:
        all_keys = set()

        for epoch in self.epochs:
            all_keys.update(epoch.keys())

        return sorted(list(all_keys))

    def to_dict(self) -> Dict[str, List[float]]:
        result = {}

        for key in self.keys():
            result[key] = self.get(key)

        return result
    
    def save(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        history_path = os.path.join(save_path, "history.json")

        with open(history_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

        return history_path
    
    @staticmethod
    def from_dict(data):

        history = TrainingHistory()

        keys = list(data.keys())

        if len(keys) == 0:
            return history

        num_epochs = len(data[keys[0]])

        for i in range(num_epochs):

            epoch_data = {}

            for key in keys:
                epoch_data[key] = data[key][i]

            history.add(epoch_data)

        return history
    
    @staticmethod
    def load(path: str):

        with open(path, "r") as f:
            data = json.load(f)

        return TrainingHistory.from_dict(data)
    
    def print_history(self, precision: int = 6):
        if len(self.epochs) == 0:
            print("TrainingHistory is empty.")
            return

        keys = self.keys()

        for i, epoch in enumerate(self.epochs):
            
            parts = [f"Epoch [{i+1}/{len(self.epochs)}]"]

            for key in keys:

                if key not in epoch:
                    continue

                value = epoch[key]

                if isinstance(value, float):
                    value_str = f"{value:.{precision}f}"
                else:
                    value_str = str(value)

                parts.append(f"{key}: {value_str}")

            print(" | ".join(parts))
    
    def print_latest(self, precision: int = 6):

        if len(self.epochs) == 0:
            print("TrainingHistory is empty.")
            return

        epoch = self.latest()

        parts = [f"Epoch [{len(self.epochs)}]"]

        for key in self.keys():

            if key not in epoch:
                continue

            value = epoch[key]

            if isinstance(value, float):
                value_str = f"{value:.{precision}f}"
            else:
                value_str = str(value)

            parts.append(f"{key}: {value_str}")

        print(" | ".join(parts))
    
    def plot(
        self,
        title: str = "Training History",
        save_path: str = None,
    ):

        history_dict = self.to_dict()

        plt.figure(figsize=(10, 5))

        for key, values in history_dict.items():

            if "loss" in key:
                plt.plot(values, label=key)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)

        plt.xticks(range(1, len(values) + 1))

        plt.legend()
        plt.grid(True)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()

    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):
        return self.epochs[idx]