from dataclasses import dataclass, field
from typing import Dict, List

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

    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):
        return self.epochs[idx]