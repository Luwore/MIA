from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class AbstractDataLoader(ABC):
    data_path: str = field(default="data")
    sets: Dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def load_data(self, file_name: str) -> None:
        """Load the data, check if it's available locally or needs to be downloaded."""
        pass

    @abstractmethod
    def preprocess_data(self) -> None:
        """Preprocess the data and split it into different sets."""
        pass

    @abstractmethod
    def get_data(self, set_name: str) -> Any:
        """Get data from a specific set."""
        return self.sets.get(set_name, None)


class AttackInterface(ABC):

    @abstractmethod
    def train_target_model(self, *args, **kwargs):
        """
        Train the target model on the provided data.
        """
        pass

    @abstractmethod
    def train_shadow_model(self, *args, **kwargs):
        """
        Train the shadow model(s) on the provided data.
        """
        pass

    @abstractmethod
    def perform_attack(self, *args, **kwargs):
        """
        Execute the attack using the trained models and data.
        """
        pass

    @abstractmethod
    def calculate_score(self, *args, **kwargs):
        """
        Calculate the score or effectiveness of the attack.
        """
        pass
