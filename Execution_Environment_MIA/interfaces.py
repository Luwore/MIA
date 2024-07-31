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
