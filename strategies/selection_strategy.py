from abc import ABC, abstractmethod
from typing import Tuple, List

from art.artist import Artist
from models.neural_network import NeuralNetwork
from data.data_set import DataSet


class SelectionStrategy(ABC):
    # TODO... figure out what these are supposed to be
    DEFAULT_MEMORY_PERCENT = 10.0
    DEFAULT_EPOCHS = 5

    def __init__(self, model: NeuralNetwork, data: DataSet, artist: Artist, strategy_name: str, memory_percent: float,
                 epochs: int):
        self.model = model
        self.data = data
        self.artist = artist
        self.strategy_name = strategy_name

        # Probably a better way to do this
        if memory_percent == 0:
            self.memory_percent = self.DEFAULT_MEMORY_PERCENT
        else:
            self.memory_percent = memory_percent

        if epochs == 0:
            self.epochs = self.DEFAULT_EPOCHS
        else:
            self.epochs = 0

    def run(self) -> None:
        for _ in range(self.data.get_tasks().len):
            self.select_memories(self.memory_percent)
            training_results, validation_results = self.train_model()
            self.artist.add_results(training_results, validation_results)

        self.artist.draw()

    def train_model(self) -> Tuple[List[float], List[float]]:
        return self.model.train(self.data, self.epochs)

    @abstractmethod
    def select_memories(self, percentage: float) -> None:
        pass
