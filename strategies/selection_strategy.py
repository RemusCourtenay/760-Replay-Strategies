from abc import ABC, abstractmethod

from art.artist import Artist
from models.neural_network import NeuralNetwork
from data.data_set import DataSet


class SelectionStrategy(ABC):

    # TODO... figure out what these are supposed to be
    DEFAULT_MEMORY_PERCENT = 10
    DEFAULT_EPOCHS = 5

    def __init__(self, model: NeuralNetwork, data: DataSet, artist: Artist, memory_percent=0, epochs=0):
        self.model = model
        self.data = data
        self.artist = artist

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

    @abstractmethod
    def train_model(self) -> 'tuple[float, float]':
        pass
    
    @abstractmethod
    def select_memories(self, percentage: int) -> None:
        pass
