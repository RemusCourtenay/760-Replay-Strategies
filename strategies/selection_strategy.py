from abc import ABC, abstractmethod

from art import Artist
from models.neural_network import NeuralNetwork
from data.data_set import DataSet


class SelectionStrategy(ABC):

    def __init__(self, model: NeuralNetwork, data: DataSet, artist: Artist):
        self.model = model
        self.data = data
        self.artist = artist

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def train_model(self):
        pass
    
    @abstractmethod
    def select_memories(self, percentage: int):
        pass
