import numpy as np
from abc import ABC, abstractmethod

from art import artist
from models.neural_network import NeuralNetwork
from data.data_set import DataSet


class SelectionStrategy(ABC):

    def __init__(self, model: NeuralNetwork, data: DataSet, artist: artist):
        self.model = model
        self.data = data
        self.artist = artist

    @abstractmethod
    def run(self) -> None:
        pass

    @abstractmethod
    def train_model(self) -> 'tuple[float, float]':
        pass
    
    @abstractmethod
    def select_memories(self, percentage: int) -> None:
        pass
