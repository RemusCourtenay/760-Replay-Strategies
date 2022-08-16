from abc import ABC, abstractmethod
from model.neural_network import NeuralNetwork
from data.data_set import DataSet


class SelectionStrategy(ABC):

    def __init__(self, model: NeuralNetwork, data: DataSet):
        self.model = model
        self.data = data

    @abstractmethod
    def train_model(self):
        pass
    
    @abstractmethod
    def select_memories(self):
        pass
