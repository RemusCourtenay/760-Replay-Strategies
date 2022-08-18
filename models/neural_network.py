from abc import ABC, abstractmethod
from data.data_set import DataSet


class NeuralNetwork(ABC):

    @abstractmethod
    def train(self, training_data: DataSet, epochs) -> 'tuple[float, float]':
        pass
