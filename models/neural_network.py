from abc import ABC, abstractmethod
from typing import Tuple, List

from data.data_set import DataSet


class NeuralNetwork(ABC):

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def train(self, training_data: DataSet, epochs) -> Tuple[List[float], List[float]]:
        pass
