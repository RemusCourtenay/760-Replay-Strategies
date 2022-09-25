from abc import ABC, abstractmethod

from data.data_set import DataSet
from data.task import TaskResult


class NeuralNetwork(ABC):

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def train_task(self, training_data: DataSet, epochs) -> TaskResult:
        pass
