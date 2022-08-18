from abc import ABC, abstractmethod
from data.data_set import DataSet


class NeuralNetwork(ABC):

    def __init__(self):
        self.training_accuracy = []
        self.test_accuracy = []
    
    # TODO... set default values for test_acc and epochs
    @abstractmethod
    def train(self, training_data: DataSet, epochs) -> 'tuple[float, float]':
        # TODO... figure out what methods this abstract class needs
        pass
