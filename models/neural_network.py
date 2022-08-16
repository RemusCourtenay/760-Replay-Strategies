from abc import ABC, abstractmethod
from data.data_set import DataSet


class NeuralNetwork(ABC):
    
    # TODO... set default values for test_acc and epochs
    @abstractmethod
    def train(self, training_data: DataSet, test_acc, epochs):
        # TODO... figure out what methods this abstract class needs
        pass
