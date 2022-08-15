from abc import ABC, abstractmethod


class SelectionStrategy(ABC):

    def __init__(self, model: NeuralNetwork):
        self.model = model

    @abstractmethod
    def train_model(self, data: DataSet):
        pass
