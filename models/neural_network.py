from abc import ABC, abstractmethod

from data.task import TaskResult, Task


class NeuralNetwork(ABC):

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def train_task(self, task: Task, epochs) -> TaskResult:
        pass
