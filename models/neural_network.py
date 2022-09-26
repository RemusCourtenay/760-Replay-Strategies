from abc import ABC, abstractmethod

from data.task import Task
from data.task_result import TaskResult


class NeuralNetwork(ABC):

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def train_task(self, task: Task, epochs) -> TaskResult:
        pass
