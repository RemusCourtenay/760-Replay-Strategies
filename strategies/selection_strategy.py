from abc import ABC, abstractmethod

from models.neural_network import NeuralNetwork
from data.task import Task, TaskResult


class SelectionStrategy(ABC):

    def __init__(self, model: NeuralNetwork, strategy_name: str):
        self.model = model
        self.strategy_name = strategy_name

    @abstractmethod
    def select_memories(self, num_memories) -> List[]:
        pass
