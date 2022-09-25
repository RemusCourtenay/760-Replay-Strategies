from abc import ABC, abstractmethod
from typing import List, Tuple

from models.neural_network import NeuralNetwork
from data.task import Task, TaskResult


class SelectionStrategy(ABC):

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

    @abstractmethod
    def select_memories(self, task: Task, task_results: TaskResult, num_memories: int) -> Tuple[List, List]:
        pass
