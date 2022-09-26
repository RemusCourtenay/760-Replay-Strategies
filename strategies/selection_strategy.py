from abc import ABC, abstractmethod
from typing import List, Tuple

from data.task import Task
from data.task_result import TaskResult


class SelectionStrategy(ABC):

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

    @abstractmethod
    def select_memories(self, task: Task, task_results: TaskResult, num_memories: int) -> Tuple[List, List]:
        pass
