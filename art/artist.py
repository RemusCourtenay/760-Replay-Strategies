from abc import ABC, abstractmethod

from data.task import TaskResult


class Artist(ABC):

    @abstractmethod
    def draw(self) -> None:
        pass

    @abstractmethod
    def add_results(self, strategy_number: int, task_results: TaskResult) -> None:
        pass
