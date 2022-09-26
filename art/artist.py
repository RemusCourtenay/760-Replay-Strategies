from abc import ABC, abstractmethod

from data.task_result import TaskResult


class Artist(ABC):

    @abstractmethod
    def draw(self) -> None:
        pass

    @abstractmethod
    def add_results(self, task_results: TaskResult) -> None:
        pass
