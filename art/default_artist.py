import matplotlib.pyplot as plt
from art.artist import Artist
from data.task import TaskResult


class DefaultArtist(Artist):

    def __init__(self):
        self.results_dict = {}

    def draw(self) -> None:

        print(self.results_dict)

    def add_results(self, strategy_number: int, task_results: TaskResult) -> None:

        if self.results_dict.get(strategy_number) is None:
            self.results_dict[strategy_number] = []

        self.results_dict[strategy_number].append(task_results)

