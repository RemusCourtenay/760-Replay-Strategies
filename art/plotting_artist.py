from art.artist import Artist
from data.task_result import TaskResult


class PlottingArtist(Artist):

    def __init__(self):
        self.results_dict = {}

    def draw(self) -> None:
        # Implement pyplot stuff here using the data from self.results_dict
        pass

    def add_results(self, task_results: TaskResult) -> None:
        if self.results_dict.get(task_results.strategy_name) is None:
            self.results_dict[task_results.strategy_name] = []

        self.results_dict[task_results.strategy_name].append(task_results)
