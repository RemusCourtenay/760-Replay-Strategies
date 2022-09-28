from datetime import datetime

from art.artist import Artist
from data.task_result import TaskResult


class DefaultArtist(Artist):

    def __init__(self):
        self.results_dict = {}
        self.start = datetime.now()

    def draw(self) -> None:
        runtime = datetime.now() - self.start
        print("Script finished after " + str(runtime))

    def add_results(self, task_results: TaskResult) -> None:

        if self.results_dict.get(task_results.strategy_name) is None:
            self.results_dict[task_results.strategy_name] = []

        self.results_dict[task_results.strategy_name].append(task_results)

