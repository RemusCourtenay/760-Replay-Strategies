from typing import List, Tuple

from art.artist import Artist
from data.mnist_data_set import MnistDataSet as DataSet
from data.task import TaskResult, Task
from models.simple_cnn import SimpleCNN as NeuralNetwork
from strategies.selection_strategy import SelectionStrategy
import tensorflow as tf


class NoSelectionStrategy(SelectionStrategy):
    STRATEGY_NAME = "No Selection Strategy"

    def __init__(self, strategy_name: str = STRATEGY_NAME):
        super().__init__(strategy_name)

    def select_memories(self, task: Task, task_results: TaskResult, num_memories: int) -> Tuple[List, List]:
        # We don't choose any memories so just pass None to let the update function know
        return [], []
