import numpy as np
from typing import Tuple, List

from data.task import Task, TaskResult
from strategies.selection_strategy import SelectionStrategy


class RandomSelectionStrategy(SelectionStrategy):
    STRATEGY_NAME = "Random Selection Strategy"

    def __init__(self, strategy_name=STRATEGY_NAME):
        super().__init__(strategy_name)

    def select_memories(self, task: Task, task_result: TaskResult, num_memories: int) -> Tuple[List, List]:
        old_training_data = task.training_set
        old_training_labels = task.training_labels

        n = len(old_training_data)
        # create an array of indexes to shuffle
        s = np.arange(0, n)
        np.random.shuffle(s)
        # shuffle the data and label arrays
        old_data_subset = old_training_data[s]
        old_label_subset = old_training_labels[s]

        # cut subset down to desired size
        old_data_subset = old_data_subset[:num_memories]
        old_label_subset = old_label_subset[:num_memories]

        return old_data_subset, old_label_subset
