import numpy as np
from typing import Tuple, List

from data.task import Task
from data.task_result import TaskResult
from strategies.selection_strategy import SelectionStrategy
import random


class RandomSelectionStrategy(SelectionStrategy):
    STRATEGY_NAME = "random"

    def __init__(self, strategy_name=STRATEGY_NAME):
        super().__init__(strategy_name)
        self.replay_data = []
        self.replay_label = []
        self.task_num = 1

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
        if self.task_num == 1:
            self.replay_data = old_data_subset
            self.replay_label = old_label_subset
        else:
            for i in range(num_memories):
                index = random.randint(0, num_memories * self.task_num)
                if index < num_memories:
                    self.replay_data[index] = old_data_subset[index]
                    self.replay_label[index] = old_label_subset[index]
        self.task_num += 1

        return self.replay_data, self.replay_label
