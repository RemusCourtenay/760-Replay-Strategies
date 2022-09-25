from typing import List

from tensorflow.python.keras.callbacks import History


class Task:

    def __init__(self,
                 training_set: List,
                 training_labels: List,
                 validation_set: List,
                 validation_labels: List):
        self.training_set = training_set
        self.training_labels = training_labels
        self.validation_set = validation_set
        self.validation_labels = validation_labels


class TaskResult:

    def __init__(self, task_history: History):
        self.task_history = task_history
