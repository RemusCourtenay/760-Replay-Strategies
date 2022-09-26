import numpy as np
import numpy.typing as npt
from tensorflow.python.keras.callbacks import History


class Task:

    def __init__(self,
                 task_num: int,
                 training_set: npt.ArrayLike,
                 training_labels: npt.ArrayLike,
                 validation_set: npt.ArrayLike,
                 validation_labels: npt.ArrayLike):
        self.task_num = task_num
        self.training_set = np.array(training_set)
        self.training_labels = np.array(training_labels)
        self.validation_set = np.array(validation_set)
        self.validation_labels = np.array(validation_labels)


class TaskResult:

    def __init__(self, task_history: History):
        self.task_history = task_history
