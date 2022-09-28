import numpy as np
import numpy.typing as npt


class Task:

    def __init__(self,
                 task_num: int,
                 dataset_name: str,
                 training_set: npt.ArrayLike,
                 training_labels: npt.ArrayLike,
                 validation_set: npt.ArrayLike,
                 validation_labels: npt.ArrayLike):
        self.task_num = task_num
        self.dataset_name = dataset_name
        self.training_set = np.array(training_set)
        self.training_labels = np.array(training_labels)
        self.validation_set = np.array(validation_set)
        self.validation_labels = np.array(validation_labels)

        self.strategy_name = ""

    def set_strategy_name(self, strategy_name: str) -> None:
        self.strategy_name = strategy_name

