from typing import List

from tensorflow.python.keras.callbacks import History


class TaskResult:

    def __init__(self, task_history: History):
        self.strategy_name = ""
        self.task_history = task_history

    def set_strategy_name(self, name: str) -> None:
        self.strategy_name = name

    def get_history(self) -> History:
        return self.task_history

    def get_num_epochs(self) -> int:
        # Checking the last value of the list might be a better way to do this
        return len(self.task_history.epoch)

    def get_loss(self) -> List[float]:
        return self.task_history.history["loss"]

    def get_accuracy(self) -> List[float]:
        return self.task_history.history["accuracy"]

    def get_val_loss(self) -> List[float]:
        return self.task_history.history["val_loss"]

    def get_val_accuracy(self) -> List[float]:
        return self.task_history.history["val_accuracy"]


