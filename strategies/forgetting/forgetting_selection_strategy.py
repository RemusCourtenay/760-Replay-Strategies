from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import numpy as np

from data.task import Task
from data.task_result import TaskResult
from models.forgetting_neural_network import ForgettingNeuralNetwork
from strategies.selection_strategy import SelectionStrategy


class ForgettingSelectionStrategy(SelectionStrategy, ABC):

    # Forgetting statistics is obtained through another NN, NN must be either a ForgettingNeuralNetwork or an
    # implementation of ForgettingNeuralNetwork in order to gain access to prediction data
    def __init__(self, model: ForgettingNeuralNetwork, strategy_name: str):
        super().__init__(strategy_name)
        self.model = model

    def select_memories(self, task: Task, task_result: TaskResult, num_memories: int) -> Tuple[List, List]:

        old_training_data = task.training_set
        old_training_labels = task.training_labels

        # Generate new copy of labelling model
        self.model = self.model.reset()

        # Get list of predictions over several epochs of label fitting, we ignore History info
        predictions_list = self.model.train_task(task).get_predictions()

        # Use predictions to calculate stat values
        stat = {}
        for predictions in predictions_list:
            for i in range(len(old_training_data)):
                if i in stat:
                    stat[i].append(np.argmax(predictions[i]) == old_training_labels[i])
                else:
                    stat[i] = [np.argmax(predictions[i]) == old_training_labels[i]]

        # Use stat values to calculate forgetness
        forgetness = {}
        for index, lst in stat.items():
            acc_full = np.array(list(map(int, lst)))
            transition = acc_full[1:] - acc_full[:-1]
            if len(np.where(transition == -1)[0]) > 0:
                forgetness[index] = len(np.where(transition == -1)[0])
            elif len(np.where(acc_full == 1)[0]) == 0:
                forgetness[index] = len(predictions_list)
            else:
                forgetness[index] = 0

        # (1) Choose the examples with the highest forgetting statistics
        forgetness = self.sort_forgetness(forgetness)

        selected_memories = []
        selected_memory_labels = []
        for i, _ in forgetness.items():
            if len(selected_memories) < num_memories:
                selected_memories.append(old_training_data[i])
                selected_memory_labels.append(old_training_labels[i])

        return selected_memories, selected_memory_labels

    @abstractmethod
    def sort_forgetness(self, forgetness: Dict) -> Dict:
        pass
