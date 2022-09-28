from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import keras.activations
import numpy as np

from data.task import Task
from data.task_result import TaskResult
from models.confidence_neural_network import ConfidenceNeuralNetwork
from strategies.selection_strategy import SelectionStrategy


class ConfidenceStrategy(SelectionStrategy):
    def __init__(self, model: ConfidenceNeuralNetwork):
        super().__init__('confidence')
        self.model = model

    def select_memories(self, task: Task, task_result: TaskResult, num_memories: int) -> Tuple[List, List]:

        old_training_data = task.training_set
        old_training_labels = task.training_labels

        # Generate new copy of labelling model
        self.model = self.model.reset()

        # Get list of predictions over several epochs of label fitting, we ignore History info
        predictions_list = self.model.train_task(task).get_predictions()

        # Get confidence
        scores = {}
        for predictions in predictions_list:
            for i in range(len(old_training_data)):
                score = 0
                if np.argmax(predictions[i]) == old_training_labels[i]:
                    score = max(predictions[i])
                if i in scores:
                    scores[i].append(score)
                else:
                    scores[i] = [score]

        # Use stat values to calculate forgetness
        forgetness = {}
        for index, lst in scores.items():
            forgetness[index] = max(lst)-min(lst)

        # Choose the examples with the highest confidence difference
        forgetness = sorted(forgetness.items(), key=lambda x: x[1], reverse=True)

        selected_memories = []
        selected_memory_labels = []
        for i in range(len(forgetness)):
            if len(selected_memories) < num_memories:
                selected_memories.append(old_training_data[i])
                selected_memory_labels.append(old_training_labels[i])

        return selected_memories, selected_memory_labels
