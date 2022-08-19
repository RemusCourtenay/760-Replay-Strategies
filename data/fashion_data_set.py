from typing import List, Any
import numpy.typing as npt

from data.data_set import DataSet, build_tasks

import tensorflow as tf
import numpy as np


class FashionDataSet(DataSet):
    Image = npt.NDArray[np.int_]

    NUM_TASKS = 10
    LABELS_PER_TASK = 2

    def __init__(self):
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

        self.tasks = build_tasks(train_images, train_labels, self.NUM_TASKS, self.LABELS_PER_TASK)
        self.validation_data = test_images
        self.validation_labels = test_labels

        self.current_task = 0
        self.current_training_data = np.array(self.tasks[0][0])
        self.current_training_labels = np.array(self.tasks[0][1])

    def update_training_set(self,
                            selected_memory_data: npt.ArrayLike, selected_memory_labels: npt.ArrayLike):
        self.current_task += 1

        current_task_tuple = self.tasks[self.current_task]
        current_task_data = current_task_tuple[0]
        current_task_labels = current_task_tuple[1]

        random_ordering = np.arange(len(current_task_data) + len(selected_memory_data))

        data = np.concatenate(current_task_data, selected_memory_data)
        labels = np.concatenate(current_task_labels, selected_memory_labels)

        self.current_training_data = data[random_ordering]
        self.current_training_labels = labels[random_ordering]

    def get_tasks(self) -> List[List[List[npt.NDArray[np.uint8]], List[np.uint8]]]:
        return self.tasks.copy()

    def get_training_set(self) -> npt.NDArray[np.uint8]:
        return self.current_training_data.copy()

    def get_training_labels(self) -> npt.NDArray[np.uint8]:
        return self.current_training_labels.copy()

    def get_validation_set(self) -> npt.NDArray[np.uint8]:
        return self.validation_data.copy()

    def get_validation_labels(self) -> npt.NDArray[np.uint8]:
        return self.validation_labels.copy()
