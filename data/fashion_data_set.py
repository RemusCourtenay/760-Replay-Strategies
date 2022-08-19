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

        # Probably way overcomplicated this
        new_training_data_array = np.ndarray(self.tasks[self.current_task])
        selected_memory_data_array = np.ndarray(selected_memory_data)
        selected_memory_labels_array = np.ndarray(selected_memory_labels)
        # Put labels under images
        # TODO... ensure this doesn't break because the selected_memory_data_array is already 2D
        selected_memories = np.concatenate((selected_memory_data_array, selected_memory_labels_array), axis=0)
        # Append selected memories
        new_training_data = np.concatenate((new_training_data_array, selected_memories), axis=1)
        # Shuffle set
        np.random.shuffle(new_training_data)

        self.current_training_data = new_training_data[0]
        self.current_training_labels = new_training_data[1]

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

