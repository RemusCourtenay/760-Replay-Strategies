from typing import List
import numpy.typing as npt

from data.data_set import DataSet, build_tasks

import tensorflow as tf
import numpy as np

from data.task import Task


class FashionDataSet(DataSet):
    Image = npt.NDArray[np.int_]

    NUM_TASKS = 5
    LABELS_PER_TASK = 2

    def __init__(self):
        super().__init__(self.NUM_TASKS)

        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

        self.tasks = build_tasks(train_images, train_labels, self.NUM_TASKS, self.LABELS_PER_TASK)
        self.task_validations = build_tasks(test_images, test_labels, self.NUM_TASKS, self.LABELS_PER_TASK)
        self.current_task = Task(0,
                                 self.tasks[0][0], self.tasks[0][1],
                                 self.task_validations[0][0], self.task_validations[0][1])

    def update_training_set(self,
                            selected_memory_data: npt.ArrayLike, selected_memory_labels: npt.ArrayLike):
        task_num = self.current_task.task_num + 1

        if len(selected_memory_data) > 0: # Cursed method to handle no selection strategy
            # Combine task data with selected memory data
            current_task_data = np.concatenate((self.tasks[task_num][0], selected_memory_data), axis=0)
            current_task_labels = np.concatenate((self.tasks[task_num][1], selected_memory_labels), axis=0)
        else:
            current_task_data = np.array(self.tasks[task_num][0])
            current_task_labels = np.array(self.tasks[task_num][1])

        # Randomise order
        random_ordering = np.arange(len(current_task_data))
        np.random.shuffle(random_ordering)

        current_task_data = current_task_data[random_ordering]
        current_task_labels = current_task_labels[random_ordering]

        self.current_task = Task(task_num,
                                 current_task_data, current_task_labels,
                                 self.task_validations[task_num][0], self.task_validations[task_num][1])

    def get_task(self):
        return self.current_task

    def reset(self):
        return FashionDataSet()
