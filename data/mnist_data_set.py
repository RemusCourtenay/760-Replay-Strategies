from data.data_set import DataSet
import numpy as np
import tensorflow as tf
import math

from data.task import Task


class MnistDataSet(DataSet):
    NUM_LABELS = 10

    def __init__(self, num_labels_per_task=3, labelled_validation=True):
        super().__init__(self.NUM_LABELS, num_labels_per_task)
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        self.train_tasks = self.build_tasks(train_images, train_labels, self.num_tasks, self.num_labels_per_task)
        self.validation_tasks = self.build_tasks(test_images, test_labels, self.num_tasks, self.num_labels_per_task)
        self.current_task = Task(0,
                                 self.train_tasks[0][0], self.train_tasks[0][1],
                                 self.validation_tasks[0][0], self.validation_tasks[0][1])

    # updates current task by concatenating next task and selected data
    def update_training_set(self, selected_memory_data, selected_memory_labels):
        task_index = self.current_task.task_num + 1
        new_training_data = np.array(self.train_tasks[task_index][0])
        new_training_labels = np.array(self.train_tasks[task_index][1])

        # concatenate selected data if it is not empty
        if len(selected_memory_data) > 0:
            new_training_data = np.concatenate((new_training_data, selected_memory_data), axis=0)
            new_training_labels = np.concatenate((new_training_labels, selected_memory_labels), axis=0)

        # Randomise order
        random_ordering = np.arange(len(new_training_data))
        np.random.shuffle(random_ordering)

        current_training_data = new_training_data[random_ordering]
        current_training_labels = new_training_labels[random_ordering]

        self.current_task = Task(task_index,
                                 current_training_data, current_training_labels,
                                 self.validation_tasks[task_index][0], self.validation_tasks[task_index][1])

    def get_task(self):
        return self.current_task

    # function to reset current task back to first task (used when multiple NN share one dataset)
    def reset(self):
        return MnistDataSet()
