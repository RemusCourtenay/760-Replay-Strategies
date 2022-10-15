import math
from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np
import numpy.typing as npt
import random

from data.task import Task

# set as True if storing replay samples in buffer
USE_BUFFER = True

class DataSet(ABC):

    def __init__(self, name, data, num_labels: int, num_labels_per_task: int, replay_size: int, labelled_validation: bool):
        self.name = name
        self.labelled_validation = labelled_validation
        self.replay_size = replay_size

        # Calculate number of tasks based off of total label number and labels per task
        self.num_labels = num_labels
        self.num_labels_per_task = num_labels_per_task
        self.num_tasks = math.ceil(self.num_labels / self.num_labels_per_task)

        # Setup task storage and current task object
        self.train_tasks = self.build_tasks(data[0])
        self.validation_tasks = self.build_tasks(data[1])
        self.current_task = self.build_initial_task()

        self.buffer_data = []
        self.buffer_labels = []

    def build_initial_task(self) -> Task:
        current_validation_data, current_validation_labels = self.get_validation_data_tuple(0)
        return Task(0, self.name,
                    self.train_tasks[0][0], self.train_tasks[0][1],
                    current_validation_data, current_validation_labels)

    def get_validation_data_tuple(self, task_index) -> Tuple[List, List]:
        if self.labelled_validation:
            current_validation_data = self.validation_tasks[task_index][0]
            current_validation_labels = self.validation_tasks[task_index][1]
        else:
            current_validation_data = []
            current_validation_labels = []
            for task in self.validation_tasks:
                current_validation_data = current_validation_data + task[0]
                current_validation_labels = current_validation_labels + task[1]

        return current_validation_data, current_validation_labels

    def get_num_tasks(self) -> int:
        return self.num_tasks

    def update_training_set(self, selected_memory_data: npt.ArrayLike, selected_memory_labels: npt.ArrayLike):
        """Updates the currently selected training set by choosing a new task/tasks to train on and then adding any
        old data values that the selection policy decided to keep."""
        task_index = self.current_task.task_num + 1
        new_training_data = np.array(self.train_tasks[task_index][0])
        new_training_labels = np.array(self.train_tasks[task_index][1])

        # concatenate selected data if it is not empty
        if len(selected_memory_data) > 0:
            task_number = self.current_task.get_num()
            if task_number == 0 or not USE_BUFFER:
                self.buffer_data = selected_memory_data
                self.buffer_labels = selected_memory_labels
            else:
                # if using buffer, replace a portion of the buffer with new replay samples
                # probability is 1/(tasks seen)
                for i in range(self.replay_size):
                    index = random.randint(0, self.replay_size * (task_number+1))
                    if index < self.replay_size:
                        self.buffer_data[index] = selected_memory_data[index]
                        self.buffer_labels[index] = selected_memory_labels[index]

            new_training_data = np.concatenate((new_training_data, self.buffer_data), axis=0)
            new_training_labels = np.concatenate((new_training_labels, self.buffer_labels), axis=0)

        current_training_data, current_training_labels = shuffle_labelled_data(new_training_data, new_training_labels)
        current_validation_data, current_validation_labels = self.get_validation_data_tuple(task_index)

        self.current_task = Task(task_index, self.name,
                                 current_training_data, current_training_labels,
                                 current_validation_data, current_validation_labels)

    def get_task(self, strategy_name: str):
        self.current_task.set_strategy_name(strategy_name)
        return self.current_task

    @abstractmethod
    def reset(self):
        pass

    def build_tasks(self, task_data: Tuple[List, List]):
        training_images = task_data[0]
        training_labels = task_data[1]

        # cursed list instantiation
        train_tasks = [[[], []] for _ in range(self.num_tasks)]

        # create array of labels to generate random ordering in tasks
        labels = np.arange(0, self.num_labels)
        # shuffle labels
        np.random.shuffle(labels)
        # store the labels in an array then split the shuffled labels into specified number of tasks
        # task_labels = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5]), np.array([6, 7]), np.array([8, 9])]
        task_labels = []
        for i in range(self.num_tasks):
            task_labels.append(
                labels[i * self.num_labels_per_task:((i * self.num_labels_per_task) + self.num_labels_per_task)])

        # merge last two tasks if last task is smaller
        if len(task_labels[-1]) < len(task_labels[-2]):
            last_tasks = task_labels.pop(-1)
            task_labels[-1] = np.concatenate((task_labels[-1], last_tasks))
            self.num_tasks -= 1
            
        for image, label in zip(training_images, training_labels):
            # find out which task the data should belong to by checking with all the task_labels

            for i in range(self.num_tasks):
                if label in task_labels[i]:
                    task_num = i
                    break
            train_tasks[task_num][0].append(image)
            train_tasks[task_num][1].append(label)

        return train_tasks


# Shuffles data for a given dataset and label
def shuffle_labelled_data(data, label):
    data = np.array(data)
    label = np.array(label)

    n = len(data)

    # create list of indexes and shuffle them
    indexes = np.arange(0, n)
    np.random.shuffle(indexes)

    # rearrange data and label sets using shuffled indexes
    randomised_data = data[indexes]
    randomised_label = label[indexes]

    return randomised_data, randomised_label
