import math
from abc import ABC, abstractmethod
import numpy as np


class DataSet(ABC):

    def __init__(self, num_labels: int, num_labels_per_task: int):
        self.num_labels = num_labels
        self.num_labels_per_task = num_labels_per_task
        self.num_tasks = math.ceil(self.num_labels / self.num_labels_per_task)

    def get_num_tasks(self) -> int:
        return self.num_tasks

    @abstractmethod
    def update_training_set(self, selected_memory_data, selected_memory_labels):
        """Updates the currently selected training set by choosing a new task/tasks to train on and then adding any
        old data values that the selection policy decided to keep."""
        pass

    @abstractmethod
    def get_task(self):
        pass

    @abstractmethod
    def reset(self):
        pass


def build_tasks(training_images, training_labels, num_tasks: int, num_labels_per_task: int, num_labels: int):
    # cursed list instantiation
    train_tasks = [[[], []] for _ in range(num_tasks)]

    # create array of labels to generate random ordering in tasks
    labels = np.arange(0, num_labels)
    # shuffle labels
    np.random.shuffle(labels)
    # store the labels in an array then split the shuffled labels into specified number of tasks
    task_labels = []
    for i in range(num_tasks):
        task_labels.append(labels[i * num_labels_per_task:((i * num_labels_per_task) + num_labels_per_task)])

    for image, label in zip(training_images, training_labels):
        # find out which task the data should belong to by checking with all the task_labels
        for i in range(num_tasks):
            if label in task_labels[i]:
                task_num = i
                break
        train_tasks[task_num][0].append(image)
        train_tasks[task_num][1].append(label)

    return train_tasks


# Shuffles data for a given dataset and label
def shuffle_data(data, label):
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


