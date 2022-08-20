from abc import ABC, abstractmethod
import numpy as np


class DataSet(ABC):

    @abstractmethod
    def update_training_set(self, selected_memory_data, selected_memory_labels):
        """Updates the currently selected training set by choosing a new task/tasks to train on and then adding any
        old data values that the selection policy decided to keep."""
        pass

    @abstractmethod
    def get_tasks(self):
        """Returns the full set of all non-validation data-label tuples as a two dimensional array sorted by task."""
        pass

    @abstractmethod
    def get_training_set(self):
        """Returns the set of currently selected data objects that are either in the currently training task or were
        selected by the selection policy"""
        pass

    @abstractmethod
    def get_training_labels(self):
        """Returns the set of currently selected data labels that are either in the currently training task or were
        selected by the selection policy"""
        pass

    @abstractmethod
    def get_validation_set(self):
        """Returns the set of validation data objects that are used to ensure that the model isn't over-fitting"""
        pass

    @abstractmethod
    def get_validation_labels(self):
        """Returns the set of validation data labels that are used to ensure that the model isn't over-fitting"""
        pass


def build_tasks(training_images, training_labels, num_tasks: int, num_labels_per_task: int) \
        -> 'list[list[list, list]]':
    # cursed list instantiation
    tasks = [[[], []] for _ in range(num_tasks)]

    for image, label in zip(training_images, training_labels):
        task_num = int(label / num_labels_per_task)
        tasks[task_num][0].append(image)
        tasks[task_num][1].append(label)

    return tasks


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

