from abc import ABC, abstractmethod


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
