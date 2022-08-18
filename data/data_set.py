from abc import ABC, abstractmethod


class DataSet(ABC):

    @abstractmethod
    def update_training_set(self, selected_memory_data, selected_memory_labels):
        pass

    @abstractmethod
    def get_tasks(self):
        pass

    @abstractmethod
    def get_training_set(self):
        # Implementations should return the training set
        pass

    @abstractmethod
    def get_training_labels(self):
        # Implementations should return the training labels
        pass

    @abstractmethod
    def get_validation_set(self):
        # Implementations should return the validation set
        pass

    @abstractmethod
    def get_validation_labels(self):
        # Implementations should return the validation labels
        pass
