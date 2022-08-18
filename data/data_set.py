from abc import ABC, abstractmethod


class DataSet(ABC):

    @abstractmethod
    def get_training_set(self):
        # Implementations should return the training set
        pass

    @abstractmethod
    def get_training_labels(self):
        pass

    @abstractmethod
    def get_validation_set(self):
        # Implementations should return the validation set
        pass

    @abstractmethod
    def get_validation_labels(self):
        pass

    def update_accuracy(self,   ):

