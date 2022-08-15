from abc import ABC, abstractmethod


class DataSet(ABC):

    @abstractmethod
    def get_training_set(self):
        # Implementations should return the training set
        pass

    @abstractmethod
    def get_validation_set(self):
        # Implementations should return the validation set
        pass

