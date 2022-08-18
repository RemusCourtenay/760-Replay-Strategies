from abc import ABC, abstractmethod


class Artist(ABC):

    @abstractmethod
    def draw(self):
        pass

    @abstractmethod
    def add_results(self, training_data, validation_data):
        pass
