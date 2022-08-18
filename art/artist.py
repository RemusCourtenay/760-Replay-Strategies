from abc import ABC, abstractmethod


class Artist(ABC):

    @abstractmethod
    def draw(self) -> None:
        pass

    @abstractmethod
    def add_results(self, training_data, validation_data) -> None:
        pass
