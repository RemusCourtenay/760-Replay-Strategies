from abc import ABC, abstractmethod


class Artist(ABC):

    @abstractmethod
    def draw(self) -> None:
        pass

    @abstractmethod
    def add_results(self, training_data, validation_data) -> None:
        pass

    @abstractmethod
    def draw_comparisons(self, other_artist) -> None:
        pass

    @abstractmethod
    def add_policy_name(self, name) -> None:
        pass

    @abstractmethod
    def draw_multiple_comparisons(self, other_policies: list) -> None:
        pass




    # temp function to output demonstration graph
    @abstractmethod
    def draw_demonstration(self, CNNModel):
        pass
