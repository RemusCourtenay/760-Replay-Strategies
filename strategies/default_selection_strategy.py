import numpy as np

from art.artist import Artist
from data.data_set import DataSet
from models.neural_network import NeuralNetwork
from strategies.selection_strategy import SelectionStrategy


class DefaultSelectionStrategy(SelectionStrategy):

    # TODO... figure out what number this should be
    DEFAULT_EPOCHS = 5

    def __init__(self, model: NeuralNetwork, data: DataSet, artist: Artist):
        super().__init__(model, data, artist)

    def run(self) -> None:
        for _ in range(self.data.get_tasks().len):
            self.select_memories()
            training_results, validation_results = self.train_model()
            self.artist.add_results(training_results, validation_results)

        self.artist.draw()

    # Not sure if this needs to be a method
    def train_model(self) -> 'tuple[float, float]':
        return self.model.train(self.data, self.DEFAULT_EPOCHS)

    def select_memories(self, percentage: int = 10) -> None:
        old_training_data = self.data.get_training_set()
        old_training_labels = self.data.get_training_labels()

        n = len(old_training_data)
        # create an array of indexes to shuffle
        s = np.arange(0, n)
        np.random.shuffle(s)
        # shuffle the data and label arrays
        old_data_subset = old_training_data[s]
        old_label_subset = old_training_labels[s]

        # cut subset down to desired size
        old_data_subset = old_data_subset[:int(n * percentage)]
        old_label_subset = old_label_subset[:int(n * percentage)]

        # Update data object's current training data
        self.data.update_training_set(old_data_subset, old_label_subset)
