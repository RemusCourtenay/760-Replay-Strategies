import numpy as np

from data.data_set import DataSet
from models.neural_network import NeuralNetwork
from strategies.selection_strategy import SelectionStrategy


class DefaultSelectionStrategy(SelectionStrategy):

    def __init__(self, model: NeuralNetwork, data: DataSet):
        super().__init__(model, data)

    def train_model(self):
        # TODO...
        pass
    
    # TODO... clean this up and make it use the DataSet methods
    def select_memories(self, percentage):
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
        old_data_subset = old_data_subset[:int(n*percentage)]
        old_label_subset = old_label_subset[:int(n*percentage)]

        # Update data object's current training data
        self.data.update_training_set(old_data_subset, old_label_subset)
