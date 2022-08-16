from strategies.selection_strategy import SelectionStrategy


class DefaultSelectionStrategy(SelectionStrategy):

    def __init__(self, model: NeuralNetwork, data: DataSet):
        super().__init__(model, data)

    def train_model(self):
        # TODO...
        pass
    
    # TODO... clean this up
    def select_memories(self):
        data = self.data.training_data
        label = self.data.training_data_labels
        
        n = len(data)
        # create an array of indexes to shuffle
        s = np.arange(0, n)
        np.random.shuffle(s)
        # shuffle the data and label arrays
        data_subset = data[s]
        label_subset = label[s]

        # cut subset down to desired size
        data_subset = data_subset[:int(n*percentage)]
        label_subset = label_subset[:int(n*percentage)]
        return data_subset, label_subset
