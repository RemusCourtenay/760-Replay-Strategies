from strategies.selection_strategy import SelectionStrategy


class DefaultSelectionStrategy(SelectionStrategy):

    def __init__(self, model: NeuralNetwork):
        super().__init__(model)

    def train_model(self, data: DataSet):
        # TODO...
        pass
