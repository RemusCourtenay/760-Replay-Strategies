from data.default_data_set import DefaultDataSet
from models.default_neural_network import DefaultNeuralNetwork
from strategies.default_selection_strategy import DefaultSelectionStrategy
from strategies.selection_strategy import SelectionStrategy


def run_selection_strategy(strategy: SelectionStrategy):
    strategy.train_model()
    return


if __name__ == "__main__":
    run_selection_strategy(DefaultSelectionStrategy(DefaultNeuralNetwork(), DefaultDataSet()))
