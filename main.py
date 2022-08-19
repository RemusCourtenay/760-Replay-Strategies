from art.default_artist import DefaultArtist
from data.default_data_set import DefaultDataSet
from models.default_neural_network import DefaultNeuralNetwork
from strategies.random_selection_strategy import RandomSelectionStrategy
from strategies.selection_strategy import SelectionStrategy


def run_selection_strategy(strategy: SelectionStrategy):
    strategy.train_model()
    return


if __name__ == "__main__":
    run_selection_strategy(RandomSelectionStrategy(DefaultNeuralNetwork(), DefaultDataSet(), DefaultArtist()))
