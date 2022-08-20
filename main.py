from data.mnist_data_set import MnistDataSet as DataSet
from models.simple_cnn import SimpleCNN as NeuralNetwork
from strategies.random_selection_strategy import RandomSelectionStrategy
from strategies.no_selection_strategy import NoSelectionStrategy
from strategies.selection_strategy import SelectionStrategy
from art.default_artist import DefaultArtist


def run_selection_strategy(strategy: SelectionStrategy):
    strategy.run()
    return


if __name__ == "__main__":
    print('Running Random Selection Strategy. . .')
    run_selection_strategy(RandomSelectionStrategy(NeuralNetwork(), DataSet(), DefaultArtist()))

    print('Running No Selection Strategy. . .')
    run_selection_strategy(NoSelectionStrategy(NeuralNetwork(), DataSet(), DefaultArtist()))
