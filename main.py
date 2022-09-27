from data.fashion_data_set import FashionDataSet
from data.mnist_data_set import MnistDataSet
from models.default_neural_network import DefaultNeuralNetwork
from models.forgetting_neural_network import ForgettingNeuralNetwork
from scripts.script import Script
from scripts.script_parameters import ScriptParameters
from strategies.forgetting.least_forgetting_selection_strategy import LeastForgettingSelectionStrategy
from strategies.forgetting.most_forgetting_selection_strategy import MostForgettingSelectionStrategy
from strategies.random_selection_strategy import RandomSelectionStrategy
from strategies.no_selection_strategy import NoSelectionStrategy
from art.default_artist import DefaultArtist

if __name__ == "__main__":
    script = Script(DefaultNeuralNetwork(),
                    DefaultArtist(),
                    [LeastForgettingSelectionStrategy(ForgettingNeuralNetwork()),
                     MostForgettingSelectionStrategy(ForgettingNeuralNetwork()),
                     RandomSelectionStrategy(),
                     NoSelectionStrategy()],
                    [FashionDataSet(), MnistDataSet()],
                    ScriptParameters(1000, 5))  # TODO... put real numbers here

    script.run()
