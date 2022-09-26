from data.fashion_data_set import FashionDataSet
from data.mnist_data_set import MnistDataSet
from models.default_neural_network import DefaultNeuralNetwork
from scripts.script import Script
from scripts.script_parameters import ScriptParameters
from strategies.random_selection_strategy import RandomSelectionStrategy
from strategies.no_selection_strategy import NoSelectionStrategy
from art.default_artist import DefaultArtist

if __name__ == "__main__":
    script = Script(DefaultNeuralNetwork(),
                    DefaultArtist(),
                    [RandomSelectionStrategy(), NoSelectionStrategy()],
                    [FashionDataSet(), MnistDataSet()],
                    ScriptParameters(100, 1))  # TODO... put real numbers here

    script.run()
