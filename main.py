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
import math

if __name__ == "__main__":
    # TODO... put real numbers here
    MNIST_DATA_SIZE = 60000
    MNIST_LABELS = 10
    REPLAY_MEMORY_PERCENT = 0.1
    NUM_LABELS_PER_TASK = 2



    num_task = math.floor(MNIST_LABELS / NUM_LABELS_PER_TASK)
    replay_mem = int((MNIST_DATA_SIZE / num_task) * REPLAY_MEMORY_PERCENT)
    print(replay_mem)

    parameters = ScriptParameters(replay_mem, 10)

    script = Script(DefaultNeuralNetwork(parameters),
                    DefaultArtist(),
                    [NoSelectionStrategy(), RandomSelectionStrategy(),
                     LeastForgettingSelectionStrategy(ForgettingNeuralNetwork(parameters)),
                     MostForgettingSelectionStrategy(ForgettingNeuralNetwork(parameters))],
                    [MnistDataSet(num_labels_per_task=NUM_LABELS_PER_TASK)],
                    parameters)

    script.run()
