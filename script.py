from models.neural_network import NeuralNetwork
from art.artist import Artist
from strategies.selection_strategy import SelectionStrategy
from strategies.no_selection_strategy import NoSelectionStrategy
from strategies.random_selection_strategy import RandomSelectionStrategy
from strategies.forgetting_metric_strategy import ForgettingSelectionStrategy
from models.default_neural_network import DefaultNeuralNetwork
from typing import List


class Script:
    """Defines the run variables for a set of selection strategies and handles the running of them"""

    DEFAULT_STRATEGIES_LIST = [NoSelectionStrategy(DefaultNeuralNetwork()),
                               RandomSelectionStrategy(DefaultNeuralNetwork),
                               ForgettingSelectionStrategy(DefaultNeuralNetwork)]

    def __init__(self, model: NeuralNetwork, artist: Artist, strategies: List[SelectionStrategy] = None):
        self.model = model
        self.artist = artist
        if strategies is None:
            self.strategies = self.DEFAULT_STRATEGIES_LIST
        else:
            self.strategies = strategies

    def run(self):
        # TODO...
        pass
