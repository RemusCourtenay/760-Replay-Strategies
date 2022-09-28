from typing import Dict

from models.forgetting_neural_network import ForgettingNeuralNetwork
from strategies.forgetting.forgetting_selection_strategy import ForgettingSelectionStrategy


class MostForgettingSelectionStrategy(ForgettingSelectionStrategy):
    STRATEGY_NAME = "most-forgetting"

    def __init__(self, model: ForgettingNeuralNetwork):
        super().__init__(model, self.STRATEGY_NAME)

    def sort_forgetness(self, forgetness: Dict) -> Dict:
        return dict(sorted(forgetness.items(), key=lambda item: item[1], reverse=True))
