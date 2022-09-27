from typing import Dict

from models.forgetting_neural_network import ForgettingNeuralNetwork
from scripts.script_parameters import ScriptParameters
from strategies.forgetting.forgetting_selection_strategy import ForgettingSelectionStrategy


class LeastForgettingSelectionStrategy(ForgettingSelectionStrategy):
    STRATEGY_NAME = "Least Forgetting Selection Strategy"

    def __init__(self, model: ForgettingNeuralNetwork):
        super().__init__(model, self.STRATEGY_NAME)

    def sort_forgetness(self, forgetness: Dict) -> Dict:
        return dict(sorted(forgetness.items(), key=lambda item: item[1]))
