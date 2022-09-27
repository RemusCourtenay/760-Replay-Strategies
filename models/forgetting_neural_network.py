from data.task import Task
from data.task_result import TaskResult
from models.neural_network import NeuralNetwork

import tensorflow as tf


class ForgettingNeuralNetwork(NeuralNetwork):

    INPUT_SHAPE = (28, 28, 1)

    def __init__(self):
        super().__init__(tf.keras.Sequential)
        self.model.add(tf.keras.layers......)
        # TODO...

    def reset(self):
        return ForgettingNeuralNetwork()

    def train_task(self, task: Task, epochs) -> TaskResult:
        # I will implement
        pass