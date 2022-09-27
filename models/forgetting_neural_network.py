from data.task import Task
from data.task_result import TaskResult
from models.neural_network import NeuralNetwork

import tensorflow as tf


class ForgettingNeuralNetwork(NeuralNetwork):

    INPUT_SHAPE = (28, 28, 1)

    def __init__(self):
        super().__init__(tf.keras.Sequential)
        # self.model.add(tf.keras.layers......)
        # TODO...


        self.probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])

    def reset(self):
        return ForgettingNeuralNetwork()

    def train_task(self, task: Task, epochs = 10) -> TaskResult:

        prediction_list = []
        history = None

        for i in range(epochs):
            history = self.model.fit(task.training_set, task.training_labels)
            prediction_list.append(self.probability_model.predict(task.training_set))

        return TaskResult(history, prediction_list)