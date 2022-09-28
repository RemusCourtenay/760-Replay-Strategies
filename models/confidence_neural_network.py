from data.task import Task
from data.task_result import TaskResult
from models.neural_network import NeuralNetwork

import tensorflow as tf

from scripts.script_parameters import ScriptParameters


class ConfidenceNeuralNetwork(NeuralNetwork):
    INPUT_SHAPE = (28, 28, 1)

    def __init__(self, params: ScriptParameters):
        super().__init__(tf.keras.Sequential(), params)

        self.probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])

    def setup_layers(self) -> None:
        self.model.add(tf.keras.layers.Conv2D(4, (5, 5), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(10, activation='softmax'))
        self.model.add(tf.keras.layers.Dense(10))

    def reset(self):
        return ConfidenceNeuralNetwork(self.params)

    def train_task(self, task: Task, epochs=10) -> TaskResult:
        prediction_list = []
        history = None

        for i in range(epochs):
            history = self.model.fit(task.training_set, task.training_labels)
            prediction_list.append(self.probability_model.predict(task.training_set))

        return TaskResult(history, prediction_list)
