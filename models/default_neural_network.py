from datetime import datetime
from typing import Tuple, List

from data.task import Task
from data.task_result import TaskResult
from models.neural_network import NeuralNetwork
import tensorflow as tf
import numpy as np


class DefaultNeuralNetwork(NeuralNetwork):

    def __init__(self,
                 num_filters_1=4, kernel_size_1=(5, 5), input_shape=(28, 28, 1),
                 max_pooling_shape=(2, 2),
                 num_filters_2=8, kernel_size_2=(3, 3),
                 activation_type='relu',
                 dense_layer_size=10):
        super().__init__(tf.keras.models.Sequential())
        self.model.add(tf.keras.layers.Conv2D(num_filters_1, kernel_size_1, activation=activation_type,
                                              input_shape=input_shape))
        self.model.add(tf.keras.layers.MaxPooling2D(max_pooling_shape))
        self.model.add(tf.keras.layers.Conv2D(num_filters_2, kernel_size_2, activation=activation_type))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(dense_layer_size, activation=activation_type))
        self.model.add(tf.keras.layers.Dense(dense_layer_size))

        # define optimizer and loss function to use
        self.model.compile(optimizer=self.DEFAULT_OPTIMIZER,
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=[self.ACCURACY_METRIC_TAG])

        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    def reset(self) -> NeuralNetwork:
        return DefaultNeuralNetwork()

    # function to train model on specified task
    def train_task(self, task: Task, epochs) -> TaskResult:
        history = self.model.fit(task.training_set,
                                 task.training_labels,
                                 epochs=epochs,
                                 validation_data=(task.validation_set, task.validation_labels),
                                 callbacks=[self.tensorboard_callback])
        # return TaskResults
        return TaskResult(history)
