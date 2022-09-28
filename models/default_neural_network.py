from datetime import datetime
from typing import Tuple, List

from data.task import Task
from data.task_result import TaskResult
from models.neural_network import NeuralNetwork
import tensorflow as tf
import numpy as np

from scripts.script_parameters import ScriptParameters


class DefaultNeuralNetwork(NeuralNetwork):

    def __init__(self, params: ScriptParameters, time=None):
        super().__init__(tf.keras.models.Sequential(), params)

        self.log_dir = "logs/default/"
        if time is None:
            self.time = datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            self.time = time
        self.tensorboard_callback = None

    def setup_layers(self) -> None:
        self.model.add(tf.keras.layers.Conv2D(self.params.num_filters_1,
                                              self.params.kernel_size_1,
                                              activation=self.params.activation_type,
                                              input_shape=self.params.input_shape))
        self.model.add(tf.keras.layers.MaxPooling2D(self.params.max_pooling_shape))
        self.model.add(tf.keras.layers.Conv2D(self.params.num_filters_2,
                                              self.params.kernel_size_2,
                                              activation=self.params.activation_type))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(self.params.dense_layer_size, activation=self.params.activation_type))
        self.model.add(tf.keras.layers.Dense(self.params.dense_layer_size))

    def reset(self) -> NeuralNetwork:
        return DefaultNeuralNetwork(self.params, self.time)

    # function to train model on specified task
    def train_task(self, task: Task, epochs) -> TaskResult:
        current_log_dir = self.log_dir + "/" \
                          + str(task.dataset_name) + "/" \
                          + str(task.strategy_name) + "/"
        # + str(task.task_num) + "/" \
        # + self.time

        single_task_log_dir = current_log_dir + "/" + str(task.task_num) + "/" + self.time
        current_log_dir = current_log_dir + "/" + self.time

        if self.tensorboard_callback is None:
            self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=current_log_dir, histogram_freq=1)

        single_task_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=single_task_log_dir, histogram_freq=1)

        initial_epoch = epochs * task.task_num

        history = self.model.fit(task.training_set,
                                 task.training_labels,
                                 epochs=initial_epoch + epochs,
                                 initial_epoch=initial_epoch,
                                 validation_data=(task.validation_set, task.validation_labels),
                                 callbacks=[self.tensorboard_callback, single_task_tensorboard_callback])
        # return TaskResults
        return TaskResult(history)
