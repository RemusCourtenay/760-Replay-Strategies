from typing import Tuple, List

from models.neural_network import NeuralNetwork
from data.data_set import DataSet
import tensorflow as tf


class DefaultNeuralNetwork(NeuralNetwork):

    DEFAULT_OPTIMIZER = 'adam'
    ACCURACY_METRIC_TAG = 'accuracy'

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

    # function to train model on specified training set and test set
    # TODO... fix this method because it is broken
    def train(self, data_set: DataSet, epochs) -> Tuple[List[float], List[float]]:
        # define optimizer and loss function to use
        self.model.compile(optimizer=self.DEFAULT_OPTIMIZER,
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=[self.ACCURACY_METRIC_TAG])

        for i in range(epochs):
            history = self.model.fit(data_set.get_training_set(), data_set.get_training_labels())
            train_loss, train_accuracy = self.model.evaluate(data_set.get_validation_set(),
                                                             data_set.get_validation_labels(), verbose=2)

            # return accuracy for display purposes
            return history.history[self.ACCURACY_METRIC_TAG], train_accuracy

