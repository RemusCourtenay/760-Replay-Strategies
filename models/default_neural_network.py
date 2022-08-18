from models.neural_network import NeuralNetwork
from data.data_set import DataSet
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class DefaultNeuralNetwork(NeuralNetwork):

    def __init__(self, input_shape=(28, 28, 1), max_pooling_shape=(2, 2), activation_type='relu', dense_layer_size=10):
        super().__init__()
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Conv2D(4, (5, 5), activation=activation_type, input_shape=input_shape))
        self.model.add(tf.keras.layers.MaxPooling2D(max_pooling_shape))
        self.model.add(tf.keras.layers.Conv2D(8, (3, 3), activation=activation_type))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(dense_layer_size, activation=activation_type))
        self.model.add(tf.keras.layers.Dense(dense_layer_size))

    # function to train model on specified training set and test set
    # TODO... set default test_acc and epochs
    def train(self, data_set: DataSet, test_acc, epochs):
        # define optimizer and loss function to use
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        for i in range(epochs):
            history = self.model.fit(data_set.get_training_set(), data_set.get_training_labels())
            train_loss, train_accuracy = self.model.evaluate(data_set.get_validation_set(),
                                                             data_set.get_validation_labels(), verbose=2)

            # append accuracy to lists
            self.training_accuracy.append(history.history['accuracy'])
            self.test_accuracy.append(train_accuracy)
