from models.neural_network import NeuralNetwork
from data.data_set import DataSet
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class DefaultNeuralNetwork(NeuralNetwork):
    
    def __init__(self, input_shape = (28, 28, 1), max_pooling_shape = (2, 2), activation_type = 'relu', dense_layer_size = 10):
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
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        for i in range(epochs):
            history = model.fit(data_set.train_data, data_set.train_label)
            train_loss, train_accuracy = model.evaluate(data_set.test_data,  data_set.test_label, verbose=2)

            # append accuracy to lists
            data_set.train_acc += history.history['accuracy']
            data_set.test_acc.append(train_accuracy)
