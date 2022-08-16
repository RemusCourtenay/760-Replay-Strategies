from models.neural_network import NeuralNetwork
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

    def train(self, training_data):
        # TODO...
        pass
