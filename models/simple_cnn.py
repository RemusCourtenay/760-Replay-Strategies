from typing import Tuple, List

from models.neural_network import NeuralNetwork
from data.mnist_data_set import MnistDataSet as DataSet
import tensorflow as tf


class SimpleCNN(NeuralNetwork):
    DEFAULT_OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=0.001)
    ACCURACY_METRIC_TAG = 'accuracy'

    def __init__(self):
        super().__init__(tf.keras.models.Sequential())
        self.model.add(tf.keras.layers.Conv2D(4, (5, 5), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(10, activation='relu'))
        self.model.add(tf.keras.layers.Dense(10))

        #####################################################################
        # temporary to output example graph
        self.task1_accuracy = []
        self.task2_accuracy = []
        #####################################################################

    # function to train model on specified training set and test set
    def train(self, data_set: DataSet, epochs) -> Tuple[List[float], List[float]]:
        # define optimizer and loss function to use
        self.model.compile(optimizer=self.DEFAULT_OPTIMIZER,
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=[self.ACCURACY_METRIC_TAG])

        train_accuracy = []
        test_accuracy = []

        ############ temp code to generate demsontration graph##############
        task1_data = tf.convert_to_tensor(data_set.test_tasks[0][0])
        task1_label = tf.convert_to_tensor(data_set.test_tasks[0][1])
        task2_data = tf.convert_to_tensor(data_set.test_tasks[1][0])
        task2_label = tf.convert_to_tensor(data_set.test_tasks[1][1])
        #####################################################################

        for i in range(epochs):
            train_data = data_set.get_training_set()
            # print('Training data shape: ', train_data.shape)
            history = self.model.fit(train_data, data_set.get_training_labels())
            test_loss, test_acc = self.model.evaluate(data_set.get_validation_set(),
                                                      data_set.get_validation_labels(), verbose=0)

            ############ temp code to generate demsontration graph##############
            task1_loss, task1_acc = self.model.evaluate(task1_data,
                                                        task1_label, verbose=0)
            task2_loss, task2_acc = self.model.evaluate(task2_data,
                                                        task2_label, verbose=0)
            #####################################################################

            train_accuracy.append(history.history[self.ACCURACY_METRIC_TAG])
            test_accuracy.append(test_acc)

            self.task1_accuracy.append(task1_acc)
            self.task2_accuracy.append(task2_acc)

            # return accuracy for display purposes
        return train_accuracy, test_accuracy
