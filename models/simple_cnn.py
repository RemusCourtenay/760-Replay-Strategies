from models.neural_network import NeuralNetwork
from data.mnist_data_set import MnistDataSet as DataSet
import tensorflow as tf


class SimpleCNN(NeuralNetwork):

    DEFAULT_OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=0.01)
    ACCURACY_METRIC_TAG = 'accuracy'

    def __init__(self):
        super().__init__()
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Conv2D(4, (5, 5), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(10, activation='relu'))
        self.model.add(tf.keras.layers.Dense(10))

    # function to train model on specified training set and test set
    def train(self, data_set: DataSet, epochs) -> 'tuple[list[float], list[float]]':
        # define optimizer and loss function to use
        self.model.compile(optimizer=self.DEFAULT_OPTIMIZER,
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=[self.ACCURACY_METRIC_TAG])

        train_accruacy = []
        test_accruacy = []
        for i in range(epochs):
            train_data = data_set.get_training_set()
            print('Training data shape: ', train_data.shape)
            history = self.model.fit(train_data, data_set.get_training_labels())
            test_loss, test_acc = self.model.evaluate(data_set.get_validation_set(),
                                                             data_set.get_validation_labels())
            train_accruacy.append(history.history[self.ACCURACY_METRIC_TAG])
            test_accruacy.append(test_acc)
            # return accuracy for display purposes
        return train_accruacy, test_accruacy

