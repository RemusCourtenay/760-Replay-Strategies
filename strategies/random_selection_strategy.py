import numpy as np
import tensorflow as tf

from art.artist import Artist
from data.mnist_data_set import MnistDataSet as DataSet
from models.simple_cnn import SimpleCNN as NeuralNetwork
from strategies.selection_strategy import SelectionStrategy


class RandomSelectionStrategy(SelectionStrategy):

    def __init__(self, model: NeuralNetwork, data: DataSet, artist: Artist, memory_percent=0, epochs=0):
        self.model = model
        self.data = data
        self.artist = artist

        # Probably a better way to do this
        if memory_percent == 0:
            self.memory_percent = self.DEFAULT_MEMORY_PERCENT
        else:
            self.memory_percent = memory_percent

        if epochs == 0:
            self.epochs = self.DEFAULT_EPOCHS
        else:
            self.epochs = 0

    def setMemory(self, percent):
        self.memory_percent = percent

    def select_memories(self, percentage: int) -> None:
        old_training_data = self.data.get_training_set()
        old_training_labels = self.data.get_training_labels()

        n = len(old_training_data)
        # create an array of indexes to shuffle
        s = np.arange(0, n)
        np.random.shuffle(s)
        # shuffle the data and label arrays
        old_data_subset = old_training_data[s]
        old_label_subset = old_training_labels[s]

        # cut subset down to desired size (use the parameter directly if it is an integer > 1)
        if percentage > 1:
            old_data_subset = old_data_subset[:int(percentage)]
            old_label_subset = old_label_subset[:int(percentage)]
        else:
            old_data_subset = old_data_subset[:int(n * percentage)]
            old_label_subset = old_label_subset[:int(n * percentage)]

        # Update data object's current training data
        self.data.update_training_set(old_data_subset, old_label_subset)

    # TODO... figure out what these are supposed to be
    # % of previous task to keep (randomly)
    DEFAULT_MEMORY_PERCENT = .05
    # EPOCH per task
    DEFAULT_EPOCHS = 5

    def run(self) -> None:
        for i in range(self.data.NUM_TASKS):
            print('==== task %d =====' %(i + 1))
            # only update replay memory if not the first task
            if i > 0:
                self.select_memories(self.memory_percent)
            training_accuracy, test_accuracy = self.train_model()
            self.artist.add_results(training_accuracy, test_accuracy)

        # evaluate final accuracy on the 3 sets
        self.final_evaluate()

        # draw plot
        self.artist.draw()

    def train_model(self) -> 'tuple[list[float], list[float]]':
        training_accuracy, test_accuracy = self.model.train(self.data, self.epochs)
        return training_accuracy, test_accuracy


    def final_evaluate(self) -> None:
        model = self.model.model

        # get testing data and labels for individual tasks
        task1_data = tf.convert_to_tensor(self.data.test_tasks[0][0])
        task1_label = tf.convert_to_tensor(self.data.test_tasks[0][1])
        task2_data = tf.convert_to_tensor(self.data.test_tasks[1][0])
        task2_label = tf.convert_to_tensor(self.data.test_tasks[1][1])

        all_test_data = tf.convert_to_tensor(self.data.validation_data)
        all_test_label = tf.convert_to_tensor(self.data.validation_labels)


        print('===== Final Accuracy =====')
        print('Evaluation of Task 1 tests')
        model.evaluate(task1_data, task1_label, verbose=2)
        print('==========================')
        print('Evaluation of Task 2 tests')
        model.evaluate(task2_data, task2_label, verbose=2)
        print('==========================')
        print('Evaluation of all tests')
        model.evaluate(all_test_data, all_test_label, verbose=2)
        print('==========================')