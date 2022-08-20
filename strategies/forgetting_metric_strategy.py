import numpy as np
import tensorflow as tf
import random
from art.artist import Artist
from data.mnist_data_set import MnistDataSet as DataSet
from models.simple_cnn import SimpleCNN as NeuralNetwork
from strategies.selection_strategy import SelectionStrategy


class ForgettingStrategy(SelectionStrategy):

    def __init__(self, model: NeuralNetwork, data: DataSet, artist: Artist, memory_percent=0, epochs=0):
        self.model = model
        self.data = data
        self.artist = artist

        self.forgetness = None

        # Probably a better way to do this
        if memory_percent == 0:
            self.memory_percent = self.DEFAULT_MEMORY_PERCENT
        else:
            self.memory_percent = memory_percent

        if epochs == 0:
            self.epochs = self.DEFAULT_EPOCHS
        else:
            self.epochs = 0

    def select_memories(self, percentage: int) -> None:
        old_training_data = self.data.get_training_set()
        old_training_labels = self.data.get_training_labels()

        new_data = old_training_data[self.forgetness]
        new_label = old_training_labels[self.forgetness]

        # Update data object's current training data
        self.data.update_training_set(new_data, new_label)

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
            training_accuracy, test_accuracy = self.train_forgetness(self.data, self.epochs)
            self.artist.add_results(training_accuracy, test_accuracy)

        # evaluate final accuracy on the 3 sets
        self.final_evaluate()

        # draw plot
        self.artist.draw()


    # function to train model on specified training set and test set
    def train_forgetness(self, data: DataSet, epochs):
        model = self.model.model
        train_data = data.get_training_set()
        train_label = data.get_training_labels()
        validation_data = data.get_validation_set()
        validation_label = data.get_validation_labels()
        # arrays for plotting later
        train_accuracy = []
        test_accuracy = []


        # define optimizer and loss function to use
        opt = tf.keras.optimizers.SGD(learning_rate=0.01)

        model.compile(optimizer=opt,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        stat = {}
        for i in range(epochs):
            print('Training data shape: ', train_data.shape)
            history = model.fit(train_data, train_label, verbose=2)

            # evaluate for plotting purposes
            test_loss, test_acc = model.evaluate(validation_data, validation_label)
            train_accuracy.append(history.history['accuracy'])
            test_accuracy.append(test_acc)


            probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
            predictions = probability_model.predict(train_data)
            for i in range(len(train_data)):
                if i in stat:
                    stat[i].append(np.argmax(predictions[i]) == train_label[i])
                else:
                    stat[i] = [np.argmax(predictions[i]) == train_label[i]]

        forgetness = {}

        for index, lst in stat.items():
            acc_full = np.array(list(map(int, lst)))
            transition = acc_full[1:] - acc_full[:-1]
            if len(np.where(transition == -1)[0]) > 0:
                forgetness[index] = len(np.where(transition == -1)[0])
            elif len(np.where(acc_full == 1)[0]) == 0:
                forgetness[index] = epochs
            else:
                forgetness[index] = 0

        # print(dict(sorted(forgetness.items(), key = lambda item: item[1], reverse = True)))
        result = []
        for i, j in forgetness.items():
            if j == epochs:
                result.append(i)

        self.forgetness = result
        return train_accuracy, test_accuracy


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