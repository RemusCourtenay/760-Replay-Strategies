from art.artist import Artist
from data.mnist_data_set import MnistDataSet as DataSet
from models.simple_cnn import SimpleCNN as NeuralNetwork
from strategies.selection_strategy import SelectionStrategy
import tensorflow as tf


class NoSelectionStrategy(SelectionStrategy):
    def __init__(self, model: NeuralNetwork, data: DataSet, artist: Artist, policy_name):
        self.model = model
        self.data = data
        self.artist = artist

        self.policy_name = policy_name
        self.artist.add_policy_name(policy_name)

        self.epochs = 10
        self.memory_percent = 0

    def select_memories(self, percentage: int) -> None:
        # We don't choose any memories so just pass None to let the update function know
        self.data.update_training_set([], [])

    def run(self) -> None:
        for i in range(self.data.NUM_TASKS):
            print('==== task %d =====' % (i + 1))
            # only update replay memory if not the first task
            if i > 0:
                self.select_memories(self.memory_percent)
            training_accuracy, test_accuracy = self.train_model()
            self.artist.add_results(training_accuracy, test_accuracy)

        # evaluate final accuracy on the 3 sets
        self.final_evaluate()
        # self.artist.draw()

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