from data.mnist_data_set import MnistDataSet
from models.simple_cnn import SimpleCNN as NeuralNetwork
from strategies.random_selection_strategy import RandomSelectionStrategy
from strategies.no_selection_strategy import NoSelectionStrategy
from strategies.forgetting_metric_strategy import ForgettingStrategy
from strategies.selection_strategy import SelectionStrategy
from art.default_artist import DefaultArtist
import tensorflow as tf


def run_selection_strategy(strategy: SelectionStrategy):
    strategy.run()
    return


def evaluate_strategy(policy) -> None:
    model = policy.model.model

    # get testing data and labels for individual tasks
    task1_data = tf.convert_to_tensor(policy.data.test_tasks[0][0])
    task1_label = tf.convert_to_tensor(policy.data.test_tasks[0][1])
    task2_data = tf.convert_to_tensor(policy.data.test_tasks[1][0])
    task2_label = tf.convert_to_tensor(policy.data.test_tasks[1][1])

    all_test_data = tf.convert_to_tensor(policy.data.validation_data)
    all_test_label = tf.convert_to_tensor(policy.data.validation_labels)

    print('===== Test set evaluation for %s =====' % policy.policy_name)
    print('Evaluation of Task 1 tests')
    model.evaluate(task1_data, task1_label, verbose=2)
    print('Evaluation of Task 2 tests')
    model.evaluate(task2_data, task2_label, verbose=2)
    print('Evaluation of all tests')
    model.evaluate(all_test_data, all_test_label, verbose=2)
    print('========================================================')


if __name__ == "__main__":
    # instantiate one dataset to train all models
    dataset = MnistDataSet()

    forgetting_selection = ForgettingStrategy(NeuralNetwork(), dataset, DefaultArtist(), 'Forgetting selection')
    print('Running Forgetting Metric Strategy. . .')
    run_selection_strategy(forgetting_selection)

    evaluate_strategy(forgetting_selection)

    # get number of forgetting data produced
    num_data_to_keep = len(forgetting_selection.forgetness)
    print('size of replay subset: ', num_data_to_keep)
    print('========================================================')

    print('Running Random Selection Strategy. . .')
    dataset.reset_tasks()
    random_selection = RandomSelectionStrategy(NeuralNetwork(), dataset, DefaultArtist(), 'Random selection')
    # specify number of data to use in random selection
    random_selection.setMemory(num_data_to_keep)
    run_selection_strategy(random_selection)
    # random_selection.artist.draw_demonstration(random_selection.model)

    print('Running No Selection Strategy. . .')
    dataset.reset_tasks()
    no_selection = NoSelectionStrategy(NeuralNetwork(), dataset, DefaultArtist(), 'No selection')
    run_selection_strategy(no_selection)
    # no_selection.artist.draw_demonstration(no_selection.model)

    # draw graph comparing multiple selection strategies
    random_selection.artist.draw_multiple_comparisons([forgetting_selection.artist, no_selection.artist])
