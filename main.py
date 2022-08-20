from data.mnist_data_set import MnistDataSet as DataSet
from models.simple_cnn import SimpleCNN as NeuralNetwork
from strategies.random_selection_strategy import RandomSelectionStrategy
from strategies.no_selection_strategy import NoSelectionStrategy
from strategies.forgetting_metric_strategy import ForgettingStrategy
from strategies.selection_strategy import SelectionStrategy
from art.default_artist import DefaultArtist


def run_selection_strategy(strategy: SelectionStrategy):
    strategy.run()
    return


if __name__ == "__main__":
    # instantiate one dataset to train all models
    dataset = DataSet()

    forgettingselection = ForgettingStrategy(NeuralNetwork(), dataset, DefaultArtist())
    print('Running Forgetting Metric Strategy. . .')
    run_selection_strategy(forgettingselection)
    # get number of forgetting data produced
    num_data_to_keep = len(forgettingselection.forgetness)
    print('size of replay subset: ', num_data_to_keep)
    print('==========================')

    print('Running Random Selection Strategy. . .')
    dataset.reset_tasks()
    randomselection = RandomSelectionStrategy(NeuralNetwork(), dataset, DefaultArtist())
    # specify number of data to use in random selection
    randomselection.setMemory(num_data_to_keep)
    run_selection_strategy(randomselection)

    print('Running No Selection Strategy. . .')
    dataset.reset_tasks()
    run_selection_strategy(NoSelectionStrategy(NeuralNetwork(), dataset, DefaultArtist()))
