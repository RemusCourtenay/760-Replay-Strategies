from data.data_set import DataSet
from models.neural_network import NeuralNetwork
from art.artist import Artist
from scripts.script_parameters import ScriptParameters
from strategies.selection_strategy import SelectionStrategy
from typing import List


class Script:
    """Defines the run variables for a set of selection strategies and handles the running of them"""

    def __init__(self,
                 model: NeuralNetwork,
                 artist: Artist,
                 strategies: List[SelectionStrategy],
                 datasets: List[DataSet],
                 script_parameters: ScriptParameters):
        self.model = model
        self.artist = artist
        self.strategies = strategies
        self.datasets = datasets
        self.parameters = script_parameters

    def run(self):
        for dataset in self.datasets:
            self.run_dataset(dataset)
        self.artist.draw()

    def run_dataset(self, dataset: DataSet):
        for strategy in self.strategies:

            # Reset model to initial state
            self.model = self.model.reset()
            dataset = dataset.reset()

            for i in range(dataset.get_num_tasks()):

                if i > 0:
                    # Generate a set of suitable memories based off of the task and the task results
                    # noinspection PyUnboundLocalVariable
                    memory_set, memory_labels = strategy.select_memories(task, task_results,
                                                                         self.parameters.num_memories)

                    # Update the dataset with the new memories
                    dataset.update_training_set(memory_set, memory_labels)

                # Get the currently active task to be learned
                task = dataset.get_task()

                # Train the model on the task
                task_results = self.model.train_task(task, self.parameters.epochs)

                # Send results to the artist for display or storage
                self.artist.add_results(i, task_results)