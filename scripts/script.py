from datetime import datetime
from threading import Thread

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
        # Doesn't work
        # threads = []
        # for dataset in self.datasets:
        #     thread = Thread(target=self.run_dataset, args=[dataset])
        #     thread.start()
        #     threads.append(thread)
        #
        # for thread in threads:
        #     thread.join()

        for dataset in self.datasets:
            self.run_dataset(dataset)

        self.artist.draw()

    def run_dataset(self, dataset: DataSet):
        start_dataset = datetime.now()
        for strategy in self.strategies:
            start = datetime.now()

            # Reset model to initial state
            self.model = self.model.reset()
            dataset.reset()

            # create array to store results to store to csv
            if self.parameters.save_to_csv:
                csv_accuracy = []

            for i in range(dataset.get_num_tasks()):
                start_task = datetime.now()

                if i > 0:
                    # Generate a set of suitable memories based off of the task and the task results
                    # noinspection PyUnboundLocalVariable
                    memory_set, memory_labels = strategy.select_memories(task, task_results,
                                                                         self.parameters.num_memories)

                    # Update the dataset with the new memories
                    dataset.update_training_set(memory_set, memory_labels)

                # Get the currently active task to be learned
                task = dataset.get_task(strategy.strategy_name)

                # Train the model on the task
                task_results = self.model.train_task(task, self.parameters.epochs)
                task_results.set_strategy_name(strategy.strategy_name)

                # Send results to the artist for display/storage
                self.artist.add_results(task_results)

                # evaluate and store to csv if True
                if self.parameters.save_to_csv:
                    test_data, test_label = task.get_validation()
                    csv_loss, csv_acc = self.model.model.evaluate(test_data,  test_label, verbose=2)
                    csv_accuracy.append(csv_acc)
                    print(csv_acc)

            if self.parameters.save_to_csv:
                self.artist.save_results_to_csv(csv_accuracy, strategy.strategy_name)

                print("Finished task " + str(i) + " in " + str(datetime.now() - start_task))

            print("Finished strategy " + strategy.strategy_name + " on current dataset in " + str(
                datetime.now() - start))

        print("Finished dataset " + dataset.name + " in " + str(datetime.now() - start_dataset))