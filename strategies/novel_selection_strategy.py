from strategies.selection_strategy import SelectionStrategy


class NovelSelectionStrategy(SelectionStrategy):

    def select_memories(self, percentage: int) -> None:
        # TODO... Implement our own selection policy

        # Might need these?
        previous_training_data = self.data.get_training_set()
        previous_training_labels = self.data.get_training_labels()

        # Training accuracy can be captured by overriding the run method from SelectionStrategy.
        # If you need more detailed information from the CNN then you'll need to make a new class that extends
        # NeuralNetwork.

        selected_memory_data = []
        selected_memory_labels = []
        self.data.update_training_set(selected_memory_data, selected_memory_labels)
