from strategies.selection_strategy import SelectionStrategy


class NoSelectionStrategy(SelectionStrategy):

    def train_model(self) -> 'tuple[float, float]':
        self.model.train(self.data, self.epochs)

    def select_memories(self, percentage: int) -> None:
        # We don't choose any memories so just pass empty lists in and let dataset class choose next task
        self.data.update_training_set([], [])
