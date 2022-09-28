from typing import List, Tuple

from data.task import Task
from data.task_result import TaskResult
from strategies.selection_strategy import SelectionStrategy


class NoSelectionStrategy(SelectionStrategy):
    STRATEGY_NAME = "no"

    def __init__(self, strategy_name=STRATEGY_NAME):
        super().__init__(strategy_name)

    def select_memories(self, task: Task, task_results: TaskResult, num_memories: int) -> Tuple[List, List]:
        # We don't choose any memories so just pass empty lists to let the update function know
        return [], []
