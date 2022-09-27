from abc import ABC, abstractmethod
import tensorflow as tf

from data.task import Task
from data.task_result import TaskResult
from scripts.script_parameters import ScriptParameters


class NeuralNetwork(ABC):

    DEFAULT_OPTIMIZER = 'adam'
    ACCURACY_METRIC_TAG = 'accuracy'

    def __init__(self, model, params: ScriptParameters):
        self.model = model
        self.params = params
        self.setup_layers()
        self.compile_model()

    @abstractmethod
    def setup_layers(self) -> None:
        pass

    def compile_model(self) -> None:
        # define optimizer and loss function to use
        self.model.compile(optimizer=self.DEFAULT_OPTIMIZER,
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=[self.ACCURACY_METRIC_TAG])

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def train_task(self, task: Task, epochs) -> TaskResult:
        pass
