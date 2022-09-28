from typing import Tuple, List

from data.data_set import DataSet
import tensorflow as tf


class MnistDataSet(DataSet):
    NUM_LABELS = 10
    NAME = "numbers"

    def __init__(self, num_labels_per_task=2, labelled_validation=False):
        super().__init__(self.NAME,
                         tf.keras.datasets.mnist.load_data(),
                         self.NUM_LABELS,
                         num_labels_per_task,
                         labelled_validation)

    def reset(self):
        return MnistDataSet()
