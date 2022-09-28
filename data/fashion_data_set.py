from typing import List, Tuple

import tensorflow as tf

from data.data_set import DataSet


class FashionDataSet(DataSet):

    NUM_LABELS = 10
    NAME = "fashion"

    def __init__(self, num_labels_per_task=2, labelled_validation=False):
        super().__init__(self.NAME,
                         tf.keras.datasets.fashion_mnist.load_data(),
                         self.NUM_LABELS,
                         num_labels_per_task,
                         labelled_validation)

    def reset(self):
        return FashionDataSet()
