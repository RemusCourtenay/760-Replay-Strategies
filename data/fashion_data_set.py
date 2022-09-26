from typing import List, Tuple

import tensorflow as tf

from data.data_set import DataSet


class FashionDataSet(DataSet):

    NUM_LABELS = 10

    def __init__(self, num_labels_per_task=2, labelled_validation=True):
        super().__init__(tf.keras.datasets.fashion_mnist.load_data(),
                         self.NUM_LABELS,
                         num_labels_per_task,
                         labelled_validation)

    def reset(self):
        return FashionDataSet()
