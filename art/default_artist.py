import matplotlib.pyplot as plt
from art.artist import Artist


class DefaultArtist(Artist):

    def __init__(self):
        self.training_accuracy = []
        self.validation_accuracy = []

    def draw(self) -> None:
        plt.plot(self.training_accuracy, label='training_accuracy')
        plt.plot(self.validation_accuracy, label='test_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.show()

    def add_results(self, training_data, validation_data) -> None:
        self.training_accuracy += training_data
        self.validation_accuracy += validation_data
