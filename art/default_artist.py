import matplotlib.pyplot as plt
from art.artist import Artist


class DefaultArtist(Artist):

    def __init__(self):
        self.training_accuracy = []
        self.validation_accuracy = []
        self.policy = None

    def draw(self) -> None:
        plt.plot(self.training_accuracy, label='training_accuracy')
        plt.plot(self.validation_accuracy, label='test_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.show()

    def add_policy_name(self, policy_name):
        self.policy_name = policy_name

    def add_results(self, training_data, validation_data) -> None:
        self.training_accuracy += training_data
        self.validation_accuracy += validation_data

    def get_metrics(self):
        return self.training_accuracy, self.validation_accuracy


    def draw_comparisons(self, other_artist: 'DefaultArtist'):
        other_train_acc = other_artist.training_accuracy
        other_valid_acc = other_artist.validation_accuracy

        plt.plot(self.training_accuracy, label=('%s train_accuracy' % self.policy))
        plt.plot(self.validation_accuracy, label=('%s test_accuracy' % self.policy))
        plt.plot(other_train_acc, label='training_accuracy')
        plt.plot(other_valid_acc, label='test_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.show()

    def draw_multiple_comparisons(self, other_policies: 'list[DefaultArtist]'):
        title = 'Comparing %d selection policies' % (len(other_policies) + 1)
        plt.title(title)

        plt.plot(self.validation_accuracy, label=('%s test_accuracy' % self.policy_name))
        for policy in other_policies:
            plt.plot(policy.validation_accuracy, label=('%s test_accuracy' % policy.policy_name))


        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.show()


    # temp function to output demonstration graph
    def draw_demonstration(self, CNNModel):
        task1_accuracy = CNNModel.task1_accuracy
        task2_accuracy = CNNModel.task2_accuracy

        plt.plot(self.validation_accuracy, label='All Accuracy')
        plt.plot(task1_accuracy, label='Task1 Accuracy')
        plt.plot(task2_accuracy, label='Task2 Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.show()