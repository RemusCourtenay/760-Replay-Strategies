from data.data_set import DataSet
import numpy as np
import tensorflow as tf
import math

class MnistDataSet(DataSet):

    NUM_LABELS = 10

    def __init__(self, labels_per_task, debug=False):
        self.debug = debug
        self.labels_per_task = labels_per_task
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
        self.num_tasks = math.ceil(self.NUM_LABELS / self.labels_per_task)
        self.train_tasks = self.__build_tasks(train_images, train_labels, self.num_tasks, self.labels_per_task, self.NUM_LABELS)
        self.validation_data = test_images
        self.validation_labels = test_labels
        self.current_task = 0
        self.current_training_data = np.array(self.train_tasks[0][0])
        self.current_training_labels = np.array(self.train_tasks[0][1])

    # updates current task by concatenating next task and selected data
    def update_training_set(self, selected_memory_data, selected_memory_labels):
        task_index = self.current_task + 1
        new_training_data = np.array(self.train_tasks[task_index][0])
        new_training_labels = np.array(self.train_tasks[task_index][1])

        # concatenate selected data if it is not empty
        if len(selected_memory_data) > 0:
            new_training_data = np.concatenate((new_training_data, selected_memory_data), axis=0)
            new_training_labels = np.concatenate((new_training_labels, selected_memory_labels), axis=0)

        self.current_training_data = new_training_data
        self.current_training_labels = new_training_labels

    def get_task(self):
        return self.current_training_data


    # function to reset current task back to first task (used when multiple NN share one dataset)
    def reset(self):
        self.current_task = 0
        self.current_training_data = np.array(self.train_tasks[0][0])
        self.current_training_labels = np.array(self.train_tasks[0][1])

    def get_training_set(self):
        return self.current_training_data.copy()

    def get_training_labels(self):
        return self.current_training_labels.copy()

    def get_validation_set(self):
        return self.validation_data.copy()

    def get_validation_labels(self):
        return self.validation_labels.copy()


    def __build_tasks(self, training_images, training_labels, num_tasks: int, num_labels_per_task: int, num_labels: int) \
            -> 'list[list[list, list]], list[list[list, list]]':

        # cursed list instantiation
        train_tasks = [[[], []] for _ in range(num_tasks)]

        # create array of labels to generate random ordering in tasks
        labels = np.arange(0, num_labels)
        # shuffle labels
        np.random.shuffle(labels)
        # store the labels in an array then split the shuffled labels into specified number of tasks
        task_labels = []
        for i in range(num_tasks):
            task_labels.append(labels[i * num_labels_per_task:((i * num_labels_per_task) + num_labels_per_task)])
        if self.debug:
            print(task_labels)

        for image, label in zip(training_images, training_labels):
            # find out which task the data should belong to by checking with all the task_labels
            for i in range(num_tasks):
                if label in task_labels[i]:
                    task_num = i
                    break
            train_tasks[task_num][0].append(image)
            train_tasks[task_num][1].append(label)

        return train_tasks


### debug ###
data = MnistDataSet(labels_per_task=3, debug=True)
for i in range(data.num_tasks):
    print('First 10 labels for Task %d: ' % i, data.train_tasks[i][1][:10], 'number of images:', len(data.train_tasks[i][1]))