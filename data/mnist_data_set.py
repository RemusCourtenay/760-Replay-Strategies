from data.data_set import DataSet
import numpy as np
import tensorflow as tf


class MnistDataSet(DataSet):

    NUM_LABELS = 10
    LABELS_PER_TASK = 5

    def __init__(self):
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
        self.NUM_TASKS = int(self.NUM_LABELS / self.LABELS_PER_TASK)
        self.train_tasks, self.test_tasks = self.build_tasks(train_images, train_labels, test_images, test_labels, self.NUM_TASKS, self.LABELS_PER_TASK, self.NUM_LABELS)
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

    def get_tasks(self):
        pass

    # function to reset current task back to first task (used when multiple NN share one dataset)
    def reset_tasks(self):
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


    # randomly split training data and test data into 2 tasks with 5 random labels in each
    def build_tasks(self, training_images, training_labels, test_images, test_labels, num_tasks: int, num_labels_per_task: int, num_labels: int) \
            -> 'list[list[list, list]], list[list[list, list]]':

        # cursed list instantiation
        train_tasks = [[[], []] for _ in range(num_tasks)]
        test_tasks = [[[], []] for _ in range(num_tasks)]

        # create array of labels
        labels = np.arange(0, num_labels)
        # shuffle labels
        np.random.shuffle(labels)
        # take the first num_labels_per_task of labels as task 1
        labels = labels[:num_labels_per_task]
        for image, label in zip(training_images, training_labels):
            # split into 2 tasks
            if label in labels:
                task_num = 0
            else:
                task_num = 1
            train_tasks[task_num][0].append(image)
            train_tasks[task_num][1].append(label)

        for image, label in zip(test_images, test_labels):

            # split into 2 tasks
            if label in labels:
                task_num = 0
            else:
                task_num = 1

            test_tasks[task_num][0].append(image)
            test_tasks[task_num][1].append(label)

        return train_tasks, test_tasks
