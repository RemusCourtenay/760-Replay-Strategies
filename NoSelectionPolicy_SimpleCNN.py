import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

# define model structure
def SimpleCNN():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(4, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(10))
    return model

# function to train model on specified training set and test set
def train(model, train_data, train_label, test_data, test_label, train_acc, test_acc, epochs):
    # define optimizer and loss function to use
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    for i in range(epochs):
        history = model.fit(train_data, train_label)
        train_loss, train_accuracy = model.evaluate(test_data,  test_label, verbose=2)

        # append accuracy to lists
        train_acc += history.history['accuracy']
        test_acc.append(train_accuracy)



# splits training data into 2 tasks labels (0, 1, 2, 3) and (4, 5, 6, 7, 8)
def splitTasks(train_data, train_label):
    task1_data = []
    task1_label = []
    task2_data = []
    task2_label = []
    for i in range(len(train_data)):
        if train_label[i] < 4:
            task1_data.append(train_data[i])
            task1_label.append(train_label[i])
        else:
            task2_data.append(train_data[i])
            task2_label.append(train_label[i])
    return np.array(task1_data), np.array(task1_label), np.array(task2_data), np.array(task2_label)


# download mnist data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
print(train_images.shape)
train1_data, train1_label, train2_data, train2_label = splitTasks(train_images, train_labels)
test1_data, test1_label, test2_data, test2_label = splitTasks(test_images, test_labels)

# instantiate model
model = SimpleCNN()
model.summary()

# training

# use list to store metrics
train_acc = []
test_acc = []


print()
print('==========================')
print('Training Task 1')
train(model, train1_data, train1_label, test_images, test_labels, train_acc, test_acc, 5)
print('Training Task 2')
train(model, train2_data, train2_label, test_images, test_labels, train_acc, test_acc, 5)
plt.plot(train_acc, label='training_accuracy')
plt.plot(test_acc, label='test_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
print('Evaluation of Task 1 tests')
model.evaluate(test1_data, test1_label, verbose=2)
print('==========================')
print('Evaluation of Task 2 tests')
model.evaluate(test2_data, test2_label, verbose=2)
print('==========================')
print('Evaluation of all tests')
model.evaluate(test_images, test_labels, verbose=2)
print('==========================')
plt.show()