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
def train(model, train_data, train_label, epochs):
    # define optimizer and loss function to use
    opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    
    stat = {}
    for i in range(epochs):
        random_arrange = random.sample(range(len(train_data)), len(train_data))
        train = []
        label = [] 
        for i in range(len(train_data)):
            train.append(train_data[random_arrange[i]])
            label.append(train_label[random_arrange[i]])

        model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
        history = model.fit(tf.convert_to_tensor(train), tf.convert_to_tensor(label))

        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        predictions = probability_model.predict(train_data)
        for i in range(len(train_data)):
            if i in stat:
              stat[i].append(np.argmax(predictions[i]) == train_label[i])
            else:
              stat[i] = [np.argmax(predictions[i]) == train_label[i]]
            
    forgetness = {}
    
    for index, lst in stat.items():
      acc_full = np.array(list(map(int, lst)))
      transition = acc_full[1:] - acc_full[:-1]
      if len(np.where(transition == -1)[0]) > 0:
        forgetness[index] = len(np.where(transition == -1)[0])
      elif len(np.where(acc_full == 1)[0]) == 0:
        forgetness[index] = epochs
      else:
        forgetness[index] = 0
  
    #print(dict(sorted(forgetness.items(), key = lambda item: item[1], reverse = True)))
    result = []
    for i,j in forgetness.items():
      if j == epochs:
        result.append(i)
        
    return result
