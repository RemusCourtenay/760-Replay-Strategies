#!/usr/bin/env python
# coding: utf-8

# In[9]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import time
print('--------------')
nowtime = time.strftime('%Y-%m-%d %H:%M:%S')
print(nowtime)




plt.rcParams['font.sans-serif'] = ['SimHei']


mnist = tf.keras.datasets.mnist
(train_x,train_y),(test_x,test_y) = mnist.load_data()
print('\n train_x:%s, train_y:%s, test_x:%s, test_y:%s'%(train_x.shape,train_y.shape,test_x.shape,test_y.shape)) 

     
X_train,X_test = tf.cast(train_x/255.0,tf.float32),tf.cast(test_x/255.0,tf.float32)    
y_train,y_test = tf.cast(train_y,tf.int16),tf.cast(test_y,tf.int16)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))  
model.add(tf.keras.layers.Dense(128,activation='relu'))     
model.add(tf.keras.layers.Dense(10,activation='softmax'))
print('\n',model.summary())   


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])   


print('--------------')
nowtime = time.strftime('%Y-%m-%d %H:%M:%S')
print('before：'+str(nowtime))

history = model.fit(X_train,y_train,batch_size=64,epochs=5,validation_split=0.2)

print('--------------')
nowtime = time.strftime('%Y-%m-%d %H:%M:%S')
print('after：'+str(nowtime))

model.evaluate(X_test,y_test,verbose=2)     


model.save('mnist_weights.h5')



print(history.history)
loss = history.history['loss']          
val_loss = history.history['val_loss']  
acc = history.history['sparse_categorical_accuracy']            
val_acc = history.history['val_sparse_categorical_accuracy']    

plt.figure(figsize=(10,3))

plt.subplot(121)
plt.plot(loss,color='b',label='train')
plt.plot(val_loss,color='r',label='test')
plt.ylabel('loss')
plt.legend()

plt.subplot(122)
plt.plot(acc,color='b',label='train')
plt.plot(val_acc,color='r',label='test')
plt.ylabel('Accuracy')
plt.legend()


plt.figure()
for i in range(10):
    num = np.random.randint(1,10000)

    plt.subplot(2,5,i+1)
    plt.axis('off')
    plt.imshow(test_x[num],cmap='gray')
    demo = tf.reshape(X_test[num],(1,28,28))
    y_pred = np.argmax(model.predict(demo))
    plt.title('label：'+str(test_y[num])+'\npre：'+str(y_pred))

