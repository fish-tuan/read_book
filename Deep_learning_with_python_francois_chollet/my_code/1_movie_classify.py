#coding:utf-8
from keras.datasets import imdb

#================================数据准备====================================

#数据下载
#runfile('C:/Users/tuan/Documents/code/deep/Deep_learning_women/my_code/1_movie_classify.py', wdir='C:/Users/tuan/Documents/code/deep/Deep_learning_women/my_code')
(train_data,train_labels),(test_data,test_labels)  = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])#减3是因为0，1，2代表padding，start of sequence,unknown

#数据向量化
import numpy as np

#one-hot的感觉，将整数编码转化成了二进制矩阵
def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequences in enumerate(sequences):
        results[i,sequences]=1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#===========================构建网络=====================================
from keras import models
from keras import layers
from keras import regularizers

#模型定义及编译模型
model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,),kernel_regularizer = regularizers.l2(0.01)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))#如果是多分类，就不是1了
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

#假如希望自定义优化器的参数，或者自定义损失函数和指标函数
'''
from keras import optimizers
model.compile(optimizers=optimizers.RMSprop(lr=0.001),
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

from keras import losses
from keras import metrics

model.compile(optimizers=optimizers.RMSprop(lr=0.001),
              loss = losses.binary_crossentropy,
              metrics = [metrics.binary_accuracy])
'''

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val,y_val))



