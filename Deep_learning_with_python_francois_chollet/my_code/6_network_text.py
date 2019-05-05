

#使用词嵌入
from keras.layers import Embedding
embedding_layer = Embedding(1000,64)
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten,Dense,Embedding
import numpy as np


def imdb_embedding():
    from keras.datasets import imdb
    max_features = 10000
    max_len=20

    (x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_features)

    x_train = preprocessing.sequence.pad_sequences(x_train,maxlen=max_len)
    x_test = preprocessing.sequence.pad_sequences(x_test,maxlen=max_len)


    model = Sequential()
    model.add(Embedding(10000,8,input_length=max_len))
    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
    model.summary()

    history = model.fit(x_train,y_train,
                        epochs=10,
                        batch_size=32,
                        validation_split=0.2)


#==============================整体框架============================================
#1.获取数据
#2.数据分词
#3.解析glove词嵌入文件
#4.定义模型
#5.模型加载词嵌入
#6.训练与评估

import os

#1.获取IMDB的数据
imdb_dir = 'data/aclImdb'
train_dir = os.path.join(imdb_dir,'train')

labels=[]
texts = []

for label_type in ['neg','pos']:
    dir_name = os.path.join(train_dir,label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] =='.txt':
            f = open(os.path.join(dir_name,fname),'r',encoding='utf-8')
            texts.append(f.read())
            f.close()
            if label_type =='neg':
                labels.append(0)
            else:
                labels.append(1)

#2.对数据进行分词,主要用tokenizer进行分词，以及对数据集进行打乱操作
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

maxlen = 100
training_samples = 200
validation_sample = 100
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('found %s unique word: ' %word_index)


data = pad_sequences(sequences,maxlen = maxlen )

labels = np.asarray(labels)
print('shape of data tensor: ',data.shape)
print('shape of label tensor',labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples:training_samples+validation_sample]
y_val  = labels[training_samples:training_samples+validation_sample]

#3.对嵌入进行预处理
glove_dir = 'data/glove.6B'
embedding_index = {}
f = open(os.path.join(glove_dir,'glove.6B.50d.txt'),'r',encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:],dtype = 'float32')
    embedding_index[word] = coefs

f.close()
print('find %s word veectors. '% len(embedding_index))


embedding_dim = 50
embedding_matrix = np.zeros((max_words,embedding_dim))
for word,i in word_index.items():
    if i <max_words:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

#4.定义模型
from keras.models import Sequential
from keras.layers import Embedding,Flatten,Dense

model = Sequential()
model.add(Embedding(max_words,embedding_dim,input_length=maxlen))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()

#5.模型中加载词嵌入
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False



#6.训练
model.compile(optimizer='rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['acc'])
history = model.fit(x_train,y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val,y_val))
model.save_weights('pre_trained_glove_model.h5')


test_dir = os.path.join(imdb_dir,'test')

labels = []
texts = []

for label_type in ['neg','pos']:
    dir_name = os.path.join(train_dir,label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] =='.txt':
            f = open(os.path.join(dir_name,fname),'r',encoding='utf-8')
            texts.append(f.read())
            f.close()
            if label_type =='neg':
                labels.append(0)
            else:
                labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences,maxlen=maxlen)
y_test = np.asarray(labels)

model.load_weights('pre_trained_glove_model.h5')
model.evaluate(x_test,y_test)

