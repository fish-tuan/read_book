import numpy as np
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras import layers

max_feature = 10000
maxlen = 500
batch_size = 32

#利用RNN进行实验，运行该函数即可，这一个函数为rnn，其他的为另外一个实验
def test_rnn():



    (input_train,y_train),(input_test,y_test) = imdb.load_data(num_words=max_feature)
    input_train = sequence.pad_sequences(input_train,maxlen= maxlen)
    input_test = sequence.pad_sequences(input_test,maxlen = maxlen)



    model = Sequential()
    model.add(Embedding(max_feature,32))
    model.add(SimpleRNN(32))#使用LSTM则使用model.add(LSTM(32))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
    history = model.fit(input_train,y_train,
                        epochs=10,
                        batch_size=128,
                        validation_split=0.2)


    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1,len(acc)+1)

    plt.plot(epochs,acc,'bo',label = 'Trainng acc')
    plt.plot(epochs,val_acc,'g',label = 'val Acc')
    plt.title('acc')
    plt.legend()

    plt.figure()

    plt.plot(epochs,loss,'bo',label = "training loss")
    plt.plot(epochs,val_loss,'g',label = 'validation loss')
    plt.title('loss')
    plt.legend()
    plt.figure()



#=============================================温度检测，整个实验分为以下几个步骤=======================================
#1.生成数据，及数据初步处理，这个主要是对数据的敏感程度
#2.利用传统方法进行学习，然后就是升级升级

#生成一个元组(sample,targets),其中sample是输入数据的一个批量，target是目标温度数组，生成器的参数如下
#data：      浮点数据组成的原始数组，
#lookback：  输入数据应该包括过去多少个时间步
#delay：     目标应该在未来多少个时间步之后
#min_index和max_index:   data数组中的索引，用于界定需要抽取哪些时间不，这有助于保存一部分数据用于验证，另一部分用于测试
#shuffle：   打乱样本，还是按顺序抽取样本
#batch_size: 每个批量的样本数
#step:       数据采样的周期，我们将其设为6，为的是每小时抽取一个数据点

def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

lookback = 1440 #过去十天的观测数据
step = 6        #每小时一个数据电
delay = 144  #未来24小时之后的数据
batch_size = 128    #每次抽取的数据样本点

def read_generator():
    import os
    import numpy as np
    from matplotlib import pyplot as plt

    #1.把表中的数据进行读取
    data_dir = 'data/jena_climate_2009_2016.csv'
    f = open(data_dir)
    data = f.read()
    f.close()

    lines = data.split('\n')
    header = lines[0].split(',')
    lines =lines[1:]
    print(header)
    print(len(lines))

    float_data = np.zeros((len(lines),len(header)-1))
    for i,line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i,:] = values

    #需要plot数据则取消注释
    # temp = float_data[:,1]
    # plt.plot(range(len(temp)),temp)
    # plt.figure()
    # plt.plot(range(1440),temp[:1440])
    #
    mean = float_data[:200000].mean(axis = 0)
    float_data -=mean
    std = float_data[:20000].std(axis = 0)
    float_data /= std

    #2. 从上面的读取的数据中获取模型使用的数据，这里经过处理化后得到的，主要是进行初步处理
    #主要是对时间的理解，联系过去十天，来预测未来24小时的天气，另外避免数据过多进行了1个小时抽取一个样本


    train_gen = generator(float_data,
                          lookback=lookback,
                          delay=delay,
                          min_index=0,
                          max_index=200000,
                          shuffle=True,
                          step=step,
                          batch_size=batch_size)
    val_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=200001,
                        max_index=300000,
                        step=step,
                        batch_size=batch_size)
    test_gen = generator(float_data,
                         lookback=lookback,
                         delay=delay,
                         min_index=300001,
                         max_index=None,
                         step=step,
                         batch_size=batch_size)

    # This is how many steps to draw from `val_gen`
    # in order to see the whole validation set:
    val_steps = (300000 - 200001 - lookback) // batch_size

    # This is how many steps to draw from `test_gen`
    # in order to see the whole test set:
    test_steps = (len(float_data) - 300001 - lookback) // batch_size

    return train_gen,val_gen,test_gen,val_steps,test_steps,float_data.shape
    #数据标准化

#
def evaluate_naive_method(val_gen,val_steps):
    batch_maes = []
    for step in range(val_steps):
        samples,target = next(val_gen)
        preds = samples[:,-1,1]
        mae = np.mean(np.abs(preds-target))
        batch_maes.append(mae)
    print(np.mean(batch_maes))

def plot_model(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='training loss')
    plt.plot(epochs, val_loss, 'b', label='validation loss')
    plt.title('training and validation loss')
    plt.legend()
    plt.show()

#进行一个简单的密连接模型
def Dense_model(train_gen, val_gen,val_steps,float_data_shape):
    from keras.optimizers import RMSprop

    model = Sequential()
    model.add(layers.Flatten(input_shape=(lookback//step,float_data_shape[-1])))
    model.add(layers.Dense(32,activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer=RMSprop(),loss = 'mae')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=20,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)
    return history


#添加一个GRU进行
def GRU_model(train_gen, val_gen,val_steps,float_data_shape):
    from keras.optimizers import RMSprop

    model = Sequential()
    model.add(layers.GRU(32,input_shape=(None,float_data_shape[-1])))
    model.add(layers.Dense(1))

    model.compile(optimizer=RMSprop(),loss='mae')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=10,
                                  validation_data=val_gen,
                                  validation_steps=val_steps
    )
    return history

#添加一个GRU进行，其中对于循环卷积网络，如果加入dropout后是的过拟合不严重，那么可以增大每层大小，但计算成本会高
def drop_GRU_model(train_gen, val_gen,val_steps,float_data_shape):
    from keras.optimizers import RMSprop

    model = Sequential()
    model.add(layers.GRU(32,
                         dropout=0.2,#对于该层输入单元的dropout比率
                         recurrent_dropout=0.2,#指定循环单元的dropout比率
                         input_shape=(None,float_data_shape[-1])))
    model.add(layers.Dense(1))

    model.compile(optimizer=RMSprop(),loss='mae')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=10,
                                  validation_data=val_gen,
                                  validation_steps=val_steps
    )
    return history

def main():
    train_gen, val_gen, test_gen, val_steps, test_steps,float_data_shape = read_generator()
    history = GRU_model(train_gen, val_gen, val_steps, float_data_shape)
    plot_model(history)
def tuan_test():
    def tuan_generator():
        for i in range(10):
            yield i

    a = tuan_generator()
    for i in range(5):
        b = next(a)
        print(b)



if __name__ =='__main__':
    main()
    # tuan_test()