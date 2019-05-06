from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

max_feature = 10000
max_len=500

def model_plot(history):
    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1,len(loss)+1)

    # plt.plot(epochs,acc,'bo',label = 'Trainng acc')
    # plt.plot(epochs,val_acc,'g',label = 'val Acc')
    # plt.title('acc')
    # plt.legend()
    #
    # plt.figure()

    plt.plot(epochs,loss,'bo',label = "training loss")
    plt.plot(epochs,val_loss,'g',label = 'validation loss')
    plt.title('loss')
    plt.legend()
    plt.figure()

def test_conv_sequence():
    (x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_feature)

    x_train = sequence.pad_sequences(x_train,maxlen=max_len)
    x_test = sequence.pad_sequences(x_test,maxlen=max_len)

    model = Sequential()
    model.add(layers.Embedding(max_feature,128,input_length=max_len))
    model.add(layers.Conv1D(32,7,activation='relu'))
    model.add(layers.MaxPool1D(5))
    model.add(layers.Conv1D(32,7,activation='relu'))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(1))


    model.summary()

    model.compile(optimizer=RMSprop(lr=1e-4),
                  loss = 'binary_crossentropy',
                  metrics=['acc'])
    history = model.fit(x_train,y_train,
                        epochs=5,
                        batch_size=128,
                        validation_split=0.2)

    model_plot(history)



#========================================温度预测例子===========================================================
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
# step = 6        #每小时一个数据电
step=3
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

#没有时间特性的卷积，虽然有快速与轻量，但是并不是很好用，
#想要结合卷积神经网络的速度和轻量，与RNN的顺序敏感，一种是在RNN前面使用一维卷积网络作为预处理步骤，
#对于那种非常长，以至于RNN无法处理序列，这种方法尤其有用，卷积可以转化为高级特征组成的更短序列
def Conv_model(train_gen, val_gen, val_steps, float_data_shape):
    model = Sequential()
    model.add(layers.Conv1D(32,5,activation='relu',input_shape=(None,float_data_shape[-1])))
    model.add(layers.MaxPool1D(3))
    model.add(layers.Conv1D(32,5,activation='relu'))
    model.add(layers.MaxPool1D(3))
    model.add(layers.Conv1D(32,5,activation='relu'))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(1))

    model.compile(optimizer=RMSprop(),loss='mae')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=5,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)
    return history

def Conv_RNN_model(train_gen, val_gen, val_steps, float_data_shape):
    model = Sequential()
    model.add(layers.Conv1D(32,5,activation='relu',input_shape=(None,float_data_shape[-1])))
    model.add(layers.MaxPool1D(3))
    model.add(layers.Conv1D(32,5,activation='relu'))
    model.add(layers.GRU(32,dropout=0.1,recurrent_dropout=0.5))
    model.add(layers.Dense(1))
    model.summary()

    model.compile(optimizer=RMSprop(),loss='mae')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=2,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)
    return history




def main():
    train_gen, val_gen, test_gen, val_steps, test_steps, float_data_shape = read_generator()
    history = Conv_RNN_model(train_gen, val_gen, val_steps, float_data_shape)
    model_plot(history)

if __name__=='__main__':
    main()