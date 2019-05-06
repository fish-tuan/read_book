from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import layers
from keras.models import Sequential
import  matplotlib.pyplot as plt
max_feature = 1000
maxlen=500


def model_plot(history):
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


(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_feature)

x_train = [x[::-1] for x in x_train]
x_test = [x[::-1] for x in x_test]

x_train = sequence.pad_sequences(x_train,maxlen= maxlen)
x_test = sequence.pad_sequences(x_test,maxlen = maxlen)

#与正向的效果差不多，说明反向也是有信息的，这就是双向的来源
def inver_LSTM(x_train,y_train):
    model = Sequential()
    model.add(layers.Embedding(max_feature,128))
    model.add(layers.LSTM(32))
    model.add(layers.Dense(1,activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss = 'binary_crossentropy',
                  metrics=['acc'])

    history = model.fit(x_train,y_train,
                        epochs=2,
                        batch_size=128,
                        validation_split=0.2)
    return history

#准确率有提升，但容易过拟合，因为双向的参数是单向的两倍，因此需要正则
def bi_LSTM(x_train,y_train):
    model = Sequential()
    model.add(layers.Embedding(max_feature, 128))
    model.add(layers.Bidirectional(layers.LSTM(32)))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])

    history = model.fit(x_train, y_train,
                        epochs=2,
                        batch_size=128,
                        validation_split=0.2)
    return history





def main():
    history = bi_LSTM(x_train,y_train)
    model_plot(history)

if __name__=='__main__':
    main()

