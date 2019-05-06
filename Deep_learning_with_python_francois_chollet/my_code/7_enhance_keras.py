from keras.models import Sequential,Model
from keras import layers
from keras import Input
import numpy as np
import keras

#=============================================高级用法==============================================================
#第一部分：进一步的keras结构，1-5
#第二部分：利用keras回调函数和tensorboard检查并监控深度学习模型 ，6-7
#

#1.简单api
def api_sample():
    seq_model = Sequential()
    seq_model.add(layers.Dense(32,activation='relu',input_shape=(64,)))
    seq_model.add(layers.Dense(32,activation='relu'))
    seq_model.add(layers.Dense(10,activation='softmax'))

    seq_model.summary()
    input_tensor = Input(shape=(64,))
    x = layers.Dense(32,activation='relu')(input_tensor)
    x = layers.Dense(32,activation='relu')(x)
    ouput  = layers.Dense(10,activation='softmax')(x)

    model = Model(input_tensor,ouput)

    model.summary()

#2.多输入
def muti_input():
    #多输入模型
    text_vocabulary_size=10000
    question_vocabulary_size=10000
    anser_vocabulary_size=500


    text_input = Input(shape=(None,),dtype='int32',name='text')#选择命名，并且输入长度可变的文本序列
    embedding_text = layers.Embedding(text_vocabulary_size,64)(text_input)
    encoded_text = layers.LSTM(32)(embedding_text)

    question_input = Input(shape=(None,),
                           dtype='int32',
                           name='question')

    embedded_question = layers.Embedding(question_vocabulary_size,32)(question_input)
    encoded_quetion = layers.LSTM(16)(embedded_question)

    concatenated = layers.concatenate([encoded_text,encoded_quetion],axis = -1)

    anser = layers.Dense(anser_vocabulary_size,activation='softmax')(concatenated)

    model = Model([text_input,question_input],anser)

    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])


    num_samples=1000
    max_length = 100

    text = np.random.randint(1,text_vocabulary_size,size=(num_samples,max_length))
    quetion = np.random.randint(1,question_vocabulary_size,size=(num_samples,max_length))
    answers = np.random.randint(anser_vocabulary_size,size=(num_samples))
    answers = keras.utils.to_categorical(answers,anser_vocabulary_size)
    model.fit([text,quetion],answers,epochs=10,batch_size=128)

    # model.fit({'text':text,'question':quetion},answers,epochs=10,batch_size=128)

#3.多输出
def muti_output():
    vocabulary_size = 50000
    num_income_groups=10

    posts_input = Input(shape=(None,),dtype='int32',name='posts')
    embedded_posts = layers.Embedding(256,vocabulary_size)(posts_input)
    x = layers.Conv1D(128,5,activation='relu')(embedded_posts)
    x = layers.MaxPool1D(5)(x)
    x = layers.Conv1D(256,5,activation='relu')(x)
    x = layers.Conv1D(256,5,activation='relu')(x)
    x = layers.MaxPool1D(5)(x)
    x = layers.Conv1D(256,5,activation='relu')(x)
    x = layers.Conv1D(256,5,activation='relu')(x)
    x = layers.MaxPool1D(5)(x)
    x = layers.Dense(128,activation='relu')(x)

    age_prediction = layers.Dense(1,name='age')(x)
    income_prediction = layers.Dense(num_income_groups,activation='softmax',name='income')(x)
    gender_prediction = layers.Dense(1,activation='sigmoid')

    model = Model(posts_input,[age_prediction,income_prediction,gender_prediction])

    model.compile(optimizer='rmsprop',loss=['mse','categorical_crossentropy','binary_crossentropy'],loss_weights=[0.25,1.,10.])#损失加权
    # model.compile(optimizer='rmsprop',loss={'age':'mse','income':'categorical_crossentropy','gender':'binary_crossentropy'})

#4.层组成有向无环图，常见的组建有Inception模块，和残差连接

def Inception(x):
    branch_a = layers.Conv2D(128,1,activation='relu',strides=2)(x)
    branch_b = layers.Conv2D(128,1,activation='relu')(x)
    branch_b = layers.Conv2D(128,3,activation='relu',strides=2)(branch_b)
    branch_c = layers.AveragePooling2D(3,strides=2)(x)
    branch_c = layers.Conv2D(128,3,activation='relu')(branch_c)

    branch_d = layers.Conv2D(128,1,activation='relu')(x)
    branch_d = layers.Conv2D(128,3,activation='relu')(branch_d)
    branch_d = layers.Conv2D(128,3,activation='relu',strides=2)(branch_d)

    output = layers.concatenate([branch_a,branch_b,branch_c,branch_d],axis=-1)

def residual_connection():
    x = input(None,)
    y = layers.Conv2D(128,3,activation='relu',padding='same')(x)
    y = layers.Conv2D(128,3,activation='relu',padding='same')(y)
    y = layers.MaxPool2D(2,strides=2)(y)

    residual = layers.Conv2D(128,1,strides=2,padding='same')(x)

    y = layers.add([y,residual])

#5.共享层权重
def share_lstm(left_data,right_data,target):#参数是我乱设的
    lstm = layers.LSTM(32)

    left_input = Input(shape=(None,128))
    left_output = lstm(left_input)

    right_input = Input(shape=(None,128))
    right_output = lstm(right_input)

    merged = layers.concatenate([left_output,right_output],axis=-1)
    prediction = layers.Dense(1,activation='sigmoid')(merged)

    model = Model([left_output,right_output],prediction)
    model.fit([left_data,right_data],target)

#6 回调函数 1.模型检查点     2.提前终止      3.训练动态调整参数      4.训练记录训练指标和验证指标
def callback_use(x,y,x_val,y_val):
    model = Sequential()
    calllbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor = 'acc',
            patience=1,),#如果不再改善则终止训练，patience是指如果精度在多于一轮时间（即两轮）不再改善，则中断训练
        keras.callbacks.ModelCheckpoint(
            filepath='my_model.h5',
            monitor='val_loss',
            save_best_only=True,),               #每轮保存当前权重
        keras.callbacks.ReduceLROnPlateau(      #如果损失不再改善，那么使用该函数降低学习率，
            monitor='val_loss',
            factor = 0.1,                          #触发时将学习率除以10
            patience=10,                            #10轮内都没有改善
        )
    ]

    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])#由于上面监控精度，因此需要时模型指标一部分
    model.fit(x,y,
              epochs=10,
              calllbacks_list=calllbacks_list,
              validation_data=(x_val,y_val))

#7.tensorboard可视化框架
from keras.datasets import imdb
from keras.preprocessing import sequence

max_feaures = 100
max_len = 100

(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_feaures)
x_train =sequence.pad_sequences(x_train,maxlen = max_len)
x_test = sequence.pad_sequences(x_test,maxlen=max_len)

x_train = x_train.astype(float)
x_test = x_test.astype(float)
embeddings_data = x_train

model = keras.models.Sequential()
model.add(layers.Embedding(max_feaures,128,input_length=max_len,name='embed'))
model.add(layers.Conv1D(32,7,activation='relu'))
model.add(layers.MaxPool1D(5))
model.add(layers.Conv1D(32,7,activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

callbacks = [
    keras.callbacks.TensorBoard(
        log_dir = 'my_log_dir',
        histogram_freq=1,
        embeddings_freq=1,
        # embeddings_data=x_train[0:10]
    )
]

history = model.fit(x_train,y_train,
                    epochs=5,
                    batch_size=128,
                    validation_split=0.2,
                    callbacks = callbacks)

