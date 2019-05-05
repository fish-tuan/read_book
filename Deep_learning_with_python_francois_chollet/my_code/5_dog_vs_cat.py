#coding:utf-8
import os,shutil
from keras.preprocessing.image import ImageDataGenerator
import pickle
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras import optimizers
import numpy as np


#结构
#1.路径数据准备
#2.基础模型
#3.图像增强模型（其实只有数据增强了，即移动和缩放）
#4.使用已经做好的模型，这里用道德是VGG16（使用预训练网络有两种方法：1.特征提取，2.微调模型）
    #4.1：对于提取特征，我们一般使用的是训练好的一系列池化层和卷积层（卷积基），原因是卷积基学习的表示更加通用，因此更适合重复使用
        #表示通用性取决于该层在模型的深度，一般靠近底部的层提取是局部的，高度通用的特征图（比如视觉边缘、颜色、纹理），而对于顶部
        #的层一般是提取更多的抽象的盖帘，因此如果新数据与原始训练的数据集有很大差异，那么最好只使用前基层来做特征 提取，而不是整个卷积基
    #4.2微调模型，指对顶层进行解冻，将解冻和新增加的部分进行联合训练，称为微调





#1.设置路径和数据
class path_and_data(object):
    #path setting
    def __init__(self):
        self.original_dataset_dir = 'data/dog_vs_cat/train'
        self.small_dataset = 'data/dog_vs_cat/dog_vs_cat_small/'
        if not os.path.exists(self.small_dataset):os.mkdir(self.small_dataset)

        self.train_dir = os.path.join(self.small_dataset,'train')
        if not os.path.exists(self.train_dir):os.mkdir(self.train_dir)

        self.test_dir = os.path.join(self.small_dataset,'test')
        if not os.path.exists(self.test_dir):os.mkdir(self.test_dir)

        self.validation_dir = os.path.join(self.small_dataset,'validation')
        if not os.path.exists(self.validation_dir):os.mkdir(self.validation_dir)

        self.train_cats_dir = os.path.join(self.train_dir,'cats')
        if not os.path.exists(self.train_cats_dir):os.mkdir(self.train_cats_dir)
        self.train_dogs_dir = os.path.join(self.train_dir,'dogs')
        if not os.path.exists(self.train_dogs_dir):os.mkdir(self.train_dogs_dir)

        self.test_cats_dir = os.path.join(self.test_dir,'cats')
        if not os.path.exists(self.test_cats_dir):os.mkdir(self.test_cats_dir)
        self.test_dogs_dir = os.path.join(self.test_dir,'dogs')
        if not os.path.exists(self.test_dogs_dir):os.mkdir(self.test_dogs_dir)

        self.validation_cats_dir = os.path.join(self.validation_dir,'cats')
        if not os.path.exists(self.validation_cats_dir):os.mkdir(self.validation_cats_dir)
        self.validation_dogs_dir = os.path.join(self.validation_dir,'dogs')
        if not os.path.exists(self.validation_dogs_dir):os.mkdir(self.validation_dogs_dir)

    def copy_file(self,finall_name,dst_name):
        src_name = self.original_dataset_dir
        for fname in finall_name:
            src = os.path.join(src_name,fname)
            dst = os.path.join(dst_name,fname)
            shutil.copyfile(src,dst)

    def copy_excate_file(self):
        if(len(os.listdir(self.validation_dogs_dir))==0):
            fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
            self.copy_file(fnames,self.train_cats_dir)

            fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
            self.copy_file(fnames,self.validation_cats_dir)

            fnames = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
            self.copy_file(fnames,self.test_cats_dir)


            fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
            self.copy_file(fnames,self.train_dogs_dir)

            fnames = ['dog.{}.jpg'.format(i) for i in range(1000,1500)]
            self.copy_file(fnames,self.validation_dogs_dir)

            fnames = ['dog.{}.jpg'.format(i) for i in range(1500,2000)]
            self.copy_file(fnames,self.test_dogs_dir)
        print(len(os.listdir(self.train_dir)))

#2.基础版设置
def network_model(file_data):



    path = 'cat_and_dog_small_1.h5'

    if not os.path.exists(path):
        history_path = open('model_history', 'wb')
        model = models.Sequential()
        model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
        model.add(layers.MaxPool2D((2,2)))
        model.add(layers.Conv2D(64,(3,3),activation='relu'))
        model.add(layers.MaxPool2D((2,2)))
        model.add(layers.Conv2D(128,(3,3),activation='relu'))
        model.add(layers.MaxPool2D((2,2)))
        model.add(layers.Conv2D(128,(3,3),activation='relu'))
        model.add(layers.MaxPool2D((2,2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512,activation='relu'))
        model.add(layers.Dense(1,activation='sigmoid'))


        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.RMSprop(lr=1e-4))




        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        #the fllow is generator,only stop by other produce,which name is steps_per_epoch
        train_generator = train_datagen.flow_from_directory(
            file_data.train_dir,
            target_size =(150,150),
            batch_size=20,
            class_mode='binary')

        validation_generator = test_datagen.flow_from_directory(
            file_data.validation_dir,
            target_size = (150,150),
            batch_size=20,
            class_mode='binary'
        )

        #其中的100和50正是从生成器中取的数目
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs = 5,
            validation_data=validation_generator,
            validation_steps=50
        )

        model.save('cat_and_dog_small_1.h5')
        pickle.dump(history,history_path)
    else:
        history_path = open('model_history', 'rb')
        model = models.load_model(path)
        history = pickle.load(history_path)
        history_path.close()
    return model,history

#3.增强图像版设置
def stronger_model(file_data):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,#随机错切变换角度
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'#填充新创建像素的方法
    )

    #选择一张图片进行观看和显示
    def show_it():
        from keras.preprocessing import image

        fnames = [os.path.join(file_data.train_cats_dir,fname) for fname in os.listdir(file_data.train_cats_dir)]
        img_path = fnames[3]
        img = image.load_img(img_path,target_size=(150,150))
        x = image.img_to_array(img)
        x = x.reshape((1,)+x.shape)
        i = 0
        for batch in datagen.flow(x,batch_size=1):
            plt.figure(i)
            imgplot = plt.imshow(image.array_to_img(batch[0]))
            i+=1
            if i%4==0:
                break
        plt.show()

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,  # 随机错切变换角度
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'  # 填充新创建像素的方法
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        file_data.train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        file_data.validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=100,
        validation_data = validation_generator,
        validation_steps=50
    )

    model.save('cats_and_dogs_small_2.h5')


#4.VGG16的设置,没有使用增强的特征
def VGG16_model(file_data):

    from keras.applications import VGG16
    path_model = 'vgg16_model'


    if not os.path.exists(path_model):
        path_history = open('vgg16_history', 'wb')
        #卷积基部分及去掉密连接的设置
        conv_base = VGG16(weights='imagenet',
                          include_top =False,#是否适用密集连接分类器，默认对应ImageNet的1000类别，我们使用自己的只有2类
                          input_shape=(150,150,3))

        datagen = ImageDataGenerator(rescale=1./255)
        batch_size=20
        def extract_feature(directory,sample_count):
            features = np.zeros(shape=(sample_count,4,4,512))#结构取决于VGG16的层设置
            labels = np.zeros(shape=(sample_count))
            generator = datagen.flow_from_directory(
                directory,
                target_size=(150,150),
                batch_size=batch_size,
                class_mode='binary'
            )
            i = 0
            for input_batch,labels_batch in generator:
                features_batch = conv_base.predict(input_batch)
                features[i*batch_size:(i+1)*batch_size] = features_batch
                labels[i*batch_size:(i+1)*batch_size] = labels_batch
                i+=1
                if i*batch_size>=sample_count:
                    break
            return features,labels

        train_feature,train_labels = extract_feature(file_data.train_dir,2000)
        validation_feature,validation_labels = extract_feature(file_data.validation_dir,1000)
        # test_feature,test_labels = extract_feature(file_data.test_dir,1000)

        train_feature = np.reshape(train_feature,(2000,4*4*512))
        validation_feature = np.reshape(validation_feature, (1000, 4 * 4 * 512))
        # test_feature = np.reshape(test_feature, (1000 * 4 * 4 * 512))

        #卷积部分
        model = models.Sequential()
        model.add(layers.Dense(256,activation='relu',input_dim = 4*4*512))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1,activation='sigmoid'))

        model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                      loss='binary_crossentropy',
                      metrics=['acc'])
        history = model.fit(train_feature,train_labels,
                            epochs=20,
                            batch_size=20,
                            validation_data=(validation_feature,validation_labels))
        model.save(path_model)
        pickle.dump(history,path_history)
        path_history.close()
    else:
        path_history = open('vgg16_history', 'rb')
        model = models.load_model(path_model)
        history = pickle.load(path_history)
        path_history.close()
    return model, history

    #如果需要使用数据增强，则可以调模型为下面的
    model  = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))

    conv_base.trainable=False#注意要冻结已经训练好的模型


#微调模型   1.训练好的基网络定义网络   2.冻结基网络 3.训练添加部分    4.解冻部分层     5.联合训练解冻部分和添加部分
def smallchange_VGG16(file_data):
    from keras.applications import VGG16
    path_model = 'vgg16_model_smallchange'

    if not os.path.exists(path_model):
        path_history = open('vgg16_history_smallchange', 'wb')
        # 卷积基部分及去掉密连接的设置
        conv_base = VGG16(weights='imagenet',
                          include_top=False,  # 是否适用密集连接分类器，默认对应ImageNet的1000类别，我们使用自己的只有2类
                          input_shape=(150, 150, 3))

        datagen = ImageDataGenerator(rescale=1. / 255)
        batch_size = 20

        def extract_feature(directory, sample_count):
            features = np.zeros(shape=(sample_count, 4, 4, 512))  # 结构取决于VGG16的层设置
            labels = np.zeros(shape=(sample_count))
            generator = datagen.flow_from_directory(
                directory,
                target_size=(150, 150),
                batch_size=batch_size,
                class_mode='binary'
            )
            i = 0
            for input_batch, labels_batch in generator:
                features_batch = conv_base.predict(input_batch)
                features[i * batch_size:(i + 1) * batch_size] = features_batch
                labels[i * batch_size:(i + 1) * batch_size] = labels_batch
                i += 1
                if i * batch_size >= sample_count:
                    break
            return features, labels

        conv_base.trainable=True
        set_trainable = False
        for layer in conv_base.layers:
            if layer.name =='block5_conv1':
                set_trainable=True
            if set_trainable:
                layer.trainable=True
            else:
                layer.trainable=False



        train_feature, train_labels = extract_feature(file_data.train_dir, 2000)
        validation_feature, validation_labels = extract_feature(file_data.validation_dir, 1000)
        # test_feature,test_labels = extract_feature(file_data.test_dir,1000)

        train_feature = np.reshape(train_feature, (2000, 4 * 4 * 512))
        validation_feature = np.reshape(validation_feature, (1000, 4 * 4 * 512))
        # test_feature = np.reshape(test_feature, (1000 * 4 * 4 * 512))

        # 卷积部分
        model = models.Sequential()
        model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                      loss='binary_crossentropy',
                      metrics=['acc'])
        history = model.fit(train_feature, train_labels,
                            epochs=20,
                            batch_size=20,
                            validation_data=(validation_feature, validation_labels))
        model.save(path_model)
        pickle.dump(history, path_history)
        path_history.close()
    else:
        path_history = open('vgg16_history_smallchange', 'rb')
        model = models.load_model(path_model)
        history = pickle.load(path_history)
        path_history.close()
    return model, history









#绘制损失函数图形
def plot_loss(history):

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1,len(loss)+1)

    plt.plot(epochs,loss,'ro',label='Traninging loss')
    plt.plot(epochs,val_loss,'r',label='validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()




def main():
    file_data = path_and_data()
    file_data.copy_excate_file()

    # stronger_model(file_data)

    # models,history = VGG16_model(file_data)
    models,history = smallchange_VGG16(file_data)
    # model,history = network_model(file_data)
    plot_loss(history)



if __name__ =='__main__':
    main()
