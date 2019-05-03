#coding:utf-8

#==================================获得mnist的数据集==================================================================
# 1.定义算法公式，也就是神经网络forward的计算
# 2.定义loss，选定优化器，并制定优化器优化loss
# 3.迭代地对数据进行训练
# 4.在测试集或验证集上对准确率进行评测
#=====================================================================================================================
def get_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)
    return mnist
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    print(mnist.train.images.shape,mnist.train.labels.shape)
    print(mnist.test.images.shape,mnist.test.labels.shape)
    print(mnist.validation.images.shape,mnist.validation.labels.shape)

#==========================================利用regresion对数据数字识别进行预测=========================================
def tf_regresion(mnist):
    import tensorflow as tf
    #会注册一个session,之后的运算会在这个session里，不同的session的运算和数据是独立的
    sess = tf.InteractiveSession()
    
    #place 主要存放存放数据的地方
    x = tf.placeholder(tf.float32,[None,784])
    
    #variable主要存储模型参数,是持久化的
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    
    y = tf.nn.softmax(tf.matmul(x,W)+b)
    
    #placeholder 是输入数据的地方，第一个参数是数据类型，第二个参数是数据尺寸，None代表不限。
    y_ = tf.placeholder(tf.float32,[None,10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *tf.log(y),reduction_indices =[1]))
    #优化
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
    #全局参数初始化
    tf.global_variables_initializer().run()
    
    #随机抽取，防止计算量太大
    for i in range(1000):
        batch_xs,batch_ys = mnist.train.next_batch(100)
        train_step.run({x: batch_xs,y_:batch_ys})
        
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))

#=============================================实现自编码器=============================================================
#===================输出的并不是分类结果，而是复原数据，使得能够通过频率较高的特征进行表示=====================
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#参数初始化，Xavier初始化，会根据某一层网络的输入，输出节点自动调整最合适的分布
#如果深度学习模型的权重初始化太小，那么信号将在每个层传递时逐渐缩小而难以产生作用，但如果权重初始化太大，那么信号
#在每层间传递时逐渐放大而难以产生作用，xaiver初始化其做的事情就是让权重被初始化得不大不小
#数学上解释就是让权重满足0得均值，同时方差为2/Nin+Nout
def xavier_init(fan_in,fan_out,constant = 1):
    low = -constant *np.sqrt(6.0/(fan_in +fan_out))
    high = constant *np.sqrt(6.0/(fan_in +fan_out))
    return tf.random_uniform((fan_in,fan_out),
                             minval = low,maxval = high,
                             dtype = tf.float32)


#去噪自编码器class，包括神经网络的设计、权重的初始化以及常用的几个成员函数
class AdditiveGaussianNoiseAutocoder(object):
    # n_input:输入变量，n_hidden:隐含层节点数
    # transfer_function(隐含层激活函数),optimizer(优化器，默认为Adam),scale(高斯噪声系数，默认0.1)
    def __init__(self,n_input,n_hidden,n_hidden_2,trainsfer_function= tf.nn.softplus,
                 optimizer = tf.train.AdamOptimizer(),scale = 0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_hidden_2 = n_hidden_2
        self.transfer = trainsfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        netword_weights = self._intialize_weights()
        self.weights = netword_weights

        #定义网络结构
        #将加了噪声的x与隐含层权重想乘，再加上偏置
        self.x = tf.placeholder(tf.float32,[None,self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(
            self.x+scale*tf.random_normal((n_input,)),
            self.weights['w1']),self.weights['b1']))

        self.hidden_2 = self.transfer(tf.matmul(
            self.hidden,self.weights['w3']))

        self.reconstruction = tf.add(tf.matmul(self.hidden_2,
                                               self.weights['w2']),self.weights['b2'])

        #定义损失函数
        self.cost = 0.5*tf.reduce_sum(tf.pow(tf.subtract(
            self.reconstruction,self.x),2.0))
        self.optimizer = optimizer.minimize(self.cost)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    #给与了模型的参数
    def _intialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype = tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],dtype=tf.float32))

        all_weights['w3'] = tf.Variable(xavier_init(self.n_hidden,self.n_hidden_2))
        all_weights['b3'] = tf.Variable(tf.zeros([self.n_input],dtype = tf.float32))
        return all_weights

    #定义计算损失cost及执行一步训练的函数partial_fit，执行两个计算图的节点，该函数就是用一个batch数据进行训练并返回当前的损失cost。
    def partial_fit(self,X):
        cost,opt = self.sess.run((self.cost,self.optimizer),
                                 feed_dict  = {self.x: X,self.scale: self.training_scale})
        return cost

    #只求cost，只让session执行一个计算图，这个函数是在自编码器训练完毕后，在测试集上对模型性能进行评测时用到的
    def calc_total_cost(self,X):
        return self.sess.run(self.cost,feed_dict = {self.x:X,
                                                    self.scale:self.training_scale
        })

    #返回自编码器隐含层的结果。目的是提供一个接口获取抽象后的特征，自编码器的隐含层学习数据中高阶特征
    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict={self.x:X,
                                                    self.scale:self.training_scale})

    def generate(self,hidden = None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights["b1"])
        return self.sess.run(self.reconstruction,feed_dict= {self.hidden: hidden})

    #加上一个隐含层
    def generate_2(self,hidden_2 = None):
        if hidden_2 is None:
            hidden_2 = np.random.normal(size = self.weights["b3"])
        return self.sess.run(self.reconstructino,feed_dict ={self.hidden_2:hidden_2})


    def reconstruct(self,X):
        return self.sess.run(self.reconstruction,feed_dict = {self.x:X,self.scale:self.training_scale})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])
    def getBiases(self):
        return self.sess.run(self.weights['b1'])


#对数据进行处理，标准化，另外训练集和测试集要一样
def standard_scale(X_train,X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train,X_test

def get_random_block_from_data(data,batch_size):
    start_index = np.random.randint(0,len(data)-batch_size)
    return data[start_index:(start_index + batch_size)]


def main():
    mnist = get_mnist()
    #print(mnist.train.images.shape, mnist.train.labels.shape)
    X_train,X_test = standard_scale(mnist.train.images,mnist.test.images)

    n_samples = 55000
    train_epochs = 20#训练次数
    batch_size = 128#每一次就
    display_step = 1

    autoencoder = AdditiveGaussianNoiseAutocoder(n_input = 784,
                                                 n_hidden = 400,
                                                 n_hidden_2= 400,
                                                 trainsfer_function=tf.nn.softplus,
                                                 optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                                 scale = 0.01)

    for epoch in range(train_epochs):
        avg_cost = 0.
        total_batch = int(n_samples/batch_size)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train,batch_size)
            cost = autoencoder.partial_fit(batch_xs)
            avg_cost += cost/n_samples *batch_size

        if epoch %display_step ==0:
            print("Epoch:",'%04d' % (epoch+1),"cost = ","{:.9f}".format(avg_cost))

    print("total cost:" + str(autoencoder.calc_total_cost(X_test)))




if __name__ == "__main__":
    main()

    

    





