#coding:utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)
sess = tf.InteractiveSession()

#=========================================第一步  定义算法公式===========================================================
#设置权重和偏置，制造噪声打破完全对称，制造正值防止relu的死亡节点
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

#卷积层和池化层其中x是输入，W是卷积参数，如 [5,5,1,32] 前两个数代表卷积核尺寸，第三个channel，灰色为1，彩色为3，最后一个为提取特征数
#strides代表步长，padding代表边界的处理方式,SAME代表让卷积的输入输出保持同样的尺寸
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides = [1,1,1,1],padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')

#正式设计输入，先定义placeholder,x是特征,y是label，卷积网络需要把输入转成二维信息，因为一个颜色通道
#所以为[-1,28,28,1]，-1代表不确定输入数量
x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
x_image = tf.reshape(x,[-1,28,28,1])

#第一个卷积层
w_conv1 = weight_variable([5,5,1,32])#32代表卷积核
b_conv1 = bias_variable([32])
h_conv1 =tf.nn.relu(conv2d(x_image,w_conv1) +b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第二个卷积层
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2) +b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#使用tf.reshape函数对第二个卷积层的输出tensor进行变形，将其转化成1D的向量，然后连接成一个全连接层
#隐含节点为1024，并使用ReLU激活函数
w_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1) +b_fc1)

#减轻过拟合，使用一个dropout层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#将Dropout层输出到一个Softmax层
w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)


#=========================================第二步  定义损失及优化公式===========================================================
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),
                                              reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


#=========================================第三步  迭代地对数据进行训练===========================================================
tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict = {x:batch[0],y_:batch[1],
                                                    keep_prob:1.0})
        print("step %d,training accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict = {x:batch[0],y_:batch[1],keep_prob:0.5})

#第四步
print("test accuracy %g"%accuracy.eval(feed_dict = {
    x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0
}))