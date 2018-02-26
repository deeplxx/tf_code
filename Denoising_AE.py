import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# xavier 初始化器（均匀分布）
#
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


# 去噪自编码器
#
class AGN_AE(object):

    def __init__(self, n_input, n_hidden, activ_function=tf.nn.softplus,
                 optimiter=tf.train.AdamOptimizer(learning_rate=0.001), scale=0.01):
        self.n_input = n_input  # 输入层单元数
        self.n_hidden = n_hidden  # 隐层单元数
        self.activ_function = activ_function  # 激活函数
        self.scale = tf.placeholder(tf.float32)
        self.noise_scale = scale  # 高斯噪声系数
        self.weights = self.initialize_weights()  # 初始化权重
        self.x = tf.placeholder(tf.float32, shape=(None, self.n_input))  # 输入
        self.hidden = self.activ_function(tf.add(tf.matmul(self.x + self.noise_scale * tf.random_normal((n_input,)),
                                                           self.weights['w1']), self.weights['b1']))  # 隐层输出
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])  # 输出层恢复后输出

        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimiter = optimiter.minimize(self.cost)  # 优化器
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    # 初始化权重函数
    def initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros((self.n_hidden, ), dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros((self.n_hidden, self.n_input), dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros((self.n_input, ), dtype=tf.float32))

        return all_weights

    # 运行优化器并返回代价函数值
    def partial_cost(self, x):
        cost, opt = self.sess.run((self.cost, self.optimiter), feed_dict={self.x: x, self.scale: self.noise_scale})
        return cost

    def total_cost(self, x):
        return self.sess.run(self.cost, feed_dict={self.x: x, self.scale: self.noise_scale})

    # 运行隐层激活函数(网络前半部分)
    def transform(self, x):
        return self.sess.run(self.hidden, feed_dict={self.x: x, self.scale: self.noise_scale})

    # 运行输出层输出（网络后半部分）
    def generation(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    # 运行整个流程（整个网络）
    def reconstructino(self, x):
        return self.sess.run(self.reconstruction, feed_dict={self.x: x, self.scale: self.noise_scale})

    # 获取权重w1 (降维使用)
    def get_weights(self):
        return self.sess.run(self.weights['w1'])

    # 获取b1 （降维使用）
    def get_bias(self):
        return self.sess.run(self.weights['b1'])


# 数据标准化 （目标函数的基础都是假设所有的特征都是零均值并且具有同一阶数上的方差！！！！！！！！！）
#
def standard_scale(x_train, x_test):
    preprocessor = prep.StandardScaler().fit(x_train)  # 返回的scaler的均值和方差跟x_train相同，train和test必须公用才可以保证后面的一致性
    x_train = preprocessor.transform(x_train)
    x_test = preprocessor.transform(x_test)

    return x_train, x_test


# 随机获取 batch
#
def get_random_batch(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index: (start_index + batch_size)]


# 训练
#
if __name__ is '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    x_Train, x_Test = standard_scale(mnist.train.images, mnist.test.images)
    n_Sample = int(mnist.train.num_examples)  # 总训练样本数
    train_Epoch = 20  # 训练轮数
    batch_Size = 128  # batch大小
    display_Step = 1  # print用

    auto_Encoder = AGN_AE(784, 200)
    for epoch in range(train_Epoch):
        avg_cost = 0
        batch_count = int(n_Sample / batch_Size)
        for i in range(batch_count):
            batch_x = get_random_batch(x_Train, batch_Size)
            _cost = auto_Encoder.partial_cost(batch_x)
            avg_cost += _cost / n_Sample * batch_Size

        if epoch % display_Step == 0:
            print('Epoch: %04d' % (epoch + 1), 'cost:', '{:.9f}'.format(avg_cost))
