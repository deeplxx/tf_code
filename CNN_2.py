# import cifar10.cifar10_input
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import time
import math

# 超参数
epochs = 3000
batch_size = 128
# data_dir = 'd:/work/source/tmp/cifar10_data/cifar-10-batches-bin'  # cifar10数据集位置
# images_train, labels_train = cifar10.cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)  # 数据集增强
# images_test, labels_test = cifar10.cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 占位
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# 辅助函数 ###
#
# 初始化weight 并将对应的weight_loss添加到‘loss’collection 中
def variable_weight_loss(shape, b_init, stddev, w_loss):
    """
    :param b_init: 
    :param shape: 
    :param stddev: 标准差
    :param w_loss: 惩罚系数 
    :return: 
    """
    w = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    b = tf.Variable(tf.constant(b_init, tf.float32, (shape[-1], )))
    if w_loss is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(w), w_loss, name='weight_loss')
        tf.add_to_collection('loss', weight_loss)
    return w, b


# 卷积层
def conv2d(x, w_filter):
    return tf.nn.conv2d(x, w_filter, strides=(1, 1, 1, 1), padding='SAME')


# 池化层
def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME')  # ksize池化窗口


# loss（cross_entropy + l2正则）
def loss(y, y_):
    y_ = tf.cast(y_, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y, name='cross_entropy_per')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('loss', cross_entropy_mean)
    return tf.add_n(tf.get_collection('loss'), name='total_loss')


# 网络结构 ###
#
# 卷积层1
w_filter1 = (5, 5, 3, 64)
w_conv1, b_conv1 = variable_weight_loss(w_filter1, b_init=0.0, stddev=4e-2, w_loss=0.0)
tf.summary.histogram('w_conv1', w_conv1)
h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x, w_conv1), b_conv1))
h_pool1 = max_pool_3x3(h_conv1)
norm1 = tf.nn.lrn(h_pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# 卷积层2
w_filter2 = (5, 5, 64, 64)
w_conv2, b_conv2 = variable_weight_loss(w_filter2, b_init=0.1, stddev=5e-2, w_loss=0.0)
tf.summary.histogram('w_conv2', w_conv2)
h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(norm1, w_conv2), b_conv2))
norm2 = tf.nn.lrn(h_conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
h_pool2 = max_pool_3x3(norm2)

# 全连接层1
h_pool2_flat = tf.reshape(h_pool2, (batch_size, -1))
w_filter3 = (h_pool2_flat.get_shape()[1].value, 384)
w_fc1, b_fc1 = variable_weight_loss(w_filter3, b_init=0.1, stddev=4e-2, w_loss=4e-3)
h_fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_pool2_flat, w_fc1), b_fc1))

# 全连接层2
w_filter4 = (384, 192)
w_fc2, b_fc2 = variable_weight_loss(w_filter4, b_init=0.1, stddev=4e-2, w_loss=4e-3)
h_fc2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_fc1, w_fc2), b_fc2))

# 全连接层3
w_filter5 = (192, 10)
w_fc3, b_fc3 = variable_weight_loss(w_filter5, b_init=0.0, stddev=1 / 192.0, w_loss=0.0)
h_fc3 = tf.nn.bias_add(tf.matmul(h_fc2, w_fc3), b_fc3)

# 输出
y = h_fc3

# 优化
loss = loss(y, y_)
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op = tf.nn.in_top_k(y, y_, 1)  # 返回bool型数组（预测的准确与否）

# 运行 ###
#
sess = tf.InteractiveSession()
merge = tf.summary.merge_all()
tf.summary.FileWriter('/logs', sess.graph)
tf.global_variables_initializer().run()
# tf.train.start_queue_runners()  # 前面的数据增强用到了线程队列，故这里需要启动·

# 训练
for step in range(epochs):
    start_time = time.time()
    _, loss_value = sess.run((train_step, loss), feed_dict={x: mnist.train.images, y_: mnist.train.labels})
    delta_time = time.time() - start_time

    if step % 10 is 0:
        examples_per_sec = batch_size / delta_time
        sec_per_batch = float(delta_time)
        print('step %d, loss = %.2f, (%.1f examples/sec, %.1f sec/batch)' %
              (step, loss_value, examples_per_sec, sec_per_batch))

# 预测
num_examples = len(mnist.test.images)
num_batch = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_batch * batch_size
for step in range(num_batch):
    image_test_batch, labels_test_batch = sess.run((mnist.test.images, mnist.test.labels))
    prediction = sess.run(top_k_op, feed_dict={x: image_test_batch, y_: labels_test_batch})
    true_count += np.sum(prediction)

print('precision @ 1 = %.3f' % (true_count / total_sample_count))
