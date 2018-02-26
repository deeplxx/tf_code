from datetime import datetime
import tensorflow as tf
import time
import math


# 超参数
num_batch = 100
batch_size = 32
learning_rate = 1e-3

# 网络参数

# 占位符
keep_prob = tf.placeholder(tf.float32)


# 卷积层
def conv_op(input_op, name, p, kernel_shape, strides):
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + 'w', shape=kernel_shape, dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, strides=strides, padding='SAME')
        bias = tf.Variable(tf.constant(0.0, tf.float32, (kernel_shape[-1], )), name='b')
        activ = tf.nn.relu(tf.nn.bias_add(conv, bias), name=scope)
        p += (kernel, bias)

    return activ


# 全连接层
def fc_op(input_op, name, p, kernel_shape):
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + 'w', shape=kernel_shape, dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        bias = tf.Variable(tf.constant(0.1, tf.float32, (kernel_shape[-1], )), name='b')
        activ = tf.nn.relu_layer(input_op, kernel, bias, name=scope)
        p += (kernel, bias)

    return activ


# 最大池化层
def max_pool_op(input_op, name, ksize, strides):
    return tf.nn.max_pool(input_op, ksize=ksize, strides=strides, padding='SAME', name=name)


# 网络结构
def inference_op(input_op, keep_prob):
    p = list()

    conv1_1 = conv_op(input_op, 'conv1_1', p, kernel_shape=(3, 3, input_op.get_shape()[-1].value, 64),
                      strides=(1, 1, 1, 1))
    conv1_2 = conv_op(conv1_1, 'conv1_2', p, kernel_shape=(3, 3, 64, 64), strides=(1, 1, 1, 1))
    pool1 = max_pool_op(conv1_2, 'pool1', ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1))

    conv2_1 = conv_op(pool1, 'conv2_1', p, kernel_shape=(3, 3, pool1.get_shape()[-1].value, 128),
                      strides=(1, 1, 1, 1))
    conv2_2 = conv_op(conv2_1, 'conv2_2', p, kernel_shape=(3, 3, 128, 128), strides=(1, 1, 1, 1))
    pool2 = max_pool_op(conv2_2, 'pool2', ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1))

    conv3_1 = conv_op(pool2, 'conv3_1', p, kernel_shape=(3, 3, pool2.get_shape()[-1].value, 256),
                      strides=(1, 1, 1, 1))
    conv3_2 = conv_op(conv3_1, 'conv3_2', p, kernel_shape=(3, 3, 256, 256), strides=(1, 1, 1, 1))
    conv3_3 = conv_op(conv3_2, 'conv3_3', p, kernel_shape=(3, 3, 256, 256), strides=(1, 1, 1, 1))
    pool3 = max_pool_op(conv3_3, 'pool3', ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1))

    conv4_1 = conv_op(pool3, 'conv4_1', p, kernel_shape=(3, 3, pool3.get_shape()[-1].value, 512),
                      strides=(1, 1, 1, 1))
    conv4_2 = conv_op(conv4_1, 'conv4_2', p, kernel_shape=(3, 3, 512, 512), strides=(1, 1, 1, 1))
    conv4_3 = conv_op(conv4_2, 'conv4_3', p, kernel_shape=(3, 3, 512, 512), strides=(1, 1, 1, 1))
    pool4 = max_pool_op(conv4_3, 'pool4', ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1))

    conv5_1 = conv_op(pool4, 'conv5_1', p, kernel_shape=(3, 3, pool4.get_shape()[-1].value, 512),
                      strides=(1, 1, 1, 1))
    conv5_2 = conv_op(conv5_1, 'conv5_2', p, kernel_shape=(3, 3, 512, 512), strides=(1, 1, 1, 1))
    conv5_3 = conv_op(conv5_2, 'conv5_3', p, kernel_shape=(3, 3, 512, 512), strides=(1, 1, 1, 1))
    pool5 = max_pool_op(conv5_3, 'pool5', ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1))

    pool5_shape = pool5.get_shape()
    flat1 = pool5_shape[1].value * pool5_shape[2].value * pool5_shape[3].value
    pool5_flat = tf.reshape(pool5, (-1, flat1), 'pool5_flat')
    fc1 = fc_op(pool5_flat, 'fc1', p, (flat1, 4096))
    fc1_drop = tf.nn.dropout(fc1, keep_prob)

    fc2 = fc_op(fc1_drop, 'fc2', p, (4096, 4096))
    fc2_drop = tf.nn.dropout(fc2, keep_prob)

    fc3 = fc_op(fc2_drop, 'fc3', p, (4096, 1000))

    softmax = tf.nn.softmax(fc3)
    predic = tf.argmax(softmax, 1)

    return predic, softmax, fc3, p


# 评测运算时间
def time_tensor_run(sess, target, feed, info_string):
    num_step_burn_in = 10
    total_dur = 0.0
    total_dur_squ = 0.0

    for i in range(num_step_burn_in + num_batch):
        start_time = time.time()
        sess.run(target, feed_dict=feed)
        dur = time.time() - start_time
        if i >= num_step_burn_in and i % 10 == 0:
            total_dur += dur
            total_dur_squ += dur ** 2
            print('%s: %s cross step %d, dur: %.3f' % (datetime.now(), info_string, i - num_step_burn_in, dur))

    mean = total_dur / num_batch
    stddev = math.sqrt(total_dur_squ / num_batch - mean ** 2)
    print('%.3f +/- %.3f sec/batch)' % (mean, stddev))


# 测试函数
def run_benchmark():
    image_size = 224
    images = tf.Variable(tf.random_normal((batch_size, image_size, image_size, 3), stddev=1e-1))
    predic, _, fc3, p = inference_op(images, keep_prob=keep_prob)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    time_tensor_run(sess, predic, {keep_prob: 1.0}, 'Forward')
    objective = tf.nn.l2_loss(fc3)
    grad = tf.gradients(objective, p)
    time_tensor_run(sess, grad, {keep_prob: 0.5}, 'Forward-Backward')
    sess.close()


if __name__ is '__main__':
    run_benchmark()
