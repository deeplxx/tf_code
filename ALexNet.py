# import
from datetime import datetime

import tensorflow as tf
import time
import math

# 超参数
batch_size = 32
num_batch = 100
learning_rate = 0.01


# # 占位
# x = tf.placeholder(tf.float32, (None, -1))
# y_ = tf.placeholder(tf.float32, (None, 10))
# keep_prob = tf.placeholder(tf.float32)

# # 网络参数
# weights = {
#     'w_conv1': tf.Variable(tf.truncated_normal((11, 11, 3, 64), stddev=1e-1), name='weights'),
#     'w_conv2': tf.Variable(tf.truncated_normal((11, 11, 3, 64), stddev=1e-1), name='weights'),
#     'w_conv3': tf.Variable(tf.truncated_normal((11, 11, 3, 64), stddev=1e-1), name='weights'),
#     'w_conv4': tf.Variable(tf.truncated_normal((11, 11, 3, 64), stddev=1e-1), name='weights'),
#     'w_conv5': tf.Variable(tf.truncated_normal((11, 11, 3, 64), stddev=1e-1), name='weights'),
#     'w_fc1': tf.Variable(tf.truncated_normal((11, 11, 3, 64), stddev=1e-1), name='weights'),
#     'w_fc2': tf.Variable(tf.truncated_normal((11, 11, 3, 64), stddev=1e-1), name='weights'),
#     'w_fc3': tf.Variable(tf.truncated_normal((11, 11, 3, 64), stddev=1e-1), name='weights')
# }
#
# bias = {
#     'b_conv1': tf.Variable(tf.constant(0.0, shape=(64, ), dtype=tf.float32), name='bias'),
#     'b_conv2': tf.Variable(tf.constant(0.0, shape=(64, ), dtype=tf.float32), name='bias'),
#     'b_conv3': tf.Variable(tf.constant(0.0, shape=(64, ), dtype=tf.float32), name='bias'),
#     'b_conv4': tf.Variable(tf.constant(0.0, shape=(64, ), dtype=tf.float32), name='bias'),
#     'b_conv5': tf.Variable(tf.constant(0.0, shape=(64, ), dtype=tf.float32), name='bias'),
#     'b_fc1': tf.Variable(tf.constant(0.0, shape=(64, ), dtype=tf.float32), name='bias'),
#     'b_fc2': tf.Variable(tf.constant(0.0, shape=(64, ), dtype=tf.float32), name='bias'),
#     'b_fc3': tf.Variable(tf.constant(0.0, shape=(64, ), dtype=tf.float32), name='bias')
# }


# 辅助函数 ###
# 展示每层输出tensor的尺寸和名字
def print_activ(t):
    print(t.op.name, ':', t.get_shape().as_list())


# 评估每轮计算时间
def time_tensorflow_run(session, target, info_string):
    num_step_burn_in = 10  # 定义预热轮数（预热阶段有显存加载等影响时间）
    total_dur = 0.0
    total_dur_squ = 0.0

    for i in range(num_batch + num_step_burn_in):
        start_time = time.time()
        session.run(target)
        dur = time.time() - start_time

        if i >= num_step_burn_in:
            if not i % 10:
                print('%s: step %d, dur = %.3f' % (datetime.now(), i - num_step_burn_in, dur))
            total_dur += dur
            total_dur_squ += dur ** 2
    mean_dur = total_dur / num_batch  # 每轮迭代平均用时
    stddev = math.sqrt(total_dur_squ / num_batch - mean_dur ** 2)
    print('%s: %s across %d steps, %.3f +- %.3f sec/batch' %
          (datetime.now(), info_string, num_batch, mean_dur, stddev))


# 网络结构
def inference(images):
    parameters = list()  # 保存每层 weights 和 bias

    # 1 卷积 + lrn + 池化
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal((11, 11, 3, 64), stddev=1e-1), name='weights')
        tf.summary.histogram('w_conv1', kernel)
        conv = tf.nn.conv2d(images, kernel, strides=(1, 4, 4, 1), padding='SAME')
        bias = tf.Variable(tf.constant(0.0, shape=(64, ), dtype=tf.float32), name='bias')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv, bias), name=scope)
        parameters += (kernel, bias)
        print_activ(conv1)
    lrn1 = tf.nn.lrn(conv1, depth_radius=4, bias=1.0, alpha=1e-3 / 9, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID', name='pool1')
    print_activ(pool1)

    # 2 卷积 + lrn + 池化
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal((5, 5, 64, 192), stddev=1e-1), name='weights')
        tf.summary.histogram('w_conv2', kernel)
        conv = tf.nn.conv2d(pool1, kernel, strides=(1, 1, 1, 1), padding='SAME')
        bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=(192, )),  name='bias')
        conv2 = tf.nn.relu(tf.nn.bias_add(conv, bias), name=scope)
        parameters += (kernel, bias)
        print_activ(conv2)
    lrn2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=1e-3 / 9, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID', name='pool2')
    print_activ(pool2)

    # 3 卷积
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal((3, 3, 192, 384), stddev=1e-1), name='weights')
        tf.summary.histogram('w_conv3', kernel)
        conv = tf.nn.conv2d(pool2, kernel, strides=(1, 1, 1, 1), padding='SAME')
        bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=(384, )),  name='bias')
        conv3 = tf.nn.relu(tf.nn.bias_add(conv, bias), name=scope)
        parameters += (kernel, bias)
        print_activ(conv3)

    # 4 卷积
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal((3, 3, 384, 256), stddev=1e-1), name='weights')
        tf.summary.histogram('w_conv4', kernel)
        conv = tf.nn.conv2d(conv3, kernel, strides=(1, 1, 1, 1), padding='SAME')
        bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=(256, )),  name='bias')
        conv4 = tf.nn.relu(tf.nn.bias_add(conv, bias), name=scope)
        parameters += (kernel, bias)
        print_activ(conv4)

    # 5 卷积 + lrn + 池化
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal((3, 3, 256, 256), stddev=1e-1), name='weights')
        tf.summary.histogram('w_conv5', kernel)
        conv = tf.nn.conv2d(conv4, kernel, strides=(1, 1, 1, 1), padding='SAME')
        bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=(256, )),  name='bias')
        conv5 = tf.nn.relu(tf.nn.bias_add(conv, bias), name=scope)
        parameters += (kernel, bias)
        print_activ(conv5)
    lrn5 = tf.nn.lrn(conv5, depth_radius=4, bias=1.0, alpha=1e-3 / 9, beta=0.75, name='lrn5')
    pool5 = tf.nn.max_pool(lrn5, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID', name='pool5')
    print_activ(pool5)

    # # 6 全连接 + dropout
    # with tf.name_scope('fc1') as scope:
    #     pool5_flat = tf.reshape(pool5, (images.get_shape()[0].value, -1))
    #     kernel = tf.Variable(tf.truncated_normal((pool5_flat.get_shape()[1], 4096), stddev=1e-1), name='weights')
    #     bias = tf.Variable(tf.constant(0.0, tf.float32, (4096, )), name='bias')
    #     fc = tf.nn.relu(tf.nn.bias_add(tf.matmul(pool5_flat, kernel), bias), name='fc')
    #     fc1 = tf.nn.dropout(fc, keep_prob, name=scope)
    #     parameters += (kernel, bias)
    #     print_activ(fc1)
    #
    # # 7 全连接 + dropout
    # with tf.name_scope('fc2') as scope:
    #     kernel = tf.Variable(tf.truncated_normal((4096, 4096), stddev=1e-1), name='weights')
    #     bias = tf.Variable(tf.constant(0.0, tf.float32, (4096, )), name='bias')
    #     fc = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc1, kernel), bias), name='fc')
    #     fc2 = tf.nn.dropout(fc, keep_prob, name=scope)
    #     parameters += (kernel, bias)
    #     print_activ(fc2)
    #
    # # 8 全连接 + dropout
    # with tf.name_scope('fc3') as scope:
    #     kernel = tf.Variable(tf.truncated_normal((4096, 1000), stddev=1e-1), name='weights')
    #     bias = tf.Variable(tf.constant(0.0, tf.float32, (1000, )), name='bias')
    #     fc = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc2, kernel), bias), name='fc')
    #     fc3 = tf.nn.dropout(fc, keep_prob, name=scope)
    #     parameters += (kernel, bias)
    #     print_activ(fc3)

    return pool5, parameters


# 主函数 ###
#
def run_benchmark():
    image_size = 224
    image = tf.Variable(tf.random_normal((batch_size, image_size, image_size, 3), stddev=1e-1))
    y, parameter = inference(image)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y, name='cross_entropy')
    time_tensorflow_run(sess, y, 'Forward')
    objective = tf.nn.l2_loss(y)
    grad = tf.gradients(objective, parameter)
    time_tensorflow_run(sess, grad, 'Forward-backward')
    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter('/logs', sess.graph)
    writer.close()


if __name__ is '__main__':
    run_benchmark()
