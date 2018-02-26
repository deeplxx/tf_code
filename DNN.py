import tensorflow as tf
import numpy as np

# 定义batch大小
#
batch = 8

# 初始化参数
#
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))


# 声明训练数据
#
x = tf.placeholder(tf.float32, shape=(None, 2), name='input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='output')

# 传播过程
#
h = tf.matmul(x, w1)
y = tf.matmul(h, w2)

# 定义损失函数
# clip函数参数为（t, min, max, name=None）
#
# cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 生成数据集
#
data_size = 128
X = np.random.rand(data_size * 2).reshape((data_size, 2))
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

# 创建会话来运行
#
with tf.Session() as sess:
    # 初始化变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 初始化参数
    print(sess.run(w1))
    print(sess.run(w2))
    # 运行批量优化
    STEPS = 5000
    for i in range(STEPS):
        start_index = (i * batch) % data_size
        end_index = min(start_index + batch, data_size)
        # sess.run(train_step, feed_dict={x: X, y_: Y})
        sess.run(train_step, feed_dict={x: X[start_index: end_index], y_: Y[start_index: end_index]})
        if i % 1000 is 0:
            total_cross_entropy = sess.run(cross_entropy,
                                           feed_dict={x: X, y_: Y})
            print('After %d step, the total_cross_entropy is %g' % (i, total_cross_entropy))

    print(sess.run(w1))
    print(sess.run(w2))
