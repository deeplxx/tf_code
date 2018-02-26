from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()


# 权重初始化
#
def weight_init(w_shape):  # w_filter四个参数，12尺寸，3channel，4核数量
    weight = tf.truncated_normal(w_shape, stddev=0.1)
    bias = tf.constant(0.1, shape=(w_shape[-1], ))
    return tf.Variable(weight), tf.Variable(bias)


# 定义卷积层和池化层 ###
#
def conv2d(x, w_filter):
    return tf.nn.conv2d(x, w_filter, strides=(1, 1, 1, 1), padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # ksize池化窗口


# 定义运算 ###
#
x = tf.placeholder(tf.float32, (None, 784))
y_ = tf.placeholder(tf.float32, (None, 10))
x_image = tf.reshape(x, (-1, 28, 28, 1))  # 第二个参数四个数，1代表样本数量，23代表二维结构，4代表channel。-1代表推断

# 卷积层1
w_shape1 = (5, 5, 1, 32)
w_conv1, b_conv1 = weight_init(w_shape1)
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 卷积层2
w_shape2 = (5, 5, 32, 64)
w_conv2, b_conv2 = weight_init(w_shape2)
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层1 + dropput
w_shape3 = (7 * 7 * 64, 1024)
w_fc1, b_fc1 = weight_init(w_shape3)
h_pool2_flat = tf.reshape(h_pool2, (-1, 7*7*64))
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 全连接2 softmax层
w_shape4 = (1024, 10)
w_fc2, b_fc2 = weight_init(w_shape4)
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# 优化
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=1))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 评估
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 运行 ###
#
epoch = 1000
tf.global_variables_initializer().run()
for i in range(epoch):
    batch = mnist.train.next_batch(50)
    if i % 100 is 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy is %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

for i in range(10):
    print('Test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images[i*1000: (i+1)*1000],
                                                        y_: mnist.test.labels[i*1000: (i+1)*1000], keep_prob: 1.0}))

