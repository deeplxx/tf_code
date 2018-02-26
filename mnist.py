from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import time

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 定义运算
#
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, w) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=1))  # log_likelihood
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)  # 优化算法
tf.summary.histogram('weight', w)
tf.summary.histogram('bias', b)
tf.summary.scalar('loss', cross_entropy)
tf.summary.image('input', x, 3)


if 'session' in locals() and sess is not None:
    print('Close interactive session')
    sess.close()

# run
#
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('/logs', sess.graph)
t1 = time.time()
tf.global_variables_initializer().run()
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

# test
#
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# writer.add_summary('accur', accuracy)
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
t2 = time.time() - t1
print(t2)
# sess.close()
