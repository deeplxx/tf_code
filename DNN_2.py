from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

# 定义运算
#
in_unit = 784
h1_unit = 300
out_unit = 10
w1 = tf.Variable(tf.truncated_normal((in_unit, h1_unit), stddev=0.1))  # 产生截断的正态分布，目的是增加点噪声来打破完全对称并且避免0梯度
b1 = tf.Variable(tf.zeros((h1_unit, )))
w2 = tf.Variable(tf.zeros((h1_unit, out_unit)))
b2 = tf.Variable(tf.zeros((out_unit, )))
tf.summary.histogram('w1', w1)
tf.summary.histogram('b1', b1)
tf.summary.histogram('w2', w2)
tf.summary.histogram('b2', b2)
x = tf.placeholder(tf.float32, (None, in_unit))
y_ = tf.placeholder(tf.float32, (None, out_unit))
keep_prob = tf.placeholder(tf.float32)  # dropout保留的神经元的比率

h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
h1_drop = tf.nn.dropout(h1, keep_prob)
y = tf.nn.softmax(tf.matmul(h1_drop, w2) + b2)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=1))
tf.summary.scalar('cross_entropy', cross_entropy)
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

# 运行训练
#
epoch = 3000
tf.global_variables_initializer().run()
for i in range(epoch):
    batch_x, batch_y = mnist.train.next_batch(100)
    train_step.run({x: batch_x, y_: batch_y, keep_prob: 0.75})  # keep_prob训练时小于1，预测时等于1

# 预测
#
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # corret_pre 是一个bool型
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 这里要进行类别转换
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1}))
merge = tf.summary.merge_all()
witer = tf.summary.FileWriter('/logs', sess.graph)
witer.close()