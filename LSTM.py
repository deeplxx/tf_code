import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn, legacy_seq2seq, framework
from rnn.ptb import reader
import inspect


class PTBInput(object):
    """定义输入数据的参数"""
    def __init__(self, config, data, name=None):
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps  # LSTM反向传播的展开步数（状态数，上下文关联数，LSTMCell的个数）
        self.epoch_size = ((len(data) // self.batch_size) - 1) // self.num_steps
        """ ??? """
        self.input_data, self.targets = reader.ptb_producer(data, self.batch_size, self.num_steps, name)
        """shape：[batch_size, num_steps]"""


class PTBModel(object):
    """LSTM模型"""
    def __init__(self, is_training, config, ptb_input):
        self._input = ptb_input
        self._is_training = is_training

        batch_size = ptb_input.batch_size
        num_steps = ptb_input.num_steps  # 反向传播的展开步数（状态数）
        hidden_size = config.hidden_size  # LSTMCell的节点数（隐层列个数）
        vocab_size = config.vocab_size  # 词汇表大小（输出层列个数）

        def lstm_cell():
            """返回一个LSTMcell，每个cell是一个单隐层的网络"""
            return rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, reuse=tf.get_variable_scope().reuse)

        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                """若需要dropout则返回一个经过dropout的cell"""
                return rnn.DropoutWrapper(lstm_cell(), output_keep_prob=config.keep_prob)
        cell = rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)])
        """用num_layers个LSTMCell堆叠成一个cell,即一个cell中，第一个LSTMCell的输出变成下一个LSTMCell的输入"""

        # 初始状态
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        """state是个tuple，大小为num_layers"""

        # 输入
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', (vocab_size, hidden_size), tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, ptb_input.input_data)
            """inputs[batch_size, num_steps, hidden_size],其中第二个维度在vocab_size中取值
                num_steps个cell的输入，每个cell的inputs是 [batch, hidden_size]
            """
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # 隐层输出
        outputs = list()
        state = self._initial_state  # 细胞状态
        with tf.variable_scope('RNN'):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
                """outputs[num_steps, batch_size, hidden_size]"""
        outputs_flat = tf.reshape(tf.concat(outputs, 1), (-1, hidden_size))
        """outputs_flat：[y1, y2, y3, y1, y2, y3, ...].T"""

        # 输出层
        softmax_w = tf.get_variable('softmax_w', (hidden_size, vocab_size), tf.float32)
        softmax_b = tf.get_variable('sotfmax_b', [vocab_size], tf.float32)
        logits = tf.nn.bias_add(tf.matmul(outputs_flat, softmax_w), softmax_b)
        """logits[num_steps * batch_size, vocab_size]"""
        loss = legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(ptb_input.targets, [-1])],
                                                       [tf.ones([batch_size * num_steps])])
        """对每个logit，target对分别计算loss然后对这些loss进行加权求和"""
        self._cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state
        tf.summary.histogram('softmax_w', softmax_w)
        tf.summary.histogram('softmax_b', softmax_b)
        tf.summary.scalar('cost', self._cost)

        if not is_training:
            return

        # 优化
        self._lr = tf.Variable(0.0, trainable=False)
        trainable_var = tf.trainable_variables()  # 获取所有可训练的变量
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_var), config.max_grad_norm)  # 梯度截断
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, trainable_var),
                                                   global_step=framework.get_or_create_global_step())
        self.new_lr = tf.placeholder(tf.float32, [], name='new_learing_rate')
        self.lr_update = tf.assign(self.lr, self.new_lr)
        tf.summary.scalar('lr', self._lr)
        self._merge = tf.summary.merge_all()

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def merge(self):
        return self._merge

    @property
    def is_training(self):
        return self._is_training


class SmallConfig(object):

    init_scale = 0.1  # 权重的初始scale
    learning_rate = 1.0  # 初始学习率
    max_grad_norm = 5  # 梯度的最大范数，截断用
    num_layers = 2  # cell的层数
    num_steps = 20  # cell数
    hidden_size = 200  # 隐层单元个数
    max_epoch = 4  # 初始学习率迭代次数
    max_max_epoch = 13  # 总的epoch数
    keep_prob = 1.0
    lr_delay = 0.5  # 学习速率的衰减
    batch_size = 20
    vocab_size = 10000  # 输出单元数


class MediumConfig(object):

    init_scale = 0.05  # 小一些有助于温和训练
    learning_rate = 1.0
    max_grad_norm = 5  # 梯度的最大范数，截断用
    num_layers = 2  # cell的层数
    num_steps = 35  # cell数
    hidden_size = 650  # 隐层单元个数
    max_epoch = 6  # 初始学习率迭代次数
    max_max_epoch = 39  # 总的epoch数
    keep_prob = 0.5
    lr_delay = 0.8  # 迭代次数增加所以衰减速率变小
    batch_size = 20
    vocab_size = 10000  # 输出单元数


class LargeConfig(object):

    init_scale = 0.04  # 权重的初始scale
    learning_rate = 1.0  # 初始学习率
    max_grad_norm = 10  # 梯度的最大范数，截断用
    num_layers = 2  # cell的层数
    num_steps = 35  # cell数
    hidden_size = 1500  # 隐层单元个数
    max_epoch = 14  # 初始学习率迭代次数
    max_max_epoch = 55  # 总的epoch数
    keep_prob = 0.35
    lr_delay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000  # 输出单元数


def run_epoch(sess, model, writer=None, eval_op=None, verbose=False):
    """训练函数
    
    Args:
        writer: A FileWriter to add merge
        sess: A Session
        model: A PTBModle
        eval_op: A op 额外需要计算
        verbose: 是否打印训练过程

    Returns: 
        preplexity
    """
    start_time = time.time()
    costs = 0.0
    iters = 0  # 迭代次数：epoch_size * num_steps
    state = sess.run(model.initial_state)
    fetchs = {
        'cost': model.cost,
        'final_state': model.final_state,
    }
    if model.is_training:
        fetchs['merge'] = model.merge

    if eval_op is not None:
        fetchs['eval_op'] = eval_op

    # run
    for step in range(model.input.epoch_size):
        feed_dict = dict()
        for i, (h1, h2) in enumerate(model.initial_state):
            feed_dict[h1] = state[i].c
            feed_dict[h2] = state[i].h

        vals = sess.run(fetchs, feed_dict)

        cost = vals['cost']
        # state = vals['final_state']
        if model.is_training and (writer is not None):
            writer.add_summary(vals['merge'], step)

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print('complete: %.3f, perplexity: %.3f, speed: %.0f wps' %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)

if __name__ == '__main__':
    raw_data = reader.ptb_raw_data('D:/work/source/simple-examples/data/')
    train_data, valid_data, test_data, _ = raw_data

    config = SmallConfig()
    eval_config = SmallConfig()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        with tf.name_scope('Train'):
            train_input = PTBInput(config, train_data, name='TrainInput')
            with tf.variable_scope('Model', initializer=initializer):
                m = PTBModel(True, config, train_input)

        with tf.name_scope('Valid'):
            valid_input = PTBInput(config, valid_data, name='ValidInput')
            with tf.variable_scope('Model', reuse=True, initializer=initializer):
                m_valid = PTBModel(False, config, valid_input)

        with tf.name_scope('Test'):
            test_input = PTBInput(eval_config, test_data, name='TestInput')
            with tf.variable_scope('Model', reuse=True, initializer=initializer):
                m_test = PTBModel(False, eval_config, test_input)

        sv = tf.train.Supervisor()
        with sv.managed_session() as sess:
            writer = tf.summary.FileWriter('/logs', tf.get_default_graph())
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_delay ** max(i + 1 - config.max_epoch, 0)
                m.assign_lr(sess, config.learning_rate * lr_decay)

                print('Epoch: %d, learinig rate: %.3f' % (i + 1, sess.run(m.lr)))
                train_perplexity = run_epoch(sess, m, writer)
                print('Epoch: %d, Train perplexity: %.3f' % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(sess, m_valid)
                print('Epoch: %d, Valid perplexity: %.3f' % (i + 1, valid_perplexity))

            test_perplexity = run_epoch(sess, m_test)
            print('Test perplexity: %.3f' % test_perplexity)
            writer.close()
