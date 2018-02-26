import tensorflow as tf
from tensorflow.contrib import slim

from ALexNet import time_tensorflow_run


# 截断初始化函数的简化lambda
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def inception_v3_arg_scope(w_loss=4e-5, stddev=0.1, batch_norm_var_collection='moving_vars'):
    """
    设置函数的默认参数环境，包括weights_regularizer，weiths_initializer,normalizer_fn，normalizer_params
        :param w_loss: 正则项系数
        :param stddev: 标准差 
        :param batch_norm_var_collection: 不知道啥东西 
        :return: scope
    """
    batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 1e-3,  # 归一化时防止分母为0的一个小量
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'variables_collections': {
            'beta': None,  # 学习的系数
            'gamma': None,  # 。。。
            'moving_mean': [batch_norm_var_collection],  # 训练过程中的mean和var是一直变化的，
            'moving_variance': [batch_norm_var_collection]  # 。。。
        }
    }

    with slim.arg_scope((slim.conv2d, slim.fully_connected), weights_regularizer=slim.l2_regularizer(w_loss)):
        with slim.arg_scope([slim.conv2d], weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params) as sc:
            return sc


def inception_v3_base(inputs, scope=None):
    """
    实现所有卷积层
        :param inputs: 输入 
        :param scope: 默认参数环境
        :return: 卷积层输出，辅助节点字典
    """
    end_points = dict()

    with tf.variable_scope(scope, 'Inception_V3', [inputs]):

        # 非Inception Module
        with slim.arg_scope((slim.conv2d, slim.max_pool2d, slim.avg_pool2d), stride=1, padding='VALID'):
            net = slim.conv2d(inputs, 32, (3, 3), stride=2, scope='Conv2d_0a_3x3')
            net = slim.conv2d(net, 32, (3, 3), scope='Conv2d_0b_3x3')
            net = slim.conv2d(net, 64, (3, 3), padding='SAME', scope='Conv2d_0c_3x3')
            net = slim.max_pool2d(net, (3, 3), stride=2, scope='MaxPool_0d_3x3')
            net = slim.conv2d(net, 80, (1, 1), scope='Conv2d_0e_1x1')
            net = slim.conv2d(net, 192, (3, 3), scope='Conv2d_0f_3x3')
            net = slim.max_pool2d(net, (3, 3), stride=2, scope='MaxPool_0g_3x3')

        # Inception Module组
        with slim.arg_scope((slim.conv2d, slim.max_pool2d, slim.avg_pool2d), stride=1, padding='SAME'):

            # 35 x 35的module组，包括3个module, 输出为 35 x 35 x 288
            #
            with tf.variable_scope('Mixed_5b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, (1, 1), scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, (1, 1), scope='Conv2d_1a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, (5, 5), scope='Conv2d_1b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, (1, 1), scope='Conv2d_2a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, (3, 3), scope='Conv2d_2b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, (3, 3), scope='Conv2d_2c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, (3, 3), scope='AvgPool_3a_3x3')
                    branch_3 = slim.conv2d(branch_3, 32, (1, 1), scope='Conv2d_3b_1x1')
                net = tf.concat((branch_0, branch_1, branch_2, branch_3), 3)  # 在axis=3上合并即在输出通道上合并（通道数之和）

            with tf.variable_scope('Mixed_5c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, (1, 1), scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, (1, 1), scope='Conv2d_1a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, (5, 5), scope='Conv2d_1b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, (1, 1), scope='Conv2d_2a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, (3, 3), scope='Conv2d_2b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, (3, 3), scope='Conv2d_2c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, (3, 3), scope='AvgPool_3a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, (1, 1), scope='Conv2d_3b_1x1')
                net = tf.concat((branch_0, branch_1, branch_2, branch_3), 3)

            with tf.variable_scope('Mixed_5d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, (1, 1), scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, (1, 1), scope='Conv2d_1a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, (5, 5), scope='Conv2d_1b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, (1, 1), scope='Conv2d_2a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, (3, 3), scope='Conv2d_2b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, (3, 3), scope='Conv2d_2c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, (3, 3), scope='AvgPool_3a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, (1, 1), scope='Conv2d_3b_1x1')
                net = tf.concat((branch_0, branch_1, branch_2, branch_3), 3)

            # 17 x 17的module组，包括5个module, 输出为 17 x 17 x 768,并将第五个module的输出作为辅助节点
            #
            with tf.variable_scope('Mixed_6a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 384, (3, 3), stride=2, padding='VALID', scope='Conv2d_0a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 64, (1, 1), scope='Conv2d_1a_1x1')
                    branch_1 = slim.conv2d(branch_1, 96, (3, 3), scope='Conv2d_1b_3x3')
                    branch_1 = slim.conv2d(branch_1, 96, (3, 3), stride=2, padding='VALID', scope='Conv2d_1c_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, (3, 3), stride=2, padding='VALID', scope='MaxPool_2a_3x3')
                net = tf.concat((branch_0, branch_1, branch_2), 3)

            with tf.variable_scope('Mixed_6b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, (1, 1), scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 128, (1, 1), scope='Conv2d_1a_1x1')
                    branch_1 = slim.conv2d(branch_1, 128, (1, 7), scope='Conv2d_1b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, (7, 1), scope='Conv2d_1c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 128, (1, 1), scope='Conv2d_2a_1x1')
                    branch_2 = slim.conv2d(branch_2, 128, (7, 1), scope='Conv2d_2b_7x1')
                    branch_2 = slim.conv2d(branch_2, 128, (1, 7), scope='Conv2d_2c_1x7')
                    branch_2 = slim.conv2d(branch_2, 128, (7, 1), scope='Conv2d_2d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, (1, 7), scope='Conv2d_2e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, (3, 3), scope='AvgPool_3a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, (1, 1), scope='Conv2d_3b_1x1')
                net = tf.concat((branch_0, branch_1, branch_2, branch_3), 3)

            with tf.variable_scope('Mixed_6c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, (1, 1), scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, (1, 1), scope='Conv2d_1a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, (1, 7), scope='Conv2d_1b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, (7, 1), scope='Conv2d_1c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, (1, 1), scope='Conv2d_2a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, (7, 1), scope='Conv2d_2b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, (1, 7), scope='Conv2d_2c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, (7, 1), scope='Conv2d_2d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, (1, 7), scope='Conv2d_2e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, (3, 3), scope='AvgPool_3a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, (1, 1), scope='Conv2d_3b_1x1')
                net = tf.concat((branch_0, branch_1, branch_2, branch_3), 3)

            with tf.variable_scope('Mixed_6d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, (1, 1), scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, (1, 1), scope='Conv2d_1a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, (1, 7), scope='Conv2d_1b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, (7, 1), scope='Conv2d_1c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, (1, 1), scope='Conv2d_2a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, (7, 1), scope='Conv2d_2b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, (1, 7), scope='Conv2d_2c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, (7, 1), scope='Conv2d_2d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, (1, 7), scope='Conv2d_2e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, (3, 3), scope='AvgPool_3a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, (1, 1), scope='Conv2d_3b_1x1')
                net = tf.concat((branch_0, branch_1, branch_2, branch_3), 3)

            with tf.variable_scope('Mixed_6e'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, (1, 1), scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 192, (1, 1), scope='Conv2d_1a_1x1')
                    branch_1 = slim.conv2d(branch_1, 192, (1, 7), scope='Conv2d_1b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, (7, 1), scope='Conv2d_1c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 192, (1, 1), scope='Conv2d_2a_1x1')
                    branch_2 = slim.conv2d(branch_2, 192, (7, 1), scope='Conv2d_2b_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, (1, 7), scope='Conv2d_2c_1x7')
                    branch_2 = slim.conv2d(branch_2, 192, (7, 1), scope='Conv2d_2d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, (1, 7), scope='Conv2d_2e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, (3, 3), scope='AvgPool_3a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, (1, 1), scope='Conv2d_3b_1x1')
                net = tf.concat((branch_0, branch_1, branch_2, branch_3), 3)

            end_points['Mixed_6e'] = net

            # 8 x 8的Inception Module,包含3个Module，输出8 x 8 x 2018
            #
            with tf.variable_scope('Mixed_7a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, (1, 1), scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, 320, (3, 3), stride=2, padding='VALID', scope='Conv2d_0b_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 192, (1, 1), scope='Conv2d_1a_1x1')
                    branch_1 = slim.conv2d(branch_1, 192, (1, 7), scope='Conv2d_1b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, (7, 1), scope='Conv2d_1c_7x1')
                    branch_1 = slim.conv2d(branch_1, 192, (3, 3), stride=2, padding='VALID', scope='Conv2d_1d_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, (3, 3), stride=2, padding='VALID')
                net = tf.concat((branch_0, branch_1, branch_2), 3)

            with tf.variable_scope('Mixed_7b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, (1, 1), scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, (1, 1), scope='Conv2d_1a_1x1')
                    branch_1 = tf.concat((
                        slim.conv2d(branch_1, 384, (1, 3), scope='Conv2d_1b_1x3'),
                        slim.conv2d(branch_1, 384, (3, 1), scope='Conv2d_1b_3x1')
                    ), 3)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, (1, 1), scope='Conv2d_2a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, (3, 3), scope='Conv2d_2b_3x3')
                    branch_2 = tf.concat((
                        slim.conv2d(branch_2, 384, (1, 3), scope='Conv2d_2c_1x3'),
                        slim.conv2d(branch_2, 384, (3, 1), scope='Conv2d_2c_3x1')
                    ), 3)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, (3, 3), scope='AvgPool_3a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, (1, 1), scope='Conv2d_3b_1x1')
                net = tf.concat((branch_0, branch_1, branch_2, branch_3), 3)

            with tf.variable_scope('Mixed_7c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, (1, 1), scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, (1, 1), scope='Conv2d_1a_1x1')
                    branch_1 = tf.concat((
                        slim.conv2d(branch_1, 384, (1, 3), scope='Conv2d_1b1_1x3'),
                        slim.conv2d(branch_1, 384, (3, 1), scope='Conv2d_1b2_3x1')
                    ), 3)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, (1, 1), scope='Conv2d_2a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, (3, 3), scope='Conv2d_2b_3x3')
                    branch_2 = tf.concat((
                        slim.conv2d(branch_2, 384, (1, 3), scope='Conv2d_2c_1x3'),
                        slim.conv2d(branch_2, 384, (3, 1), scope='Conv2d_2c_3x1')
                    ), 3)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, (3, 3), scope='AvgPool_3a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, (1, 1), scope='Conv2d_3b_1x1')
                net = tf.concat((branch_0, branch_1, branch_2, branch_3), 3)

    return net, end_points


def inception_v3(inputs,
                 num_classes=1000,
                 is_training=True,
                 keep_prob=0.8,
                 predic_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='Inception_v3'):
    """
    实现全局平均池化，Softmax和Auxiliary Logits
        :param inputs: 输入
        :param num_classes: 需要分类的数量 
        :param is_training: False则无BN和dropout
        :param keep_prob: dropout比例
        :param predic_fn: 分类函数
        :param spatial_squeeze: 是否对输出进行squeeze操作，即去除维数为 1 的维度
        :param reuse: 是否会对网络和Variable进行重用
        :param scope: 默认参数环境
        :return: 预测类别，辅助节点字典
    """
    with tf.variable_scope(scope, 'Inception_v3', (inputs, num_classes), reuse=reuse) as scope:

        # 卷积层输出
        with slim.arg_scope((slim.batch_norm, slim.dropout), is_training=is_training):
            net, end_points = inception_v3_base(inputs, scope=scope)

        # 全连接层输出
        with slim.arg_scope((slim.conv2d, slim.max_pool2d, slim.avg_pool2d),
                            stride=1, padding='SAME'):

            # 辅助节点贡献的输出
            aux_logits = end_points['Mixed_6e']
            with tf.variable_scope('Aux_logits'):
                aux_logits = slim.avg_pool2d(aux_logits, (5, 5), stride=3, padding='VALID', scope='AvgPool_1a_5x5')
                aux_logits = slim.conv2d(aux_logits, 128, (1, 1), scope='Conv2d_1b_1x1')
                aux_logits = slim.conv2d(aux_logits, 768, (5, 5), weights_initializer=trunc_normal(0.01),
                                         padding='VALID', scope='Conv2d_1c_5x5')
                aux_logits = slim.conv2d(aux_logits, num_classes, (1, 1), activation_fn=None, normalizer_fn=None,
                                         weights_initializer=trunc_normal(0.001), scope='Conv2d_1d_1x1')
                if spatial_squeeze:
                    """以axis作为索引，移除此索引列表指示的维度中维数为1的维度"""
                    aux_logits = tf.squeeze(aux_logits, (1, 2), name='SpatialSqueeze')

                end_points['Aux_Logits'] = aux_logits

            # 输出
            with tf.variable_scope('Logits'):
                net = slim.avg_pool2d(net, (8, 8), padding='VALID', scope='AvgPool_1a_8x8')
                net = slim.dropout(net, keep_prob, scope='Dropout_1b')
                end_points['Pre_Logits'] = net

                logits = slim.conv2d(net, num_classes, (1, 1), activation_fn=None, normalizer_fn=None,
                                     scope='Conv2d_1c_1x1')
                if spatial_squeeze:
                    logits = tf.squeeze(logits, (1, 2), name='SpatialSqueeze')

                end_points['Logits'] = logits
                end_points['Predic'] = predic_fn(logits, scope='Predic')

    return logits, end_points


if __name__ is '__main__':

    batch_size = 32
    height, weight = 299, 299
    inputs = tf.random_normal((batch_size, height, weight, 3))

    with slim.arg_scope(inception_v3_arg_scope()):
        logits, end_points = inception_v3(inputs, False)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    num_batch = 100
    time_tensorflow_run(sess, logits, 'Forward')
