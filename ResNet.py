import collections
import tensorflow as tf
import tensorflow.contrib.slim as slim

from ALexNet import time_tensorflow_run


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tupel describing a ResNet block.
    Its parts are：
        scope: scope
        unit_fn: 残差学习单元
        args： 列表元素代表一个瓶颈（bottleneck）残差学习单元的三个元素的列表（每个单元包含3个卷积层）
    """


def subsample(inputs, factor, scope=None):
    """
    降采样方法，若factor为1则不做修改直接返回inputs，否则用最大池化进行降采样
        :param inputs: 输入
        :param factor: 降采样因子
        :param scope: scope
        :return: 采样结果输出
    """
    if factor is 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, (1, 1), stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, ksize, stride, scope=None):
    """
    创建卷积层，若stride为1则直接卷积并且Padding=SAME，否则需要对输入进行手工填充0
    等价于：net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')；
    net = subsample(net, factor=stride)
        :param inputs: 输入
        :param num_outputs: 输出通道数
        :param ksize: 核尺寸（int）
        :param stride: 步长
        :param scope: scope
        :return: 卷积输出
    """
    if stride is 1:
        return slim.conv2d(inputs, num_outputs, ksize, scope=scope)
    else:  # 这个补零什么意思？？？
        pad_total = ksize - 1
        pad_beg = pad_total // 2  # 除后向下取整
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, ((0, 0), (pad_beg, pad_end), (pad_beg, pad_end), (0, 0)))
        return slim.conv2d(inputs, num_outputs, ksize, stride, padding='VALID', scope=scope)


@slim.add_arg_scope
def stack_blocks_dense(net, blocks, outputs_collections=None):
    """
    堆叠block的函数
        :param net: 输入
        :param blocks: Block列表
        :param outputs_collections: 收集辅助节点的collection
        :return: 单元输出
    """
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:  # ???

            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i+1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    net = block.unit_fn(net,
                                        depth=unit_depth,
                                        depth_bottleneck=unit_depth_bottleneck,
                                        stride=unit_stride)

            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)  # ???
            """
            def collect_named_outputs(collections, alias, outputs):
                if collections:
                    append_tensor_alias(outputs, alias)
                    ops.add_to_collections(collections, outputs)
                return outputs
            """

    return net


def resnet_arg_scope(is_training=True,
                     weight_decay=1e-4,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    """返回scope"""
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=slim.variance_scaling_initializer(),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, outputs_collections=None, scope=None):
    """
    瓶颈残差单元，先对输入进行BN和预激活，然后分别计算直连部分和卷积部分，最后求和
        :param inputs: 输入
        :param depth: 单元输出通道数
        :param depth_bottleneck: 前两层的输出通道数
        :param stride: 中间层的步长
        :param outputs_collections: 辅助节点collection 
        :param scope: scope
        :return: 单元输出
    """
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')

        # 这是为了使直连的输出与残差单元的输出保持一致，进行求和
        if depth is depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, 1, stride=stride, normalizer_fn=None,
                                   activation_fn=None, scope='shortcut')

        residual = slim.conv2d(preact, depth_bottleneck, 1, stride=1, scope='conv1')
        residual = slim.conv2d(residual, depth_bottleneck, 3, stride, scope='conv2')
        residual = slim.conv2d(residual, depth, 1, stride=1, normalizer_fn=None, activation_fn=None, scope='conv3')

        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)  # ???


def resnet_v2(inputs, blocks, num_classes=None, global_pool=True, include_root_block=True, reuse=None, scope=None):
    """
    根据预先定义好的残差学习模块blocks生成完整的resnet
        :param inputs: 输入
        :param blocks: 残差学习模块列表
        :param num_classes: 输出通道数
        :param global_pool: 是否在最后进行全局平均池化
        :param include_root_block: 是否在最前面进行7x7卷积和最大池化
        :param reuse: 重用
        :param scope: scope
        :return: 输出， 辅助节点dict
    """
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'  # ???
        with slim.arg_scope((slim.conv2d, bottleneck, stack_blocks_dense),
                            outputs_collections=end_points_collection):
            net = inputs

            if include_root_block:
                with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                    net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
                net = slim.max_pool2d(net, 3, 2, scope='pool1')

            net = stack_blocks_dense(net, blocks)
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')

            if global_pool:
                net = tf.reduce_mean(net, (1, 2), keep_dims=True, name='pool5')

            if num_classes is not None:
                net = slim.conv2d(net, num_classes, 1, activation_fn=None, normalizer_fn=None, scope='logits')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)  # ???
            if num_classes is not None:
                end_points['predic'] = slim.softmax(net, scope='predic')

            return net, end_points


# noinspection PyArgumentList
def resnet_v2_50(inputs, num_classes=None, global_pool=True, reuse=None, scope='resent_v2_50'):

    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]

    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)


# noinspection PyArgumentList
def resnet_v2_152(inputs, num_classes=None, global_pool=True, reuse=None, scope='resnet_v2_50'):

    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3),
    ]

    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)


if __name__ == '__main__':
    batch_size = 32
    num_batches = 100
    height, weight = 224, 224
    inputs = tf.random_uniform((batch_size, height, weight, 3))

    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net, end_points = resnet_v2_152(inputs, 1000)

    sess = tf.Session()
    writer = tf.summary.FileWriter('/logs', sess.graph)
    sess.run(tf.global_variables_initializer())
    time_tensorflow_run(sess, net, 'Forward')
