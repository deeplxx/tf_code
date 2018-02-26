import numpy as np
import tensorflow as tf
import gym


def test():
    """随机测试"""
    env = gym.make('CartPole-v0')  # 创建一个环境
    env.reset()  # 初始化环境

    random_episodes = 0  # 测试次数
    reward_sum = 0  # 每次的总奖励

    while random_episodes < 10:
        env.render()  # 渲染每一帧任务图像
        observ, reward, done, _ = env.step(np.random.randint(0, 2))  # 环境状态（观察），奖励，结束标记，info诊断信息
        reward_sum += reward
        if done:
            random_episodes += 1
            print('Reward for this episodes was:', reward_sum)
            reward_sum = 0
            env.reset()


def discount_rewards(r, gamma):
    """估算每个Action所对应的折扣reward（从后向前）,越靠前的价值越大
        
    Args:
        r: A array,表示每个Action对应的reward
        gamma: 折扣率

    Returns: Action对应的价值

    """
    discounted_r = np.zeros_like(r)
    running_add = 0  # Action的价值（潜在+直接reward）
    for t in reversed(range(r.size)):  # 从后向前
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


# noinspection PyUnusedLocal
def inference(hidden_size=50, batch_size=25, lr=0.1, d=4, gamma=0.99):
    """策略网络
    
    Args:
        hidden_size:隐层节点数 
        batch_size:batch_size 
        lr: lr
        d: 环境状态的维度
        gamma: 折扣

    Returns:

    """
    observ_ = tf.placeholder(tf.float32, (None, d), name='input_x')  # 状态，即输入
    y_ = tf.placeholder(tf.float32, (None, 1), name='input_y')  # 人工设置的虚拟label，对应Action（Action为1则label为0）
    advantages = tf.placeholder(tf.float32, name='reward_signal')  # Action对应的价值

    w1 = tf.get_variable('w1', (d, hidden_size), tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    h1 = tf.nn.relu(tf.matmul(observ_, w1))
    w2 = tf.get_variable('w2', (hidden_size, 1), tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    score = tf.matmul(h1, w2)
    probability = tf.nn.sigmoid(score)
    tf.summary.histogram('w1', w1)
    tf.summary.histogram('w2', w2)

    loss = -tf.reduce_mean(tf.log(y_ * (y_ - probability) + (1 - y_) * (y_ + probability)) * advantages)
    """当前Action对应的概率的对数 * 对应的价值。优化后会让advantage大的Action对应的概率更大"""

    optimizer = tf.train.AdamOptimizer(lr)
    w1_grad = tf.placeholder(tf.float32, name='batch_grad1')
    w2_grad = tf.placeholder(tf.float32, name='batch_grad2')
    batch_grad = (w1_grad, w2_grad)
    tvars = tf.trainable_variables()
    update_grad = optimizer.apply_gradients(zip(batch_grad, tvars))  # 更新权值
    new_grad = tf.gradients(loss, tvars)

    # run_bench
    xs, ys, rs = [], [], []  # 输入，label，每一步的奖励
    reward_sum = 0  # batch的reward总和
    episode_id = 1
    total_episodes = 1000
    env = gym.make('CartPole-v0')  # 创建一个环境

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('/logs', tf.get_default_graph())
        merge = tf.summary.merge_all()
        is_render = False  # 渲染标志
        sess.run(tf.global_variables_initializer())
        observ = env.reset()
        grad_buffer = sess.run(tvars)  # 存储参数梯度的缓冲器，初始化为0
        for ix, grad in enumerate(grad_buffer):
            grad_buffer[ix] = grad * 0

        # sub_ = 0  # 记录上一次的avg_reward
        while episode_id <= total_episodes:

            if (reward_sum / batch_size) > 100 or (is_render is True):
                env.render()
                is_render = True

            x = np.reshape(observ, (1, d))
            prob = sess.run(probability, feed_dict={observ_: x})
            action = 1 if np.random.uniform() < prob else 0  # prob是action取1的概率

            xs.append(x)
            y = 1 - action
            ys.append(y)
            observ, reward, done, info = env.step(action)  # 做个动作并返回状态
            reward_sum += reward
            rs.append(reward)

            if done:
                episode_id += 1
                epx = np.vstack(xs)  # 纵向合并
                epy = np.vstack(ys)
                epr = np.vstack(rs)
                xs, ys, rs = [], [], []

                # 标准化epr
                epr_norm = discount_rewards(epr, gamma)
                epr_norm -= np.mean(epr_norm)
                epr_norm /= np.std(epr_norm)

                t_grad = sess.run(new_grad, feed_dict={observ_: epx, y_: epy, advantages: epr_norm})
                for ix, grad in enumerate(t_grad):
                    grad_buffer[ix] += grad

                # 每batch_size个样本更新一次
                if episode_id % batch_size == 0:
                    sess.run(update_grad, feed_dict={w1_grad: grad_buffer[0], w2_grad: grad_buffer[1]})
                    merge_ = sess.run(merge, feed_dict={observ_: x, y_: epy, advantages: epr_norm})
                    writer.add_summary(merge_, episode_id)

                    for ix, grad in enumerate(grad_buffer):  # 将gradbuffer清空以备下次使用
                        grad_buffer[ix] = grad * 0

                    avg_reward = reward_sum / batch_size
                    print('Average reward for episode %d: %f.' % (episode_id, avg_reward))

                    # if avg_reward - sub_ < 10:
                    #     print('Task solved in', episode_id)
                    # sub_ = avg_reward
                    if avg_reward > 200:
                        print('task solved in', episode_id)
                        break

                    reward_sum = 0

                observ = env.reset()
        writer.close()


if __name__ == '__main__':
    inference()
