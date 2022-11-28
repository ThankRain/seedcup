import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01  # 学习率
EPSILON = 0.95  # 贪心策略
GAMMA = 0.8  # 奖励衰减
TARGET_REPLACE_ITER = 100  # 目标更新频率
MEMORY_CAPACITY = 2000
ACTION_SHAPE = (9, 3, 6)
N_ACTIONS = 9 + 3 + 6  # 操作数 (方向6+隐身3) * (Move + 攻击 * 2) * 排序(6) = 162
N_STATES = 16 * 16 * 1 + 1 + 13 * 2  # 状态数 = 地图信息 16*16 + 当前击杀 + 两个角色的13个属性 = 283


# N_STATES * N_ACTIONS = 45846

class DQN(object):  # 强化神经网络
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()  # 定义两个网络：评估网络 & 目标网络
        self.learn_step_counter = 0  # 用于目标网络延迟更新
        self.memory_counter = 0  # 存储计数器
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # 初始化记忆全为 0
        # 记忆单元为 当前状态 行为 奖励 操作后状态
        # 单元数为 状态维度 * 2 + 行为维度 + 奖励维度(1)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # todo Adam 优化器
        self.loss_func = nn.MSELoss()  # 均方误差损失函数

    def choose_action(self, x):  # 传入当前的 State，计算行为
        x = torch.unsqueeze(torch.FloatTensor(x), 0)  # 解包
        # 只输入一个样本
        if np.random.uniform() < EPSILON:  # 90% 概率使用评估网络的结果
            actions_value = self.eval_net.forward(x)  # 使用评估网络获取行为
            # actions_value = [a1,b1,c1,d1]
            # actions_value = [a1,b1,c1,d1]
            action = torch.max(actions_value, 1)[1].data.numpy()  # 每行的最大值索引
            action = action[0]  # if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
            # 返回argmax索引
        else:  # 随机
            action = np.random.randint(0, N_ACTIONS)
            action = action.reshape(ACTION_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        # print('[state]:',s)
        # print('[action,reward]:',[a, r])
        # print('[stated]:',s_)
        transition = np.hstack((s, [a, r], s_))
        # 用新内存替换旧内存
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # 目标参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 样本批次转换
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t体验中的行动
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, N_STATES * N_ACTIONS)  # 传入网络 状态数 -> 状态 * 行动
        self.fc1.weight.data.normal_(0, 0.1)  # 随机初始化权重
        self.out = nn.Linear(N_ACTIONS * N_STATES, N_ACTIONS)  # 输出网络 状态 * 行动 -> 输出Action数
        self.out.weight.data.normal_(0, 0.1)  # 随机初始化权重

    def forward(self, x):  # 前向传播
        x = self.fc1(x)  # 第一层映射
        x = F.relu(x)  # 激活函数 线性整流函数 f(x) = max(0,x)
        actions_value = self.out(x)  # 输出映射
        return actions_value


class AI(object):
    def __init__(self):
        self.times = 0
        self.ep_r = 0
        self.start = time.time()
        self.dqn = DQN()
        try:
            print("读取模型中...")
            self.dqn.target_net = torch.load("target.pth")
            self.dqn.eval_net = torch.load("eval.pth")
            print("读取模型成功!")
        except Exception as _:
            print("读取模型失败!")

    def get_action(self, state):  # 获取Action
        self.dqn.choose_action(state)

    def resp(self, state, action, reward, state_):  # 存储响应
        self.dqn.store_transition(state, action, reward, state_)
        self.times += 1
        self.ep_r += reward
        if self.dqn.memory_counter > MEMORY_CAPACITY:
            self.dqn.learn()

    def done(self, episode, score):
        t = time.time() - self.start
        print('轮数: ', episode,
              '| 次数: ', round(self.times, 2),
              '| 奖励: ', round(self.ep_r, 2),
              '| 结果: ', round(score, 4),
              '| 时间: ', round(t, 2), 's'
              )

    def kill(self):
        print("保存模型中...")
        torch.save(self.dqn.target_net, "target.pth")
        torch.save(self.dqn.eval_net, "eval.pth")
        print("保存成功，程序已退出")
