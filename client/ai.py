import random
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import float32

from client.base import SlaveWeaponType
from client.resp import PacketResp, Character, Block, Item, BuffType, SlaveWeapon

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01  # 学习率
EPSILON = 0.95  # 贪心策略
GAMMA = 0.8  # 奖励衰减
TARGET_REPLACE_ITER = 100  # 目标更新频率
MEMORY_CAPACITY = 5000  # 记忆容量
ACTION_SHAPE = (9, 3, 6)
N_ACTIONS = (7 + 3) * 3 + 6  # 操作数 (方向6 + 无操作+隐身3) + (Move + 攻击 * 2) + 排序(6) = 36
N_STATES = 16 * 16 * 1 + 1 + 17 * 2  # 状态数 = 地图信息 16*16 + 当前击杀 + 两个角色的17个属性 = 291


# N_STATES * N_ACTIONS = 10476

class DQN(object):  # 强化神经网络
    def __init__(self):
        self.eval_net = Net()  # 定义两个网络：评估网络 & 目标网络
        # 记忆单元为 当前状态 行为 奖励 操作后状态
        # 单元数为 状态维度 * 2 + 行为维度 + 奖励维度(1)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()  # 均方误差损失函数

    def choose_action(self, x):  # 传入当前的 State，计算行为
        # 只输入一个样本
        if np.random.uniform() < 0.9:  # 90% 概率使用评估网络的结果
            actions_value = self.eval_net.forward(x)  # 使用评估网络获取行为
            # actions_value = [a1,b1,c1,d1]
            # actions_value = [0..8,9..11,12..17]
            direction_move = torch.max(actions_value[:7], 0).indices.data.item()  # 7行的最大值索引
            sneaky_move = torch.max(actions_value[7:10], 0).indices.data.item()  # 3行的最大值索引
            direction_master = torch.max(actions_value[10:17], 0).indices.data.item()  # 7行的最大值索引
            sneaky_master = torch.max(actions_value[17:20], 0).indices.data.item()  # 3行的最大值索引
            direction_slave = torch.max(actions_value[20:27], 0).indices.data.item()  # 7行的最大值索引
            sneaky_slave = torch.max(actions_value[27:30], 0).indices.data.item()  # 3行的最大值索引
            rank = torch.max(actions_value[30:], 0).indices.data.item()  # 6行的最大值索引 len = 36
            # 返回argmax索引
        else:  # 随机
            direction_move = random.randint(0, 5)
            sneaky_move = random.randint(0, 1)
            direction_master = random.randint(0, 5)
            sneaky_master = random.randint(0, 1)
            direction_slave = random.randint(0, 5)
            sneaky_slave = random.randint(0, 1)
            rank = random.randint(0, 4)
        print(
            f"{direction_move}, {sneaky_move}, {direction_master}, {sneaky_master}, {direction_slave}, {sneaky_slave}, {rank}")
        return direction_move, sneaky_move, direction_master, sneaky_master, direction_slave, sneaky_slave, rank

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 单GPU或者CPU
        # print("[Device]", device)
        self.fc1 = nn.Linear(N_STATES, N_STATES * N_ACTIONS)  # .to(device)  # 传入网络 状态数 -> 状态 * 行动
        self.fc1.weight.data.normal_(0, 0.1)  # 随机初始化权重
        self.out = nn.Linear(N_ACTIONS * N_STATES, N_ACTIONS)  # .to(device)  # 输出网络 状态 * 行动 -> 输出Action数
        self.out.weight.data.normal_(0, 0.1)  # 随机初始化权重

    def forward(self, x):  # 前向传播
        x = self.fc1(x)  # 第一层映射
        x = F.relu(x)  # 激活函数 线性整流函数 f(x) = max(0,x)
        actions_value = self.out(x)  # 输出映射 [p1,p2,p3,p4,...,pn]
        return actions_value


class AI(object):
    def __init__(self, id):
        self.last_resp = None
        self.last_state = None
        self.last_action = None
        self.times = 0
        self.ep_r = 0
        self.start = time.time()
        self.dqn = DQN()
        try:
            print("读取模型中...")
            self.dqn.eval_net = torch.load("eval-" + id + ".pth")
            print("读取模型成功!")
        except Exception as _:
            print("读取模型失败!")

    def get_action(self, resp):  # 获取Action
        state = state_trans(resp)
        action = self.dqn.choose_action(state)
        # direction_move = torch.max(actions_value[:7], 0).indices.data.item()  # 7行的最大值索引
        #             sneaky_move = torch.max(actions_value[7:10], 0).indices.data.item()  # 3行的最大值索引
        #             direction_master = torch.max(actions_value[10:17], 0).indices.data.item()  # 7行的最大值索引
        #             sneaky_master = torch.max(actions_value[17:20], 0).indices.data.item()  # 3行的最大值索引
        #             direction_slave = torch.max(actions_value[20:27], 0).indices.data.item()  # 7行的最大值索引
        #             sneaky_slave = torch.max(actions_value[27:30], 0).indices.data.item()  # 3行的最大值索引
        #             rank = torch.max(actions_value[30:], 0).indices.data.item()  # 6行的最大值索引 len = 36
        last_action = np.zeros(36, dtype=int)
        last_action[action[0]] = 1
        last_action[7 + action[1]] = 1
        last_action[10 + action[2]] = 1
        last_action[17 + action[3]] = 1
        last_action[20 + action[4]] = 1
        last_action[27 + action[5]] = 1
        last_action[30 + action[6]] = 1
        self.last_action = last_action
        self.last_state = state
        self.last_resp = resp
        return action

    def done(self, episode, score):
        t = time.time() - self.start
        print('轮数: ', episode,
              '| 次数: ', round(self.times, 2),
              '| 奖励: ', round(self.ep_r, 2),
              '| 结果: ', round(score, 4),
              '| 时间: ', round(t, 2), 's'
              )

    def reward(self):
        try:
            return self.last_resp.data.kill * 10 + self.last_resp.data.score - self.ep_r
        except Exception as e:
            return 0.0


def state_trans(state: PacketResp):
    me = state.data.characters[0]
    other = Character()
    for block in state.data.map.blocks:
        for obj in block.objs:
            if isinstance(obj, Character):
                assert isinstance(obj, Character)
                if obj.characterID != me.characterID:
                    other = obj
                    break
    # 16 * 16 +  1 + (5 + 6 + 3 + 3) * 2 = 35
    tup = block_map(state.data.map.blocks) + [state.data.kill,
                                              me.x, me.y, me.direction.value, me.hp, me.moveCD,
                                              me.moveCDLeft, me.isAlive, me.isSneaky, me.isGod, me.rebornTimeLeft,
                                              me.godTimeLeft,
                                              me.masterWeapon.weaponType.value, me.masterWeapon.attackCD,
                                              me.masterWeapon.attackCDLeft,
                                              me.slaveWeapon.weaponType.value, me.slaveWeapon.attackCD,
                                              me.slaveWeapon.attackCDLeft,
                                              other.x, other.y, other.direction.value, other.hp, other.moveCD,
                                              other.moveCDLeft, other.isAlive, other.isSneaky,
                                              other.isGod, other.rebornTimeLeft, other.godTimeLeft,
                                              other.masterWeapon.weaponType.value, other.masterWeapon.attackCD,
                                              other.masterWeapon.attackCDLeft,
                                              other.slaveWeapon.weaponType.value, other.slaveWeapon.attackCD,
                                              other.slaveWeapon.attackCDLeft]
    return torch.tensor(tup, dtype=float32)


def block_map(blocks: List[Block]):
    li = []
    for i in range(16 * 16):
        if len(blocks) > i:
            block = blocks[i]
            v = 0
            if len(block.objs) > 0:
                obj = block.objs[-1]
                if obj is Item:
                    assert isinstance(obj, Item)
                    if obj.buffType == BuffType.BuffHp:
                        v = 1
                    else:
                        v = 2
                elif obj is SlaveWeapon:
                    assert isinstance(obj, SlaveWeapon)
                    if obj.weaponType == SlaveWeaponType.Kiwi:
                        v = 3
                    else:
                        v = 4
            if block.valid:
                li.append(block.color.value + v * 10)
            else:
                li.append(-1)
        else:
            li.append(0)
    return li
