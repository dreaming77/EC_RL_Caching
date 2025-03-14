#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: DreamingWay
Date: 2025/3/6
Description:If using codes, please indicate the source. 
"""

import random
import numpy as np
from collections import deque
import torch
from model import DDQN
import torch.nn.functional as F


class DDQNAgent:
    def __init__(self, state_dim: int, action_dim: int, config: dict):
        # 本人当前所使用的设备是mac mini4的mps GPU进行试验，如果使用的是windows系统的GPU，则使用cuda
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("mps")

        # 初始化Q网络
        self.policy_net = DDQN(state_dim, action_dim).to(self.device)
        self.target_net = DDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # 优化器和经验回放
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config["lr"])
        self.memory = deque(maxlen=config["replay_size"])

        # 超参数
        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon_max"]
        self.epsilon_min = config["epsilon_min"]
        self.epsilon_decay = config["epsilon_decay"]

    def select_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 3)  # 随机选择动作
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_model(self):
        if len(self.memory) < self.batch_size:
            return

        # 从经验回放中采样
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为Tensor
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 计算目标Q值
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # 计算当前Q值
        current_q = self.policy_net(states).gather(1, actions)

        # 计算损失
        loss = F.mse_loss(current_q, target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

