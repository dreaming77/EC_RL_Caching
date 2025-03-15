#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: DreamingWay
Date: 2025/3/6
Description:If using codes, please indicate the source. 
"""

# src/agent.py
import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from src.model import DDQN


class DDQNAgent:
    def __init__(self, state_dim: int, action_dim: int, config: dict):
        # 本人当前所使用的设备是mac mini4的mps GPU进行试验，如果使用的是windows系统的GPU，则使用cuda
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("mps")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        # 初始化Q网络
        self.policy_net = DDQN(state_dim, action_dim).to(self.device)
        self.target_net = DDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络不参与训练

        # 优化器和超参数
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config["agent"]['lr'])
        self.gamma = config["agent"]['gamma']
        self.epsilon = config["agent"]['epsilon_max']
        self.epsilon_min = config["agent"]['epsilon_min']
        self.epsilon_decay = config["agent"]['epsilon_decay']
        self.batch_size = config["agent"]['batch_size']

        # 经验回放缓冲区
        self.memory = deque(maxlen=config["agent"]['replay_size'])

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        选择动作（ε-greedy策略）
        Args:
            state: 当前状态向量
            deterministic: 是否确定性选择（评估时使用）
        """
        if not deterministic and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """确保状态数据为numpy数组格式"""
        self.memory.append((
            state.astype(np.float32),  # 强制类型转换
            int(action),
            float(reward),
            next_state.astype(np.float32),
            bool(done)
        ))

    # agent.py
    def update_model(self):
        if len(self.memory) < self.batch_size:
            return

        # 改进采样逻辑
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为PyTorch张量并分配设备
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # 计算当前Q值
        current_q = self.policy_net(states_tensor).gather(1, actions_tensor)

        # 计算目标Q值（Double DQN）
        with torch.no_grad():
            next_actions = self.policy_net(next_states_tensor).argmax(1, keepdim=True)
            next_q = self.target_net(next_states_tensor).gather(1, next_actions)
            target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q

        # 计算损失
        loss = F.mse_loss(current_q, target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 探索率衰减
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        """同步目标网络参数"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path: str):
        """保存模型权重"""
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path: str):
        """加载模型权重"""
        self.policy_net.load_state_dict(torch.load(path))

