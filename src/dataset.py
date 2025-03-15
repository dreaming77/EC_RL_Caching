#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: DreamingWay
Date: 2025/3/13
Description:If using codes, please indicate the source. 
"""

# src/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
from collections import deque
import random


class ReplayBuffer(Dataset):
    """强化学习经验回放缓冲区"""

    def __init__(self, capacity: int = 100000):
        super().__init__()
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        state, action, reward, next_state, done = self.buffer[idx]
        return (
            torch.FloatTensor(state),
            torch.LongTensor([action]),
            torch.FloatTensor([reward]),
            torch.FloatTensor(next_state),
            torch.FloatTensor([done])
        )

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """添加单条经验到缓冲区"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample_batch(self, batch_size: int) -> tuple:
        """随机采样批次数据"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )

"""
    用户的数据预处理（preprocess.py）已经生成了包含必要特征的`processed_requests.parquet`文件，特征工程（feature_engineers.py）将这些特征转换为适合模型输入的张量。现在需要将处理好的数据加载到PyTorch的Dataset和DataLoader中，以便在训练过程中高效地读取和批处理数据。
    
    但这里可能存在一个问题：强化学习通常需要在线与环境交互，而传统的监督学习数据集是静态的。用户可能需要在环境中逐步处理每个请求，并根据当前状态选择动作，获得奖励，并转移到下一个状态。因此，可能需要调整数据集的使用方式，使其适合强化学习的训练循环。
    
    用户提供的`EdgeCacheDataset`类目前接受特征和标签，但强化学习中通常没有固定的标签，而是通过环境交互获得奖励。因此，可能需要重新考虑如何构建数据集。不过，在离线强化学习（Offline RL）的情况下，可以使用预先收集的经验数据进行训练。但根据用户之前的代码，他们似乎是在线学习，每个step都与环境交互。
    
    因此，可能需要将Dataset用于存储经验回放缓冲区中的数据，而不是原始的特征数据。经验回放缓冲区存储的是(state, action, reward, next_state, done)这样的元组，供模型训练时随机采样。
    
    但是用户提供的`dataset.py`目前的结构是处理特征和标签，可能不符合经验回放的需求。因此，可能需要修改Dataset类，或者另外创建一个ReplayBuffer类来管理经验存储。
    
    另外，用户之前的`agent.py`中已经有一个`memory`属性作为经验回放缓冲区，使用的是Python的deque。可能需要在训练过程中，将每个step的经验存储在ReplayBuffer中，然后通过DataLoader来随机采样批次数据。不过，通常强化学习中直接使用自定义的采样方法，而不是PyTorch的DataLoader。
    
    因此，用户可能需要将经验回放缓冲区的实现整合到Agent类中，而不是依赖于传统的Dataset和DataLoader。这可能意味着现有的`EdgeCacheDataset`类可能需要调整用途，或者用户需要另外实现一个ReplayBuffer类。
    
    此外，用户可能需要定义训练循环，包括与环境交互收集经验、更新模型参数、调整探索率等步骤。需要确保状态、动作、奖励、下一个状态和完成标志正确传递，并且模型能够根据这些数据进行学习。
    
    总结下来，用户下一步应该：
    
    1. 创建或调整Dataset/ReplayBuffer来存储和管理经验数据。
    
    2. 实现训练循环，与环境交互，收集经验并更新模型。
    
    3. 定期评估模型性能，保存检查点，并记录训练指标。
    
    4. 进行超参数调优和模型验证。
    
    在代码实现上，可能需要修改现有的`dataset.py`或创建新的`replay_buffer.py`，并编写训练脚本`train.py`，同时调整`agent.py`中的经验存储和采样逻辑。此外，需要确保环境类`environment.py`正确返回状态和奖励，并与Agent类协同工作。
    
    需要注意的潜在问题包括：经验回放缓冲区的大小和采样方式，探索与利用的平衡（如ε-greedy策略），模型更新的频率，以及如何将预处理后的特征数据正确输入到模型中。此外，需要确保所有组件的配置一致，例如状态空间的维度与模型输入的匹配。


"""