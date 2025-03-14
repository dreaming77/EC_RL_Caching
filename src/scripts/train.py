#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: DreamingWay
Date: 2025/3/6
Description:If using codes, please indicate the source. 
"""

import yaml
import numpy as np
from tqdm import tqdm
from src.environment import EdgeCachingEnv
from src.agent import DDQNAgent
from src.utils.logger import Logger
import torch


def train():
    # 加载配置
    with open("../../configs/default.yaml") as f:
        config = yaml.safe_load(f)

    # 初始化环境和智能体
    requests = np.load("../../data/processed/requests.npy", allow_pickle=True)
    env = EdgeCachingEnv(requests, config_path="../../configs/default.yaml")
    agent = DDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=config["agent"]
    )
    logger = Logger(config["log_dir"])

    # 训练循环
    for episode in tqdm(range(config["num_episodes"])):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)

            # 更新模型
            agent.update_model()

            state = next_state
            total_reward += reward

        # 更新目标网络
        if episode % config["target_update"] == 0:
            agent.update_target_network()

        # 记录日志
        logger.log({
            "episode": episode,
            "total_reward": total_reward,
            "epsilon": agent.epsilon,
            "avg_delay": np.mean(env.delay_history)
        })

        # 保存模型
        if episode % config["save_interval"] == 0:
            torch.save(agent.policy_net.state_dict(),
                       f"../../outputs/models/ddqn_episode{episode}.pth")


if __name__ == "__main__":
    train()

