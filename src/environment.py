#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: DreamingWay
Date: 2025/3/6
Description:If using codes, please indicate the source. 
"""

import gym
import numpy as np
import yaml
from gym import spaces
from collections import defaultdict, deque
from typing import Dict, List
from .utils.cache_simulator import MultiEdgeCacheManager


class EdgeCachingEnv(gym.Env):
    def __init__(self, requests: np.ndarray, num_edges: int = 3, cache_size=100, config_path: str = "../configs/default.yaml"):
        super().__init__()

        # 加载配置
        self.config = self._load_config(config_path)
        self.num_edges = num_edges
        self.cache_manager = MultiEdgeCacheManager(num_edges, cache_size)
        self.requests = requests  # 预处理后的请求序列

        # 定义动作空间：0-本地缓存，1-邻近边缘，2-云中心
        self.action_space = spaces.Discrete(3)

        # 状态空间维度：[流行度, 缓存剩余容量, 时间特征...]
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.config["state_dim"],))

        # 初始化参数
        self.current_step = 0
        self.delay_weights = self.config["delay_weights"]  # 各层延迟权重
        self.cache_capacity = self.config["cache_capacity"]

        # 初始化边缘服务器缓存（使用LRU策略）
        self.edge_caches: Dict[int, deque] = {
            edge_id: deque(maxlen=self.cache_capacity)
            for edge_id in range(self.num_edges)
        }

    def _load_config(self, path: str) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)

    def reset(self):
        """重置环境状态"""
        self.current_step = 0
        for edge_id in self.edge_caches:
            self.edge_caches[edge_id].clear()
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        current_request = self.requests[self.current_step]
        return np.concatenate([
            current_request["popularity_norm"],  # 归一化流行度
            current_request["cache_remaining"],  # 缓存剩余容量
            current_request["time_features"]  # 时间特征
        ])

    def _calculate_delay(self, action: int, user_edge: int, movie_id: int) -> float:
        """根据动作计算延迟"""
        # 本地缓存命中
        if action == 0 and movie_id in self.edge_caches[user_edge]:
            return self.delay_weights["local"]

        # 邻近边缘查询
        elif action == 1:
            for edge_id in self.edge_caches:
                if edge_id != user_edge and movie_id in self.edge_caches[edge_id]:
                    return self.delay_weights["edge"]
            # 邻近未命中，转云中心
            return self.delay_weights["cloud"]

        # 直接请求云中心
        else:
            return self.delay_weights["cloud"]

    def _update_cache(self, edge_id: int, movie_id: int):
        """更新缓存（LRU策略）"""
        if movie_id in self.edge_caches[edge_id]:
            # 移动到队列头部表示最近使用
            self.edge_caches[edge_id].remove(movie_id)
        self.edge_caches[edge_id].appendleft(movie_id)

    def step(self, action: int):
        # 获取当前请求信息
        current_request = self.requests[self.current_step]
        user_edge = current_request["user_edge"]  # 用户所属边缘服务器
        movie_id = current_request["movie_id"]

        # 计算延迟
        delay = self._calculate_delay(action, user_edge, movie_id)

        # 检查本地缓存
        local_hit = self.cache_manager.check_cache(user_edge, movie_id)

        # 根据动作执行逻辑
        if action == 0 and local_hit:  # 本地命中...
            pass
        elif action == 1:
            # 查询其他边缘服务器...
            for edge_id in self.cache_manager.caches:
                if edge_id != user_edge:
                    if self.cache_manager.check_cache(edge_id, movie_id):
                        # 邻近边缘命中...
                        break
        else:  # 请求云中心...
            pass

        # 更新缓存（仅当动作0/1且缓存未满时）
        if action in [0, 1] and len(self.edge_caches[user_edge]) < self.cache_capacity:
            self.cache_manager.update_cache(user_edge, movie_id)

        # 计算奖励（负延迟）
        reward = -delay

        # 状态转移
        self.current_step += 1
        done = self.current_step >= len(self.requests) - 1

        return self._get_state(), reward, done, {"delay": delay}

