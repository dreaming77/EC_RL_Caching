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
    def __init__(self, requests: np.ndarray, config: dict):
        super().__init__()

        # 加载配置
        self.config = config
        self.num_edges = config["environment"]["num_edges"]
        self.cache_capacity = config["environment"]["cache_capacity"]
        self.cache_manager = MultiEdgeCacheManager(self.num_edges, self.cache_capacity)
        self.requests = requests  # 预处理后的请求序列

        # 定义动作空间：0-本地缓存，1-邻近边缘，2-云中心
        self.action_space = spaces.Discrete(3)

        # 状态空间维度：[流行度, 缓存剩余容量, 时间特征...]
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.config["environment"]["state_dim"],),
            dtype=np.float32
        )

        # 初始化参数
        self.current_step = 0
        self.delay_weights = self.config["environment"]["delay_weights"]  # 各层延迟权重
        self.cache_capacity = self.config["environment"]["cache_capacity"]

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

    # environment.py
    def _get_state(self) -> np.ndarray:
        """确保字段名称与预处理数据完全一致"""
        current_request = self.requests[self.current_step]

        return np.concatenate([
            [current_request["popularity_norm"]],  # 电影流行度（需归一化）
            [current_request["cache_norm"]],  # 缓存剩余容量（0-1）
            current_request["time_features"]  # 时间特征（one-hot向量）
        ]).astype(np.float32)

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
        current_request = self.requests[self.current_step]
        user_edge = current_request["edge_id"]  # 使用预处理字段名
        movie_id = current_request["movie_id"]

        # 计算延迟
        delay = self._calculate_delay(action, user_edge, movie_id)

        # === 根据动作执行缓存更新 ===
        if action == 0:  # 本地缓存动作
            # 如果缓存未命中且容量未满，则添加内容
            if not self.cache_manager.check_cache(user_edge, movie_id):
                if len(self.cache_manager.caches[user_edge].cache_queue) < self.cache_capacity:
                    self.cache_manager.update_cache(user_edge, movie_id)

        elif action == 1:  # 邻近边缘查询
            # 仅在邻近命中时更新本地缓存（可选策略）
            found = False
            for edge_id in self.cache_manager.caches:
                if edge_id != user_edge and self.cache_manager.check_cache(edge_id, movie_id):
                    found = True
                    break
            if found and len(self.cache_manager.caches[user_edge].cache_queue) < self.cache_capacity:
                self.cache_manager.update_cache(user_edge, movie_id)

        else:  # 请求云中心...
            pass

        # 更新缓存（仅当动作0/1且缓存未满时）
        if action in [0, 1] and len(self.edge_caches[user_edge]) < self.cache_capacity:
            self.cache_manager.update_cache(user_edge, movie_id)

        # === 返回完整信息 ===
        next_state = self._get_state()
        reward = -delay  # 奖励为负延迟
        done = self.current_step >= len(self.requests) - 1
        info = {"delay": delay, "cache_hit": (action == 0)}  # 记录附加信息

        self.current_step += 1  # 必须在返回前递增步骤

        return next_state, reward, done, info

