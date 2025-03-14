#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: DreamingWay
Date: 2025/3/13
Description:If using codes, please indicate the source. 
"""

import unittest
import numpy as np
from unittest.mock import patch
from collections import deque
from src.environment import EdgeCachingEnv


class TestEdgeCachingEnv(unittest.TestCase):
    def setUp(self):
        """初始化测试环境和模拟数据"""
        # 创建模拟请求数据
        self.mock_requests = np.array([
            {'user_edge': 0, 'movie_id': 101, 'popularity_norm': 0.8, 'cache_remaining': 0.2,
             'time_features': [0.5] * 31},
            {'user_edge': 1, 'movie_id': 102, 'popularity_norm': 0.5, 'cache_remaining': 0.5,
             'time_features': [0.3] * 31},
            {'user_edge': 2, 'movie_id': 103, 'popularity_norm': 0.3, 'cache_remaining': 0.8,
             'time_features': [0.1] * 31}
        ], dtype=object)

        # 初始化环境
        self.env = EdgeCachingEnv(
            requests=self.mock_requests,
            num_edges=3,
            config_path="../configs/default.yaml"
        )

    def test_initialization(self):
        """测试环境初始化参数"""
        # 验证动作空间
        self.assertEqual(self.env.action_space.n, 3)

        # 验证状态空间维度
        self.assertEqual(self.env.observation_space.shape, (65,))

        # 验证缓存容量
        self.assertEqual(self.env.cache_capacity, 100)

        # 验证延迟权重
        self.assertEqual(self.env.delay_weights["local"], 1.0)
        self.assertEqual(self.env.delay_weights["edge"], 10.0)
        self.assertEqual(self.env.delay_weights["cloud"], 100.0)

    def test_reset(self):
        """测试环境重置功能"""
        initial_state = self.env.reset()

        # 验证缓存清空
        for cache in self.env.edge_caches.values():
            self.assertEqual(len(cache), 0)

        # 验证当前步数归零
        self.assertEqual(self.env.current_step, 0)

        # 验证状态维度
        self.assertEqual(len(initial_state), 65)

    def test_local_cache_hit(self):
        """测试本地缓存命中场景"""
        self.env.reset()

        # 手动添加电影到缓存
        self.env.edge_caches[0].append(101)

        # 执行动作0（本地缓存）
        state, reward, done, info = self.env.step(0)

        # 验证延迟
        self.assertEqual(info["delay"], 1.0)

        # 验证缓存未变化（LRU顺序更新）
        self.assertIn(101, self.env.edge_caches[0])

    def test_edge_cache_hit(self):
        """测试邻近边缘命中场景"""
        self.env.reset()

        # 在边缘服务器1添加电影
        self.env.edge_caches[1].append(102)

        # 用户来自边缘服务器0，执行动作1
        state, reward, done, info = self.env.step(1)

        # 验证延迟
        self.assertEqual(info["delay"], 10.0)

        # 验证本地缓存更新（假设缓存未满）
        self.assertIn(102, self.env.edge_caches[0])

    def test_cloud_request(self):
        """测试云中心请求场景"""
        self.env.reset()

        # 执行动作2（直接请求云）
        state, reward, done, info = self.env.step(2)

        # 验证延迟
        self.assertEqual(info["delay"], 100.0)

        # 验证缓存未更新
        self.assertEqual(len(self.env.edge_caches[0]), 0)

    def test_cache_replacement_policy(self):
        """测试LRU缓存替换策略"""
        self.env.cache_capacity = 2  # 缩小缓存容量方便测试

        # 填充缓存
        self.env.edge_caches[0].extend([201, 202])

        # 请求新电影并执行缓存动作
        self.env.step(0)  # movie_id=101
        self.env.step(0)  # movie_id=102

        # 验证缓存替换
        self.assertEqual(list(self.env.edge_caches[0]), [202, 101, 102][-2:])

    def test_full_episode_flow(self):
        """测试完整回合流程"""
        self.env.reset()

        total_reward = 0
        done = False

        while not done:
            action = 0  # 始终尝试本地缓存
            state, reward, done, info = self.env.step(action)
            total_reward += reward

        # 验证回合长度
        self.assertEqual(self.env.current_step, len(self.mock_requests))

        # 验证最终奖励合理性
        self.assertLess(total_reward, 0)  # 奖励是负延迟

    def test_boundary_conditions(self):
        """测试边界条件"""
        # 空请求列表测试
        with self.assertRaises(ValueError):
            EdgeCachingEnv(requests=np.array([]))

        # 无效动作测试
        self.env.reset()
        with self.assertRaises(AssertionError):
            self.env.step(4)  # 无效动作


if __name__ == '__main__':
    unittest.main()

# pytest tests/test_environment.py -v