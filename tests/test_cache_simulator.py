#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: DreamingWay
Date: 2025/3/14
Description:If using codes, please indicate the source. 
"""

import unittest
from src.utils.cache_simulator import CacheSimulator


class TestCacheSimulator(unittest.TestCase):
    def test_basic_operations(self):
        cache = CacheSimulator(cache_size=2)

        # 测试添加和存在性检查
        cache.add_item(1)
        self.assertTrue(cache.has_item(1))

        # 测试LRU替换
        cache.add_item(2)
        cache.add_item(3)  # 应移除1
        self.assertFalse(cache.has_item(1))
        self.assertTrue(cache.has_item(2))
        self.assertTrue(cache.has_item(3))

        # 测试访问更新LRU顺序
        cache.has_item(2)  # 将2移动到头部
        cache.add_item(4)  # 应移除3
        self.assertFalse(cache.has_item(3))
        self.assertTrue(cache.has_item(2))


