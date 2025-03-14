#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: DreamingWay
Date: 2025/3/13
Description:If using codes, please indicate the source. 
"""


from collections import deque
from typing import Any, Dict, List
import numpy as np


class CacheSimulator:
    """边缘服务器缓存模拟器（基于LRU策略）"""

    def __init__(self, cache_size: int = 100):
        """
        初始化缓存模拟器
        Args:
            cache_size (int): 缓存容量（可存储的内容数量）
        """
        self.cache_size = cache_size
        self.cache_queue = deque(maxlen=cache_size)  # 维护缓存内容的LRU队列
        self.cache_set = set()  # 用于快速存在性检查

        # 统计指标
        self.hits = 0
        self.misses = 0
        self.total_requests = 0

    def has_item(self, item_id: Any) -> bool:
        """
        检查内容是否在缓存中，并更新LRU状态
        Args:
            item_id: 内容唯一标识（如电影ID）
        Returns:
            bool: 是否存在
        """
        self.total_requests += 1
        if item_id in self.cache_set:
            # 将访问的内容移动到队列头部（表示最近使用）
            self.cache_queue.remove(item_id)
            self.cache_queue.appendleft(item_id)
            self.hits += 1
            return True
        else:
            self.misses += 1
            return False

    def add_item(self, item_id: Any) -> None:
        """
        向缓存中添加内容（遵循LRU替换策略）
        Args:
            item_id: 要添加的内容ID
        """
        if item_id in self.cache_set:
            return  # 已存在则不重复添加

        if len(self.cache_queue) >= self.cache_size:
            # 移除最久未使用的内容
            removed_item = self.cache_queue.pop()
            self.cache_set.remove(removed_item)

        self.cache_queue.appendleft(item_id)
        self.cache_set.add(item_id)

    def get_cache_remaining_ratio(self) -> float:
        """获取缓存剩余容量比例 (剩余槽位/总容量)"""
        return (self.cache_size - len(self.cache_queue)) / self.cache_size

    def get_hit_rate(self) -> float:
        """计算当前命中率"""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    def reset_statistics(self) -> None:
        """重置统计计数器"""
        self.hits = 0
        self.misses = 0
        self.total_requests = 0

    def reset_cache(self) -> None:
        """完全清空缓存"""
        self.cache_queue.clear()
        self.cache_set.clear()
        self.reset_statistics()

    @property
    def current_cache(self) -> List[Any]:
        """获取当前缓存内容列表（按LRU顺序，最近使用的在前）"""
        return list(self.cache_queue)


class MultiEdgeCacheManager:
    """多边缘服务器缓存管理器"""

    def __init__(self, num_edges: int = 3, cache_size: int = 100):
        """
        Args:
            num_edges (int): 边缘服务器数量
            cache_size (int): 每个边缘服务器的缓存容量
        """
        self.caches = {
            edge_id: CacheSimulator(cache_size)
            for edge_id in range(num_edges)
        }

    def check_cache(self, edge_id: int, item_id: Any) -> bool:
        """查询指定边缘服务器的缓存"""
        return self.caches[edge_id].has_item(item_id)

    def update_cache(self, edge_id: int, item_id: Any) -> None:
        """更新指定边缘服务器的缓存"""
        self.caches[edge_id].add_item(item_id)

    def get_all_cache_states(self) -> Dict[int, List[Any]]:
        """获取所有边缘服务器的缓存内容"""
        return {
            edge_id: cache.current_cache
            for edge_id, cache in self.caches.items()
        }

    def reset_all(self) -> None:
        """重置所有缓存和统计"""
        for cache in self.caches.values():
            cache.reset_cache()
            cache.reset_statistics()
