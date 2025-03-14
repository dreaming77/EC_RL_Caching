#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: DreamingWay
Date: 2025/3/13
Description:If using codes, please indicate the source. 
"""

from torch.utils.data import Dataset


class EdgeCacheDataset(Dataset):
    def __init__(self, features, labels=None):
        """
        :param features: 特征向量 (torch.Tensor)
        :param labels: 标签 (torch.Tensor)
        """
        self.features = features
        self.labels = labels  # 可扩展为预测目标（如缓存决策）

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]

