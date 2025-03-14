#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: DreamingWay
Date: 2025/3/13
Description:If using codes, please indicate the source.
"""

import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import yaml
from pathlib import Path

def load_config():
    with open("configs/default.yaml") as f:
        return yaml.safe_load(f)

class FeatureEngineer:
    def __init__(self):
        self.config = load_config()
        self.movie_encoder = LabelEncoder()         # 电影ID编码器
        self.popularity_scaler = MinMaxScaler()     # 流行度归一化器
        self.cache_scaler = MinMaxScaler()          # 缓存容量归一化器
        self.movie_embedding = None                 # 电影嵌入层
        self.time_columns = None                    # 时间特征列名缓存

    def fit_transform(self, df):
        """在训练集上拟合特征工程参数"""
        # 验证输入数据完整性
        required_columns = ['movie_id', 'timestamp', 'popularity', 'cache_remaining']
        assert all(col in df.columns for col in required_columns), "Missing required columns"

        # === 电影ID嵌入 ===
        # 将movie_id编码为连续整数
        df['movie_id_encoded'] = self.movie_encoder.fit_transform(df['movie_id'])
        # 初始化嵌入层
        self.movie_embedding = torch.nn.Embedding(
            num_embeddings=len(self.movie_encoder.classes_),
            embedding_dim=self.config['features']['movie_embedding_dim']
        )
        movie_features = self.movie_embedding(torch.LongTensor(df['movie_id_encoded'].values))

        # === 时间特征编码 ===
        # 提取小时和星期几
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour.astype(int)
        df['weekday'] = pd.to_datetime(df['timestamp']).dt.weekday.astype(int)
        # 确保所有类别存在（24小时+7天）
        df['hour'] = pd.Categorical(df['hour'], categories=range(24))
        df['weekday'] = pd.Categorical(df['weekday'], categories=range(7))
        # 生成one-hot编码
        time_dummies = pd.get_dummies(df[['hour', 'weekday']], prefix=['hour', 'weekday'])
        # 缓存时间特征列名
        self.time_columns = time_dummies.columns.tolist()
        time_features = torch.FloatTensor(time_dummies.values)

        # === 归一化处理 ===
        # 流行度归一化
        popularity_norm = torch.FloatTensor(
            self.popularity_scaler.fit_transform(df[['popularity']])
        )
        # 缓存剩余容量归一化
        cache_norm = torch.FloatTensor(
            self.cache_scaler.fit_transform(df[['cache_remaining']])
        )

        # === 特征拼接 ===
        features = torch.cat([
            movie_features,     # 电影嵌入 (32维)
            time_features,       # 时间特征 (24h + 7d = 31维)
            popularity_norm,    # 流行度 (1维)
            cache_norm          # 缓存容量 (1维)
        ], dim=1)

        return features

    def transform(self, df):
        """在测试集/新数据上应用特征转换"""
        # === 电影ID处理 ===
        # 处理未见过的movie_id（映射到unknown类）
        df['movie_id_encoded'] = df['movie_id'].apply(
            lambda x: x if x in self.movie_encoder.classes_ else 'unknown'
        )
        df['movie_id_encoded'] = self.movie_encoder.transform(df['movie_id_encoded'])
        movie_features = self.movie_embedding(torch.LongTensor(df['movie_id_encoded'].values))

        # === 时间特征编码 ===
        # 生成与训练集一致的列
        time_dummies = pd.get_dummies(df[['hour', 'weekday']], prefix=['hour', 'weekday'])
        # 对齐列名
        missing_cols = set(self.time_columns) - set(time_dummies.columns)
        for col in missing_cols:
            time_dummies[col] = 0
        time_dummies = time_dummies[self.time_columns]
        time_features = torch.FloatTensor(time_dummies.values)

        # === 归一化处理 ===
        popularity_norm = torch.FloatTensor(
            self.popularity_scaler.transform(df[['popularity']])
        )
        cache_norm = torch.FloatTensor(
            self.cache_scaler.transform(df[['cache_remaining']])
        )

        # === 特征拼接 ===
        return torch.cat([
            movie_features,
            time_features,
            popularity_norm,
            cache_norm
        ], dim=1)