#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: DreamingWay
Date: 2025/3/6
Description:If using codes, please indicate the source. 
"""

import pandas as pd
import yaml
import numpy as np
from pathlib import Path
from src.utils.cache_simulator import MultiEdgeCacheManager


def load_config():
    with open("../configs/default.yaml") as f:
        return yaml.safe_load(f)


def preprocess_requests():
    """
    状态空间特征化：
        电影流行度：通过统计评分次数并归一化
        用户请求：编码为电影ID嵌入 + 时间特征
        缓存容量：动态模拟缓存剩余槽位比例
    """
    config = load_config()
    raw_path = Path(config['data']['raw_path'])

    # 加载原始数据
    ratings = pd.read_csv(
        raw_path / 'ratings.dat',
        sep='::',
        names=['user_id', 'movie_id', 'rating', 'timestamp',],
        engine='python'
    )

    # 按时间戳排序生成请求序列
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    ratings_sorted = ratings.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)

    # 计算电影流行度（总评分次数）
    movie_popularity = ratings.groupby('movie_id').size().reset_index(name='popularity')
    ratings_sorted = pd.merge(ratings_sorted, movie_popularity, on='movie_id')

    # 初始化多边缘缓存管理器
    num_edges = config['environment']['num_edges']
    cache_size = config['environment']['cache_capacity']
    cache_manager = MultiEdgeCacheManager(num_edges=num_edges, cache_size=cache_size)

    # 将用户分配到边缘服务器（简单取模）
    ratings_sorted['edge_id'] = ratings_sorted['user_id'] % num_edges

    # 模拟缓存状态并生成缓存剩余容量特征
    cache_remaining_list = []
    for idx, row in ratings_sorted.iterrows():
        edge_id = int(row['edge_id'])
        movie_id = row['movie_id']

        # 检查缓存并更新
        if not cache_manager.check_cache(edge_id, movie_id):
            cache_manager.update_cache(edge_id, movie_id)

        # 记录当前缓存剩余比例
        cache_remaining = cache_manager.caches[edge_id].get_cache_remaining_ratio()
        cache_remaining_list.append(cache_remaining)

    # 添加缓存剩余容量特征
    ratings_sorted['cache_remaining'] = cache_remaining_list

    # 保存预处理数据
    processed_path = Path(config['data']['processed_path'])
    processed_path.mkdir(parents=True, exist_ok=True)
    ratings_sorted.to_parquet(processed_path / 'processed_requests.parquet')


if __name__ == '__main__':
    preprocess_requests()

# 以下是AI生成的数据预处理脚本
    # # Step 1: 预处理数据
    # preprocess_requests()
    #
    # # Step 2: 加载预处理后的数据
    # import pandas as pd
    # processed_df = pd.read_parquet('../data/processed/processed_requests.parquet')
    #
    # # Step 3: 特征工程
    # engineer = FeatureEngineer()
    # features = engineer.fit_transform(processed_df)
    #
    # # Step 4: 创建Dataset
    # from src.dataset import EdgeCacheDataset
    #
    # dataset = EdgeCacheDataset(features)


    """"
        用户之前已经完成了关于边缘缓存预测的代码项目，现在在处理数据预处理时遇到了问题。具体来说，他们的preprocess.py文件中有一个部分需要模拟边缘服务器的缓存状态，以便生成缓存剩余容量作为状态空间的一个特征，但目前这部分代码不完整。用户已经补充了cache_simulator.py文件，但现在发现preprocess.py无法与之对应，需要修改preprocess.py以便正确处理数据集，并生成包含缓存剩余容量的特征。
    
    首先，我需要理解用户的需求。他们希望预处理MovieLens 1M数据集，生成请求序列，并在预处理过程中模拟边缘服务器的缓存状态。具体来说，每个请求需要包含缓存剩余容量的特征，这个特征是通过模拟边缘服务器的缓存使用情况来动态计算的。用户之前已经写了一个CacheSimulator类，现在需要在预处理阶段使用这个类来跟踪每个边缘服务器的缓存状态，并为每个请求生成缓存剩余容量的比例。
    
    接下来，我需要查看用户提供的preprocess.py代码。代码中有一个函数preprocess_requests()，负责加载数据、排序请求、计算电影流行度，但模拟缓存状态的部分不完整。在注释“模拟边缘服务器缓存状态（是否有空槽位）”下，用户留空了代码，需要补充。
    
    用户的问题是如何修改preprocess.py，使其能够利用CacheSimulator类来生成缓存剩余容量特征。这可能涉及到为每个用户分配一个边缘服务器，然后按时间顺序处理每个请求，模拟缓存的状态变化，并记录每个请求时的缓存剩余容量。
    
    首先，用户需要确定如何将用户分配到不同的边缘服务器。比如，可以根据用户ID的模运算分配到三个边缘服务器中的一个。例如，user_id % num_edges。
    
    然后，需要为每个边缘服务器维护一个CacheSimulator实例。当处理每个请求时，根据用户所属的边缘服务器，检查请求的电影是否在缓存中，并更新缓存。同时，记录缓存剩余容量比例作为特征。
    
    需要注意的是，预处理是离线的，需要按时间顺序处理每个请求，并为每个请求生成对应的缓存状态特征。这可能需要在数据预处理阶段按时间顺序遍历所有请求，逐个处理，并实时更新缓存状态。
    
    具体步骤可能如下：
    
    1. 初始化多个CacheSimulator实例，每个对应一个边缘服务器。
    
    2. 将用户分配到不同的边缘服务器。
    
    3. 按时间顺序遍历每个请求。
    
    4. 对于每个请求，获取对应的边缘服务器。
    
    5. 检查电影是否在缓存中，并更新缓存。
    
    6. 记录当前缓存的剩余容量比例作为特征。
    
    7. 将处理后的数据保存到文件。
    
    现在，用户需要将这部分逻辑补充到preprocess.py中的ratings_sorted = 这一行之后。也就是在计算电影流行度之后，开始模拟缓存状态，生成缓存剩余容量特征。
    
    可能的实现步骤：
    
    - 创建一个MultiEdgeCacheManager实例，管理多个边缘服务器的缓存。
    
    - 遍历预处理后的请求数据（按时间排序）。
    
    - 对于每个请求，确定用户所在的边缘服务器。
    
    - 检查该边缘服务器中是否有该电影，如果没有则添加（根据LRU策略）。
    
    - 记录当前缓存剩余容量比例。
    
    - 将该比例添加到数据集中。
    
    此外，还需要注意，CacheSimulator的add_item方法只有在动作需要更新缓存时才会调用。但在预处理阶段，可能需要模拟真实的缓存行为，比如每次请求时如果缓存未命中，就将该电影加入缓存。这可能需要根据问题定义来决定。
    
    例如，在预处理阶段，假设每次用户请求电影时，如果本地缓存未命中，边缘服务器会将电影缓存（如果缓存未满）。这样，预处理时需要模拟这种缓存策略，从而为每个请求生成缓存剩余容量特征。
    
    因此，在预处理过程中，需要按时间顺序处理每个请求，模拟每个边缘服务器的缓存状态，记录每个请求时的缓存剩余容量。这需要遍历整个数据集，逐个处理每个请求，并更新对应的缓存状态。
    
    现在，用户需要将这部分逻辑写入preprocess.py中的ratings_sorted = 之后的部分。可能涉及遍历DataFrame的每一行，并为每一行生成缓存剩余容量特征。
    
    然而，遍历DataFrame的行效率较低，特别是对于大型数据集如MovieLens 1M。但考虑到预处理是一次性的，可能可以接受。不过，也可以寻找向量化操作的方法来提高效率。
    
    综上所述，修改preprocess.py的步骤包括：
    
    1. 初始化MultiEdgeCacheManager或为每个边缘服务器创建CacheSimulator实例。
    
    2. 按时间顺序处理每个请求。
    
    3. 对于每个请求，确定所属的边缘服务器。
    
    4. 检查缓存并更新。
    
    5. 记录缓存剩余容量比例。
    
    6. 将结果保存到DataFrame中。
    
    现在，我需要将上述逻辑转化为具体的代码，补充到preprocess.py的对应位置。
    """