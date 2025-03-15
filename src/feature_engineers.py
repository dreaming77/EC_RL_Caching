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
    with open("../configs/default.yaml") as f:
        return yaml.safe_load(f)

class MovieLensFeatureEngineer:
    """
    电影ID连续编码：
        使用LabelEncoder将原始movie_id转换为连续整数，解决ID不连续问题
        在transform中处理未知ID（映射到'unknown'类）

    时间特征完整性：
        使用pd.Categorical确保覆盖所有小时（0-23）和星期几（0-6）
        缓存训练集的时间特征列名，测试时严格对齐

    归一化独立性：
        为popularity和cache_remaining分别创建归一化器
        在transform中使用训练集的归一化参数

    健壮性增强：
        添加输入数据完整性检查
        处理时间戳类型转换
        增加transform方法用于推理阶段
    """

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
        required_columns = ['movie_id', 'edge_id', 'timestamp', 'popularity', 'cache_remaining']
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

        df['movie_features'] = movie_features.tolist()
        df['time_features'] = time_features.tolist()
        df['popularity_norm'] = popularity_norm
        df['cache_norm'] = cache_norm

        # 返回特征向量和原始数据（或保存到不同文件）
        return df[['movie_id', 'edge_id', 'movie_features', 'time_features', 'popularity_norm', 'cache_norm']]

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

        df['movie_features'] = movie_features.tolist()
        df['time_features'] = time_features.tolist()
        df['popularity_norm'] = popularity_norm
        df['cache_norm'] = cache_norm

        # 返回特征向量和原始数据（或保存到不同文件）
        return df[['movie_id', 'edge_id', 'movie_features', 'time_features', 'popularity_norm', 'cache_norm']]

    # 或者将元数据与特征分开保存
    def save_features(self, features, metadata, path):
        torch.save({'features': features, 'metadata': metadata}, path)


if __name__ == '__main__':

    def load_config():
        with open("../configs/default.yaml") as f:
            return yaml.safe_load(f)


    config = load_config()
    processed_path = Path(config['data']['processed_path'])
    models_path = Path(config['data']['models_path'])

    # 加载预处理后的数据
    df = pd.read_parquet(processed_path / 'processed_requests.parquet')

    # 初始化特征工程器
    engineer = MovieLensFeatureEngineer()

    # 训练集特征转换
    train_features = engineer.fit_transform(df)

    # 保存特征工程参数（用于后续推理）
    torch.save({
        'movie_encoder': engineer.movie_encoder,
        'time_columns': engineer.time_columns,
        'popularity_scaler': engineer.popularity_scaler,
        'cache_scaler': engineer.cache_scaler,
        'movie_embedding': engineer.movie_embedding.state_dict(),
    }, models_path / 'feature_engineer.pth')


"""
    **步骤一：处理电影ID嵌入**
    
    首先，需要将`movie_id`映射到连续的整数索引。可以使用`sklearn`的`LabelEncoder`将原始的电影ID转换为从0开始的连续整数。
    这可以确保嵌入层的输入是有效的索引。
    
    ```python
    
    from sklearn.preprocessing import LabelEncoder
    
    # 创建电影ID的标签编码器
    
    movie_encoder = LabelEncoder()
    
    df['movie_id_encoded'] = movie_encoder.fit_transform(df['movie_id'])
    
    num_movies = len(movie_encoder.classes_)
    
    ```
    
    这样，`movie_id_encoded`将是从0开始的连续整数，适合作为嵌入层的输入。
    
    **步骤二：时间特征处理**
    
    对于时间特征，`hour`的范围是0-23，`weekday`是0-6。使用`pd.get_dummies`时，可能需要确保所有可能的值都被覆盖，尤其是在训练和测试数据分割时。
    例如，如果某小时在训练集中没有出现，测试集可能会有新的小时值，这会导致特征维度不一致。
    
    解决方法是在生成独热编码时，预先指定所有可能的列。例如，对于小时，确保有24列，无论数据中是否包含所有小时值。同样，星期几确保有7列。
    
    ```python
    
    # 为小时创建所有可能的列（0-23）
    
    hours = pd.DataFrame({'hour': range(24)})
    
    hour_dummies = pd.get_dummies(hours['hour'], prefix='hour')
    
    # 为星期几创建所有可能的列（0-6）
    
    weekdays = pd.DataFrame({'weekday': range(7)})
    
    weekday_dummies = pd.get_dummies(weekdays['weekday'], prefix='weekday')
    
    # 合并到原始数据，填充0
    
    df = df.merge(hour_dummies, on='hour', how='left').fillna(0)
    
    df = df.merge(weekday_dummies, on='weekday', how='left').fillna(0)
    
    ```
    
    但这种方法可能较为繁琐。另一种方法是使用`pd.Categorical`指定所有可能的类别，确保生成的独热编码列完整。
    
    ```python
    
    df['hour'] = pd.Categorical(df['hour'], categories=range(24))
    
    df['weekday'] = pd.Categorical(df['weekday'], categories=range(7))
    
    time_features = pd.get_dummies(df[['hour', 'weekday']], columns=['hour', 'weekday'])
    
    ```
    
    这样，无论数据中是否存在所有的小时或星期几，独热编码都会生成24+7=31列。
    
    **步骤三：归一化处理**
    
    当前代码对`popularity`和`cache_remaining`分别进行归一化，这可能导致它们的归一化范围不一致。
    如果这两个特征的量纲差异较大，可能需要分别归一化，但用户的需求可能更倾向于统一处理。
    需要根据具体场景决定，但原问题中用户可能希望每个特征独立归一化到[0,1]范围，因此当前做法可能正确，但需要注意是否应该使用同一个`MinMaxScaler`实例还是不同的实例。
    根据代码，`self.scaler`被重复使用，但每次调用`fit_transform`会重新拟合，这会导致之前的拟合参数被覆盖，应该为每个特征使用独立的归一化器。
    
    例如：
    
    ```python
    
    self.popularity_scaler = MinMaxScaler()
    
    self.cache_scaler = MinMaxScaler()
    
    popularity_normalized = torch.FloatTensor(self.popularity_scaler.fit_transform(df[['popularity']]))
    
    cache_capacity = torch.FloatTensor(self.cache_scaler.fit_transform(df[['cache_remaining']]))
    
    ```
    
    或者在`fit_transform`中分开处理。
    
    **步骤四：代码整合与优化**
    
    结合以上分析，修改`FeatureEngineer`类：
    
    1. 使用`LabelEncoder`处理电影ID。
    
    2. 使用`pd.Categorical`确保时间特征的完整性。
    
    3. 分别对`popularity`和`cache_remaining`进行归一化。
    
    4. 添加错误处理和数据类型检查。
    
    此外，还需要确保在`fit_transform`方法中正确处理数据，例如检查时间列是否为`datetime`类型，处理可能的缺失值等。
    
    **最终修改后的代码**
    
    ```python
    
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
    
    self.movie_encoder = LabelEncoder()
    
    self.popularity_scaler = MinMaxScaler()
    
    self.cache_scaler = MinMaxScaler()
    
    self.movie_embedding = None
    
    self.time_columns = None  # 保存时间特征列名
    
    def fit_transform(self, df):
    
    # 检查必要列是否存在
    
    required_columns = ['movie_id', 'timestamp', 'popularity', 'cache_remaining']
    
    assert all(col in df.columns for col in required_columns), "Missing required columns"
    
    # 处理电影ID嵌入
    
    df['movie_id_encoded'] = self.movie_encoder.fit_transform(df['movie_id'])
    
    num_movies = len(self.movie_encoder.classes_)
    
    self.movie_embedding = torch.nn.Embedding(
    
    num_embeddings=num_movies,
    
    embedding_dim=self.config['features']['movie_embedding_dim']
    
    )
    
    movie_ids = torch.LongTensor(df['movie_id_encoded'].values)
    
    movie_features = self.movie_embedding(movie_ids)
    
    # 处理时间特征
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # 确保转换为datetime
    
    df['hour'] = df['timestamp'].dt.hour
    
    df['weekday'] = df['timestamp'].dt.weekday
    
    # 确保所有小时和星期几类别存在
    
    df['hour'] = pd.Categorical(df['hour'], categories=range(24))
    
    df['weekday'] = pd.Categorical(df['weekday'], categories=range(7))
    
    time_features = pd.get_dummies(df[['hour', 'weekday']], columns=['hour', 'weekday'])
    
    # 保存时间列名以便后续转换
    
    self.time_columns = time_features.columns.tolist()
    
    time_features = torch.FloatTensor(time_features.values)
    
    # 归一化处理
    
    popularity_normalized = torch.FloatTensor(
    
    self.popularity_scaler.fit_transform(df[['popularity']])
    
    )
    
    cache_capacity = torch.FloatTensor(
    
    self.cache_scaler.fit_transform(df[['cache_remaining']])
    
    )
    
    # 合并特征
    
    features = torch.cat([
    
    movie_features,           # 电影嵌入 (32维)
    
    time_features,            # 时间特征 (24h + 7d = 31维)
    
    popularity_normalized,    # 流行度 (1维)
    
    cache_capacity            # 缓存剩余容量 (1维)
    
    ], dim=1)
    
    return features
    
    def transform(self, df):
    
    # 用于测试集或新数据的转换
    # 处理电影ID（使用训练集的编码）
    
    df['movie_id_encoded'] = self.movie_encoder.transform(df['movie_id'])
    
    # 生成电影嵌入
    
    movie_ids = torch.LongTensor(df['movie_id_encoded'].values)
    
    movie_features = self.movie_embedding(movie_ids)
    
    # 处理时间特征
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df['hour'] = df['timestamp'].dt.hour
    
    df['weekday'] = df['timestamp'].dt.weekday
    
    # 确保与训练集相同的列
    
    time_features = pd.DataFrame(columns=self.time_columns)
    
    temp_dummies = pd.get_dummies(df[['hour', 'weekday']], columns=['hour', 'weekday'])
    
    # 对齐列
    
    time_features = time_features.reindex(columns=self.time_columns, fill_value=0)
    
    time_features.update(temp_dummies)
    
    time_features = torch.FloatTensor(time_features.values)
    
    # 归一化
    
    popularity_normalized = torch.FloatTensor(
    
    self.popularity_scaler.transform(df[['popularity']])
    
    )
    
    cache_capacity = torch.FloatTensor(
    
    self.cache_scaler.transform(df[['cache_remaining']])
    
    )
    
    # 合并特征
    
    features = torch.cat([
    
    movie_features,
    
    time_features,
    
    popularity_normalized,
    
    cache_capacity
    
    ], dim=1)
    
    return features
    
    ```
    
    **代码说明**
    
    1. **电影ID处理**：
    
    - 使用`LabelEncoder`将原始`movie_id`转换为连续整数，避免嵌入层索引越界。
    
    - 在`fit_transform`中拟合编码器，在`transform`中使用相同的编码。
    
    2. **时间特征处理**：
    
    - 将`hour`和`weekday`转换为分类变量，确保独热编码生成所有可能的列（24小时+7天）。
    
    - 保存训练集的列名，以便在测试时对齐特征。
    
    3. **归一化处理**：
    
    - 使用独立的`MinMaxScaler`实例分别对`popularity`和`cache_remaining`进行归一化。
    
    - 在`transform`方法中使用训练集的归一化参数，保证一致性。
    
    4. **健壮性增强**：
    
    - 添加了必要列的检查，确保输入数据包含所需字段。
    
    - 处理了时间列的转换，避免因数据类型错误导致的异常。
    
    **使用示例**
    
    ```python
    
    # 加载预处理后的数据
    
    import pandas as pd
    
    df = pd.read_parquet('path/processed/processed_requests.parquet')
    
    # 初始化特征工程器
    
    engineer = FeatureEngineer()
    
    # 训练集特征转换
    
    train_features = engineer.fit_transform(df)
    
    # 测试集转换（假设有新数据）
    
    test_df = pd.read_parquet('path/processed/test_requests.parquet')
    
    test_features = engineer.transform(test_df)
    
    ```
    
    **可能遇到的问题及解决方案**
    
    - **电影ID在测试集中出现新值**：由于`LabelEncoder`在训练时拟合，测试集中的新ID会导致错误。
    需要在预处理阶段处理未知ID，例如将其映射到一个特殊的“未知”类别，或在数据预处理时过滤掉未出现过的电影ID。
    
    - **时间特征列不一致**：通过保存训练集的列名并在测试时对齐，可以避免特征维度不一致的问题。
    
    - **归一化参数保存**：确保在部署模型时，保存并加载`popularity_scaler`和`cache_scaler`的归一化参数，以保持一致性。
    
    通过以上修改，`feature_engineers.py`能够正确处理预处理后的数据，生成符合要求的特征向量，并与模型训练流程无缝集成。
    
"""