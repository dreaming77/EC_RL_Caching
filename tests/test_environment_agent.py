import yaml
from pathlib import Path
from src.agent import DDQNAgent
from src.environment import EdgeCachingEnv
import pandas as pd
import numpy as np

from src.feature_engineers import MovieLensFeatureEngineer


def load_config():
    with open("../configs/default.yaml") as f:
        return yaml.safe_load(f)

config = load_config()
processed_data_path = Path(config['data']['processed_path']) / 'processed_requests.parquet'
df = pd.read_parquet(processed_data_path)
# 初始化特征工程器
engineer = MovieLensFeatureEngineer()
#特征转换
features = engineer.fit_transform(df)
# 将 Tensor 转换为 NumPy 数组
features_np = features.to_numpy()

# 创建 DataFrame
feature_names = (['movie_id'] + ['edge_id'] +\
                ['movie_features'] +\
                ['time_features'] +\
                ['popularity_norm'] +\
                ['cache_norm'])

# 将 NumPy 数组转换为 DataFrame
features_df = pd.DataFrame(features_np, columns=feature_names)

# 假设预处理数据已转换为字典列表
requests = features_df.to_dict('records')

# 测试代码片段
env = EdgeCachingEnv(requests, config)
agent = DDQNAgent(config["environment"]["state_dim"], config["environment"]["action_dim"], config)

state = env.reset()
action = agent.select_action(state)
next_state, reward, done, info = env.step(action)

print(f"State Shape: {state.shape}")  # 应输出 (state_dim,)
print(f"Reward: {reward}")            # 应根据延迟正确计算
print(f"Next State Valid: {not np.isnan(next_state).any()}")

agent.store_transition(state, action, reward, next_state, done)
