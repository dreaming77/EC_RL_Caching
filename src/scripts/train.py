# src/scripts/train.py
import yaml
import numpy as np
from tqdm import tqdm
import torch
from pathlib import Path
from src.environment import EdgeCachingEnv
from src.agent import DDQNAgent
from src.dataset import ReplayBuffer
from src.feature_engineers import FeatureEngineer
from src.utils.logger import Logger
import pandas as pd


def train():
    """
    经验回放缓冲区：
        使用ReplayBuffer类管理交互数据
        支持随机采样打破数据相关性
    """
    # 加载配置
    with open("../configs/default.yaml") as f:
        config = yaml.safe_load(f)

    # 初始化环境
    processed_data_path = Path(config['data']['processed_path']) / 'processed_requests.parquet'
    df = pd.read_parquet(processed_data_path)
    # 初始化特征工程器
    engineer = FeatureEngineer()
    #特征转换
    features = engineer.fit_transform(df)
    # 假设预处理数据已转换为字典列表
    requests = features.to_dict('records')
    env = EdgeCachingEnv(requests, config=config)

    # 初始化智能体和经验回放缓冲区
    agent = DDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=config['agent']
    )
    replay_buffer = ReplayBuffer(capacity=config['agent']['replay_size'])
    logger = Logger(config['log_dir'])

    # 训练循环
    for episode in tqdm(range(config['training']['num_episodes'])):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # 选择动作
            action = agent.select_action(state)

            # 执行动作并观察环境
            next_state, reward, done, info = env.step(action)

            # 存储经验
            replay_buffer.add(state, action, reward, next_state, done)

            # 更新模型
            if len(replay_buffer) >= config['training']['batch_size']:
                batch = replay_buffer.sample_batch(config['training']['batch_size'])
                agent.update_model(batch)

            state = next_state
            episode_reward += reward

        # 更新目标网络
        if episode % config['training']['target_update_freq'] == 0:
            agent.update_target_network()

        # 记录日志
        logger.log({
            'episode': episode,
            'reward': episode_reward,
            'epsilon': agent.epsilon,
            'avg_delay': info.get('avg_delay', 0)
        })

        # 保存模型
        if episode % config['training']['save_interval'] == 0:
            torch.save(agent.policy_net.state_dict(),
                       f"outputs/models/ddqn_episode{episode}.pth")


if __name__ == '__main__':
    train()