#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: DreamingWay
Date: 2025/3/6
Description:If using codes, please indicate the source. 
"""

# !/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import yaml
from datetime import datetime
from tqdm import tqdm

# 项目模块导入
from src.environment import EdgeCachingEnv
from src.agent import DDQNAgent
from src.model import DDQN
from src.utils.logger import Logger


def evaluate_policy(
        env: EdgeCachingEnv,
        agent: DDQNAgent,
        num_episodes: int = 10,
        disable_exploration: bool = True
) -> Dict[str, np.ndarray]:
    """
    评估策略在环境中的表现

    参数：
    env: 边缘缓存环境
    agent: 训练好的智能体
    num_episodes: 评估回合数
    disable_exploration: 是否禁用探索（评估时建议开启）

    返回：
    包含评估指标的字典
    """
    metrics = {
        "episode_rewards": [],
        "episode_delays": [],
        "local_hits": [],
        "edge_hits": [],
        "cloud_requests": []
    }

    if disable_exploration:
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0  # 评估时禁用随机探索

    for _ in tqdm(range(num_episodes), desc="Evaluating"):
        state = env.reset()
        done = False
        episode_metrics = {
            "total_reward": 0.0,
            "delays": [],
            "local_hits": 0,
            "edge_hits": 0,
            "cloud_requests": 0
        }

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            # 记录指标
            episode_metrics["total_reward"] += reward
            episode_metrics["delays"].append(info["delay"])

            # 统计请求类型
            if action == 0:
                episode_metrics["local_hits"] += 1
            elif action == 1:
                if info["delay"] == env.delay_weights["edge"]:
                    episode_metrics["edge_hits"] += 1
                else:
                    episode_metrics["cloud_requests"] += 1
            else:
                episode_metrics["cloud_requests"] += 1

            state = next_state

        # 汇总本回合数据
        metrics["episode_rewards"].append(episode_metrics["total_reward"])
        metrics["episode_delays"].append(np.mean(episode_metrics["delays"]))
        metrics["local_hits"].append(episode_metrics["local_hits"])
        metrics["edge_hits"].append(episode_metrics["edge_hits"])
        metrics["cloud_requests"].append(episode_metrics["cloud_requests"])

    if disable_exploration:
        agent.epsilon = original_epsilon  # 恢复原始探索率

    return metrics


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="评估边缘缓存策略")
    parser.add_argument("--model_path", type=str, required=True,
                        help="训练好的模型路径")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="评估回合数")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_results",
                        help="评估结果输出目录")
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载配置文件
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    # 初始化环境
    requests = np.load("data/processed/requests.npy", allow_pickle=True)
    env = EdgeCachingEnv(requests, config_path="configs/default.yaml")

    # 初始化智能体
    agent = DDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=config["agent"]
    )

    # 加载模型权重
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    state_dict = torch.load(model_path, map_location=agent.device)
    agent.policy_net.load_state_dict(state_dict)
    agent.update_target_network()  # 同步目标网络

    # 运行评估
    print(f"\n开始评估模型: {model_path.name}")
    metrics = evaluate_policy(env, agent, num_episodes=args.num_episodes)

    # 计算统计指标
    results = {
        "avg_reward": np.mean(metrics["episode_rewards"]),
        "std_reward": np.std(metrics["episode_rewards"]),
        "avg_delay": np.mean(metrics["episode_delays"]),
        "local_hit_rate": np.sum(metrics["local_hits"]) / np.sum(
            metrics["local_hits"] + metrics["edge_hits"] + metrics["cloud_requests"]),
        "edge_hit_rate": np.sum(metrics["edge_hits"]) / np.sum(
            metrics["local_hits"] + metrics["edge_hits"] + metrics["cloud_requests"]),
        "cloud_request_rate": np.sum(metrics["cloud_requests"]) / np.sum(
            metrics["local_hits"] + metrics["edge_hits"] + metrics["cloud_requests"]),
        "total_requests": int(np.sum(metrics["local_hits"] + metrics["edge_hits"] + metrics["cloud_requests"]))
    }

    # 打印结果
    print("\n评估结果摘要:")
    print(f"- 平均奖励: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"- 平均延迟: {results['avg_delay']:.2f} ms")
    print(f"- 本地命中率: {results['local_hit_rate']:.2%}")
    print(f"- 边缘命中率: {results['edge_hit_rate']:.2%}")
    print(f"- 云请求率: {results['cloud_request_rate']:.2%}")
    print(f"- 总处理请求数: {results['total_requests']}")

    # 保存评估结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"eval_{model_path.stem}_{timestamp}.json"

    with open(result_file, "w") as f:
        json.dump({
            "model_path": str(model_path),
            "config": config,
            "metrics": metrics,
            "summary": results
        }, f, indent=2)

    print(f"\n评估结果已保存至: {result_file}")


if __name__ == "__main__":
    main()

"""
边缘缓存策略评估脚本
用法示例：
python src/scripts/evaluate.py \
    --model_path outputs/models/ddqn_final.pth \
    --num_episodes 10 \
    --output_dir outputs/eval_results
"""
