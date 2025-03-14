#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: DreamingWay
Date: 2025/3/13
Description:If using codes, please indicate the source. 
"""

import os
import json
from datetime import datetime
from typing import Dict, Optional
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(
            self,
            log_dir: str = "outputs/logs",
            log_file: str = "training_log.json",
            tensorboard_dir: Optional[str] = None,
            verbose: bool = True
    ):
        """
        强化学习训练日志记录器

        参数：
        log_dir: 日志文件存储目录
        log_file: 日志文件名（JSON格式）
        tensorboard_dir: TensorBoard日志目录（None表示禁用）
        verbose: 是否在控制台打印日志
        """
        # 创建日志目录
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # 初始化文件日志
        self.log_path = os.path.join(log_dir, log_file)
        self._log_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "log_format_version": "1.0"
            },
            "metrics": []
        }

        # 初始化TensorBoard
        self.tensorboard_writer = None
        if tensorboard_dir:
            self.tensorboard_writer = SummaryWriter(tensorboard_dir)

        # 控制台打印配置
        self.verbose = verbose
        self._print_header()

    def _print_header(self):
        """打印控制台日志表头"""
        if self.verbose:
            print("\n{:<10} {:<12} {:<12} {:<12} {:<12}".format(
                "Episode", "Total Reward", "Avg Delay", "Epsilon", "Duration"
            ))
            print("-" * 65)

    def log(self, metrics: Dict, step: Optional[int] = None):
        """
        记录训练指标

        参数：
        metrics: 包含指标的字典，必须包含"episode"字段
        step: 可选的自定义步数（默认使用metrics["episode"]）
        """
        # 添加时间戳
        metrics["timestamp"] = datetime.now().isoformat()

        # 存储到内存
        self._log_data["metrics"].append(metrics)

        # 写入TensorBoard
        if self.tensorboard_writer is not None:
            for key, value in metrics.items():
                if key == "episode":
                    continue
                self.tensorboard_writer.add_scalar(
                    tag=f"train/{key}",
                    scalar_value=value,
                    global_step=metrics["episode"]
                )

        # 控制台打印
        if self.verbose:
            self._print_console(metrics)

    def _print_console(self, metrics: Dict):
        """格式化打印到控制台"""
        print("{:<10} {:<12.2f} {:<12.2f} {:<12.4f} {:<12.2f}".format(
            metrics["episode"],
            metrics.get("total_reward", 0),
            metrics.get("avg_delay", 0),
            metrics.get("epsilon", 0),
            metrics.get("duration", 0)
        ))

    def save(self):
        """将日志保存到磁盘"""
        with open(self.log_path, "w") as f:
            json.dump(self._log_data, f, indent=2)

        if self.verbose:
            print(f"\nLog saved to {self.log_path}")

    def close(self):
        """关闭日志记录器"""
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
        self.save()

    def get_metrics(self) -> list:
        """获取所有记录的指标"""
        return self._log_data["metrics"]

    def get_metric_series(self, metric_name: str) -> list:
        """获取指定指标的时间序列"""
        return [m.get(metric_name, None) for m in self._log_data["metrics"]]


# 单元测试
if __name__ == "__main__":
    # 初始化日志记录器
    logger = Logger(
        log_dir="test_logs",
        tensorboard_dir="test_logs/tensorboard",
        verbose=True
    )

    # 模拟训练日志
    for episode in range(3):
        logger.log({
            "episode": episode,
            "total_reward": np.random.uniform(0, 100),
            "avg_delay": np.random.uniform(10, 50),
            "epsilon": max(0.9 - episode * 0.2, 0.1),
            "duration": np.random.uniform(2, 5)
        })

    # 保存并关闭
    logger.close()

