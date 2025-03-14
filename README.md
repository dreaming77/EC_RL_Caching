# Edge Caching with DDQN

This project implements a Deep Double Q-Network (DDQN) for edge caching in a distributed edge computing environment. The goal is to optimize video content delivery by predicting user requests and caching content at edge servers.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/edge-caching-dqn.git
   cd edge-caching-dqn

2. pip install -r requirements.txt

Quick Start
1. Preprocess the MovieLens 1M dataset:
python src/preprocess.py

2. Train the DDQN model:
python src/train.py

3.Evaluate the trained model:
python src/evaluate.py