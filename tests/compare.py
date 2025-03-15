# src/environment.py
class EdgeCachingEnv(gym.Env):
    def __init__(self, features_path: str, metadata_path: str, config: dict):
        # 加载特征向量和元数据
        features_data = torch.load(features_path)
        self.features = features_data['features']
        self.metadata = pd.read_parquet(metadata_path)

        # 组合为完整的请求序列
        self.requests = [
            {
                **self.metadata.iloc[i].to_dict(),
                'features': self.features[i]
            }
            for i in range(len(self.metadata))
        ]