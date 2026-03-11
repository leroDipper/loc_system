import torch
import numpy as np

FAILURE_THRESHOLD = 0.20  # 10cm — fixed physical threshold

class LocalisationDataset:
    """
    Holds data for one environment (sequence + keypoint combination).
    Failure defined as pose error > 10cm.
    Class weights computed per environment to handle imbalance.
    """
    def __init__(self, name, X, y):
        self.name = name
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.failure_threshold = FAILURE_THRESHOLD
        self.failure_labels = (self.y > FAILURE_THRESHOLD).float()

        # Per-environment class weights to handle imbalance
        n_total = len(self.failure_labels)
        n_failure = self.failure_labels.sum().item()
        n_success = n_total - n_failure

        # Avoid division by zero in degenerate environments
        if n_failure == 0 or n_success == 0:
            self.pos_weight = torch.tensor(1.0)
        else:
            self.pos_weight = torch.tensor(n_success / n_failure)

    def __len__(self):
        return len(self.y)

    def sample(self, n):
        """Sample n random frames from this environment."""
        idx = torch.randperm(len(self))[:n]
        return self.X[idx], self.y[idx], self.failure_labels[idx]


class MetaLocalisationDataset:
    """
    Holds all environments and supports episode sampling for MAML.
    """
    def __init__(self, env_list):
        self.envs = env_list

    def sample_episode(self):
        """
        Sample one MAML episode.
        Returns: support environments, query environment
        """
        query_idx = np.random.randint(len(self.envs))
        query_env = self.envs[query_idx]
        support_envs = [e for i, e in enumerate(self.envs) if i != query_idx]
        return support_envs, query_env