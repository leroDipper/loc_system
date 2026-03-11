import torch
import torch.nn as nn
import numpy as np

class UncertaintyNet(nn.Module):
    """
    Small MLP with three output heads:
    - p_failure: primary output — probability of localization failure (>10cm)
    - mu: auxiliary — predicted error magnitude
    - sigma: auxiliary — predicted uncertainty
    """
    def __init__(self, n_features):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.mu_head      = nn.Linear(32, 1)
        self.sigma_head   = nn.Linear(32, 1)
        self.failure_head = nn.Linear(32, 1)

    def forward(self, x):
        features = self.backbone(x)

        mu        = self.mu_head(features).squeeze(-1)
        sigma     = nn.functional.softplus(self.sigma_head(features)).squeeze(-1) + 1e-6
        p_failure = torch.sigmoid(self.failure_head(features)).squeeze(-1)

        return mu, sigma, p_failure


def loss_fn(mu, sigma, p_failure, y, failure_labels, pos_weight):
    bce = nn.functional.binary_cross_entropy_with_logits(
        torch.logit(p_failure.clamp(1e-6, 1 - 1e-6)),
        failure_labels,
        pos_weight=pos_weight
    )
    return bce