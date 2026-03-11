import torch
import torch.nn as nn
from net import UncertaintyNet, loss_fn
from dataObjects import MetaLocalisationDataset


def maml_episode(model, support_envs, query_env,
                 inner_lr=0.01, K=1, n_support=64):
    """
    One MAML episode:
    1. Combine support environments
    2. Take K gradient steps on support data (inner loop)
    3. Evaluate on query environment (outer loop)
    Returns: query loss for meta-update
    """

    # ========== INNER LOOP ==========
    fast_params = {name: param.clone() for name, param in model.named_parameters()}

    for _ in range(K):
        support_loss = torch.tensor(0.0)
        for env in support_envs:
            X, y, failure_labels = env.sample(n_support)
            mu, sigma, p_failure = functional_forward(model, fast_params, X)
            support_loss = support_loss + loss_fn(mu, sigma, p_failure, y, failure_labels,
                                                   pos_weight=env.pos_weight)
        support_loss = support_loss / len(support_envs)

        grads = torch.autograd.grad(support_loss, fast_params.values(), 
                             create_graph=True, allow_unused=True)

        fast_params = {
            name: param - inner_lr * grad if grad is not None else param
            for (name, param), grad in zip(fast_params.items(), grads)
}

    # ========== OUTER LOOP ==========
    X_q, y_q, failure_labels_q = query_env.sample(n_support)
    mu_q, sigma_q, p_failure_q = functional_forward(model, fast_params, X_q)
    query_loss = loss_fn(mu_q, sigma_q, p_failure_q, y_q, failure_labels_q,
                         pos_weight=query_env.pos_weight)

    return query_loss


def functional_forward(model, params, x):
    """
    Forward pass using custom params instead of model.parameters().
    Needed for MAML inner loop to stay in the computation graph.
    """
    x = nn.functional.linear(x, params['backbone.0.weight'], params['backbone.0.bias'])
    x = nn.functional.relu(x)
    x = nn.functional.linear(x, params['backbone.2.weight'], params['backbone.2.bias'])
    x = nn.functional.relu(x)

    features = x

    mu = nn.functional.linear(features, params['mu_head.weight'], params['mu_head.bias']).squeeze(-1)
    sigma = nn.functional.softplus(
        nn.functional.linear(features, params['sigma_head.weight'], params['sigma_head.bias'])
    ).squeeze(-1) + 1e-6
    p_failure = torch.sigmoid(
        nn.functional.linear(features, params['failure_head.weight'], params['failure_head.bias'])
    ).squeeze(-1)

    return mu, sigma, p_failure