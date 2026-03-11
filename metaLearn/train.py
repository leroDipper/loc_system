import torch
import torch.nn as nn
from net import UncertaintyNet, loss_fn
from dataObjects import MetaLocalisationDataset
from metaObjects import maml_episode
import numpy as np


def train_maml(model, meta_dataset, n_episodes=10000,
               outer_lr=0.001, inner_lr=0.01, K=1,
               n_support=64, log_every=500):

    meta_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
    losses = []

    for episode in range(n_episodes):
        support_envs, query_env = meta_dataset.sample_episode()

        query_loss = maml_episode(
            model, support_envs, query_env,
            inner_lr=inner_lr, K=K,
            n_support=n_support
        )

        meta_optimizer.zero_grad()
        query_loss.backward()
        meta_optimizer.step()

        losses.append(query_loss.item())

        if (episode + 1) % log_every == 0:
            mean_loss = np.mean(losses[-log_every:])
            print(f"Episode {episode+1}/{n_episodes} — Mean query loss: {mean_loss:.4f}")

    return losses


def evaluate(model, env, n_samples=None):
    model.eval()
    with torch.no_grad():
        X = env.X if n_samples is None else env.X[:n_samples]
        y = env.y if n_samples is None else env.y[:n_samples]

        mu, sigma, p_failure = model(X)

    predictions  = mu.numpy()
    uncertainties = sigma.numpy()
    p_failure    = p_failure.numpy()
    actuals      = y.numpy()

    print(f"\nPrediction diagnostics:")
    print(f"  Mean predicted error: {predictions.mean()*100:.2f}cm")
    print(f"  Mean actual error:    {actuals.mean()*100:.2f}cm")
    print(f"  Mean p_failure:       {p_failure.mean():.3f}")

    return predictions, uncertainties, p_failure, actuals


def compute_metrics(predictions, uncertainties, p_failure, actuals, failure_threshold):
    from sklearn.metrics import (r2_score, f1_score, precision_score,
                                  recall_score, roc_auc_score)

    failure_labels = (actuals > failure_threshold).astype(float)
    predicted_failure = (p_failure > 0.5).astype(float)

    # Primary: failure detection metrics
    f1        = f1_score(failure_labels, predicted_failure, zero_division=0)
    precision = precision_score(failure_labels, predicted_failure, zero_division=0)
    recall    = recall_score(failure_labels, predicted_failure, zero_division=0)

    try:
        auc = roc_auc_score(failure_labels, p_failure)
    except ValueError:
        auc = float('nan')

    accuracy = np.mean(failure_labels == predicted_failure)

    # Secondary: error magnitude metrics
    correlation = np.corrcoef(actuals, predictions)[0, 1]
    r2  = r2_score(actuals, predictions)
    mae = np.mean(np.abs(actuals - predictions))

    print(f"  --- Failure Detection (primary) ---")
    print(f"  AUC:       {auc:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  Accuracy:  {accuracy*100:.1f}%")
    print(f"  --- Error Magnitude (auxiliary) ---")
    print(f"  Correlation: {correlation:.3f}")
    print(f"  R^2:         {r2:.3f}")
    print(f"  MAE:         {mae*100:.2f}cm")

    return f1, auc, precision, recall, accuracy, correlation, r2, mae