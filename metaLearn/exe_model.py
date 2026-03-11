from dataLoader import load_all_environments
from net import UncertaintyNet
from train import train_maml, evaluate, compute_metrics
import torch

if __name__ == '__main__':

    print("Loading environments...")
    meta_dataset, test_envs, scaler, features = load_all_environments(base_features=None)

    n_features = len(features)
    print(f"Using {n_features} features")
    model = UncertaintyNet(n_features)

    print(f"\nTraining MAML on {len(meta_dataset.envs)} environments...")
    losses = train_maml(
        model, meta_dataset,
        n_episodes=10000,
        outer_lr=0.001,
        inner_lr=0.01,
        K=1,
        n_support=128,
        log_every=500
    )

    print("\nEvaluating on held-out environment:")
    for env in test_envs:
        print(f"\n  Environment: {env.name} | pos_weight={env.pos_weight:.2f}")
        predictions, uncertainties, p_failure, actuals = evaluate(model, env)
        compute_metrics(predictions, uncertainties, p_failure, actuals, env.failure_threshold)

    torch.save({
        'model_state': model.state_dict(),
        'scaler': scaler,
        'features': features,
        'n_features': n_features,
        'failure_threshold': 0.10,
    }, 'results/maml_uncertainty_model.pt')
    print("\n✓ Saved model to results/maml_uncertainty_model.pt")