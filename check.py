import joblib
model = joblib.load('results/bayesian_model.joblib')

# Check if model has intercepts (alpha)
if 'alpha' in model:
    print("Dataset intercepts (baseline predictions):")
    for i, name in enumerate(model['dataset_names']):
        # Unscale the intercept
        alpha_unscaled = model['alpha'][i] * model['y_std'] + model['y_mean']
        print(f"  {name}: {alpha_unscaled*100:.2f} cm")