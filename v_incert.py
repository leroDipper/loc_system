"""
Diagnostic: Check if match features have real variation or are still placeholders.
"""

import pandas as pd
import numpy as np

# Load the CSV
df = pd.read_csv('results/uncertainty_estimates.csv')

print("="*60)
print("MATCH FEATURE VARIATION CHECK")
print("="*60)

# Check if we have the ratio and n_candidates columns
if 'mean_ratio_test' in df.columns and 'mean_n_candidates' in df.columns:
    print("\n✓ Match statistics columns found in CSV")
    
    ratio_values = df['mean_ratio_test'].values
    n_candidates = df['mean_n_candidates'].values
    
    print(f"\nRatio Test Values:")
    print(f"  Mean:   {ratio_values.mean():.3f}")
    print(f"  Std:    {ratio_values.std():.3f}")
    print(f"  Min:    {ratio_values.min():.3f}")
    print(f"  Max:    {ratio_values.max():.3f}")
    print(f"  Unique: {len(np.unique(ratio_values))}")
    
    print(f"\nNumber of Candidates:")
    print(f"  Mean:   {n_candidates.mean():.1f}")
    print(f"  Std:    {n_candidates.std():.1f}")
    print(f"  Min:    {n_candidates.min():.0f}")
    print(f"  Max:    {n_candidates.max():.0f}")
    print(f"  Unique: {len(np.unique(n_candidates))}")
    
    # Check if these are still placeholders
    if ratio_values.std() < 0.001:
        print("\n❌ WARNING: Ratio test values have NO variation!")
        print("   All values are identical (likely placeholder 0.5)")
    else:
        print(f"\n✓ Ratio test values have variation (std={ratio_values.std():.3f})")
    
    if n_candidates.std() < 0.001:
        print("\n❌ WARNING: Candidate counts have NO variation!")
        print("   All values are identical (likely placeholder 1.0)")
    else:
        print(f"\n✓ Candidate counts have variation (std={n_candidates.std():.1f})")
    
    # Show distribution
    print(f"\nRatio Test Distribution:")
    print(f"  < 0.4:  {(ratio_values < 0.4).sum()} frames ({(ratio_values < 0.4).sum()/len(ratio_values)*100:.1f}%)")
    print(f"  0.4-0.6: {((ratio_values >= 0.4) & (ratio_values < 0.6)).sum()} frames")
    print(f"  0.6-0.8: {((ratio_values >= 0.6) & (ratio_values < 0.8)).sum()} frames")
    print(f"  > 0.8:  {(ratio_values >= 0.8).sum()} frames")
    
    print(f"\nCandidate Count Distribution:")
    print(f"  1:     {(n_candidates == 1).sum()} frames ({(n_candidates == 1).sum()/len(n_candidates)*100:.1f}%)")
    print(f"  2-5:   {((n_candidates >= 2) & (n_candidates <= 5)).sum()} frames")
    print(f"  6-10:  {((n_candidates >= 6) & (n_candidates <= 10)).sum()} frames")
    print(f"  > 10:  {(n_candidates > 10).sum()} frames")
    
else:
    print("\n❌ ERROR: Match statistics columns NOT found in CSV!")
    print(f"   Available columns: {list(df.columns)}")
    print("\n   The uncertainty estimation may not be using real match features.")

print("\n" + "="*60)

# Show sample of data
print("\nSample of first 10 frames:")
if 'mean_ratio_test' in df.columns:
    print(df[['frame', 'mean_confidence', 'mean_ratio_test', 'mean_n_candidates']].head(10).to_string())
else:
    print(df[['frame', 'mean_confidence', 'mean_pixel_uncertainty_px']].head(10).to_string())