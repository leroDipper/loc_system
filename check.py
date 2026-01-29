import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/feature_lifecycles.csv")
survivors = df[df["survived"] == True]

print("Reprojection error statistics (survivors only):")
print(survivors["reprojection_error"].describe())

plt.hist(survivors["reprojection_error"], bins=50)
plt.xlabel("Reprojection Error (pixels)")
plt.ylabel("Count")
plt.title("Reprojection Error Distribution (Inliers Only)")
plt.show()