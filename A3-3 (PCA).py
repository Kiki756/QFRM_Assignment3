import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import colors as mcolors

# Load your data
data = pd.read_csv('portfolio2.csv')
data.set_index('Date', inplace=True)
data.astype(float)
data = data.pct_change().dropna()

# Check the first few rows to ensure data loaded correctly
print(data.head())

returns = data

# Standardize the data
returns_standardized = (returns - returns.mean()) / returns.std()

# Perform PCA
pca = PCA()
pca.fit(returns_standardized)

# Variance explained by each component
explained_variance_ratio = pca.explained_variance_ratio_

# Cumulative variance explained
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Principal components (loadings)
loadings = pca.components_

# Report the explained variance ratios and cumulative variance ratios
print("Explained Variance Ratio:")
print(explained_variance_ratio)
print("\nCumulative Variance Ratio:")
print(cumulative_variance_ratio)

# Report the loadings for the first few principal components
n_components_to_show = 10
for i in range(n_components_to_show):
    print(f"\nPrincipal Component {i+1} Loadings:")
    print(loadings[i])

# Additional diagnostics can be added based on your requirements
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='-')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.show()

# Eigenvalues
eigenvalues = pca.explained_variance_
print("\nEigenvalues:")
print(eigenvalues)

# Biplot (for the first two principal components)

def biplot(score, coeff, labels=None):
    plt.figure(figsize=(10, 6))
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley, c='gray', alpha=0.5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

biplot(pca.transform(returns_standardized)[:, :2], np.transpose(pca.components_[:2, :]))
plt.title('Biplot of PC1 and PC2')
plt.show()

# Interpretation of Principal Components (based on loadings)
for i in range(n_components_to_show):
    print(f"\nPrincipal Component {i+1} Interpretation:")
    sorted_indices = np.argsort(loadings[i])[::-1]  # Sort indices in descending order of loadings
    top_variables = sorted_indices[:3]  # Consider top 3 variables with highest loadings
    for j, idx in enumerate(top_variables):
        print(f"Variable {j+1}: {returns.columns[idx]} (Loading: {loadings[i][idx]})")

# Scree Test (explained variance > 1)
num_components_to_retain = np.sum(eigenvalues > 1)
print(f"\nNumber of Principal Components to Retain (explained variance > 1): {num_components_to_retain}")
# Kaiser Criterion
num_components_kaiser = np.sum(eigenvalues > 1)
print(f"\nNumber of Principal Components to Retain (Kaiser Criterion): {num_components_kaiser}")

# Percentage of Variance Retained
percentage_variance_retained = np.sum(explained_variance_ratio)
print(f"\nPercentage of Variance Retained: {percentage_variance_retained * 100:.2f}%")

# Parallel Analysis (for comparison, assuming random data)
num_random_data = returns_standardized.shape[0]
random_data = np.random.normal(size=returns_standardized.shape)
pca_random = PCA()
pca_random.fit(random_data)
random_eigenvalues = pca_random.explained_variance_
num_components_parallel = np.sum(eigenvalues > random_eigenvalues)
print(f"\nNumber of Principal Components to Retain (Parallel Analysis): {num_components_parallel}")

# Outlier Detection
from sklearn.ensemble import IsolationForest

outlier_detector = IsolationForest(contamination=0.05)  # Assuming 5% contamination
outlier_detector.fit(returns_standardized)
outliers = outlier_detector.predict(returns_standardized)
num_outliers = np.sum(outliers == -1)
print(f"\nNumber of Outliers Detected: {num_outliers}")
