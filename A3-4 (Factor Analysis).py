import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from factor_analyzer import FactorAnalyzer

# Load your data
data = pd.read_csv('portfolio2.csv')
data.set_index('Date', inplace=True)
data.astype(float)

#make data returns
data = data.pct_change().dropna()

# Preprocessing data: assuming the data is returns and not prices, drop NaN if any
data = data.dropna()

# make date column the index

# Setting up Factor Analysis using oblique rotation
# The number of factors is set to be less than the number of variables (you may adjust this number)
fa = FactorAnalyzer(rotation='oblimin', n_factors=5, method='principal')
fa.fit(data)

# Check the Loadings (correlations between variables and factors)
loadings = fa.loadings_
print("Factor Loadings:\n", loadings)

# Get Eigenvalues and explained variance
ev, v = fa.get_eigenvalues()
print("Eigenvalues:\n", ev)

# Compare Factors with PCA Components
pca = PCA(n_components=10)
pca.fit(data)
print("\nPCA Components:\n", pca.components_)
print("\nExplained Variance (PCA):\n", pca.explained_variance_ratio_)

# Plotting the factor loadings next to PCA loadings for comparison
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Factor Loadings plot
im = ax[0].imshow(loadings, cmap='coolwarm', aspect='auto')
ax[0].set_title('Factor Loadings (FA with oblique rotation)')
ax[0].set_ylabel('Variables')
ax[0].set_xlabel('Factors')
fig.colorbar(im, ax=ax[0])

# PCA Loadings plot
im2 = ax[1].imshow(pca.components_, cmap='coolwarm', aspect='auto')
ax[1].set_title('PCA Loadings')
ax[1].set_ylabel('Variables')
ax[1].set_xlabel('Principal Components')
fig.colorbar(im2, ax=ax[1])

plt.show()

