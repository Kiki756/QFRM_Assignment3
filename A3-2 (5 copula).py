import numpy as np
import pandas as pd
from pycopula.copula import GaussianCopula, StudentCopula
from scipy.stats import rankdata
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data_portfolio = pd.read_csv('portfolio.csv')
data_portfolio['Date'] = pd.to_datetime(data_portfolio['Date'])
data_portfolio.set_index('Date', inplace=True)
data_portfolio = data_portfolio[['Gold USD', 'Gold EUR', 'JPM Close', 'Siemens Close', 'EUROSTOXX Close']]
data_portfolio.replace(',', '', regex=True, inplace=True)
data_portfolio = data_portfolio.astype(float)

# make data returns
data = np.log(data_portfolio / data_portfolio.shift(1)).dropna()
data = data.apply(pd.to_numeric, errors='coerce')
data.dropna(inplace=True)

# Normalize the data using ranks to transform the margins to uniform
ranked_data = np.array([rankdata(data[col])/len(data[col]) for col in data.columns]).T

# Initialize a Gaussian copula
copula = GaussianCopula(dim=5)

# Fit the copula model
copula.fit(ranked_data)

# Retrieve correlation matrix (parameter of the Gaussian copula)
correlation_matrix = copula.params

# Visualize the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of the Gaussian Copula')
plt.show()

# Generate samples from the copula
samples = copula.sample(1000)

# Plot the generated samples
sns.pairplot(pd.DataFrame(samples, columns=data.columns))
plt.suptitle('Pairplot of Samples from 5-variate Gaussian Copula', y=1.02)
plt.show()