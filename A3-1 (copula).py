import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copulas.bivariate import Clayton, Frank, Gumbel
from scipy.stats import rankdata, kendalltau
from itertools import combinations

# Load and prepare data
data_portfolio = pd.read_csv('portfolio.csv')
data_portfolio['Date'] = pd.to_datetime(data_portfolio['Date'])
data_portfolio.set_index('Date', inplace=True)
data_portfolio = data_portfolio[['Gold USD', 'Gold EUR', 'JPM Close', 'Siemens Close', 'EUROSTOXX Close']]
data_portfolio.replace(',', '', regex=True, inplace=True)
data_portfolio = data_portfolio.astype(float)
log_returns = np.log(data_portfolio / data_portfolio.shift(1)).dropna()

# Initialize copula models
copulas = {
    'Clayton': Clayton(),
    'Frank': Frank(),
    'Gumbel': Gumbel(),
}

# Generate all possible pairs of assets
asset_pairs = list(combinations(log_returns.columns, 2))

# Process each pair
for asset1, asset2 in asset_pairs:
    pair_name = f'{asset1} and {asset2}'
    rank1 = rankdata(log_returns[asset1].values) / len(log_returns[asset1])
    rank2 = rankdata(log_returns[asset2].values) / len(log_returns[asset2])

    # Compute Kendall's tau for the pair
    tau, _ = kendalltau(rank1, rank2)
    formatted_tau = f"{tau:.3f}"  # Format tau to three decimal places

    fig, ax = plt.subplots(1, 4, figsize=(24, 6))  # One row, four columns of subplots (added one for just empirical data)
    fig.suptitle(f'Copula Analysis for {pair_name}', fontsize=16)

    # Plot just the empirical data in the first subplot
    ax[0].scatter(rank1, rank2, alpha=0.25, s=1, label='Empirical Data')  # Smaller dots
    ax[0].set_title('Empirical Data')
    ax[0].set_xlabel(f'{asset1} Rank')
    ax[0].set_ylabel(f'{asset2} Rank')
    ax[0].legend()

    # Process each copula
    for i, (name, copula) in enumerate(copulas.items(), start=1):  # start=1 to use subplots after the first one
        data = np.column_stack((rank1, rank2))
        try:
            copula.fit(data)
            samples = copula.sample(1000)
            parameters = copula.theta
            formatted_parameters = f"{parameters:.3f}"  # Format parameters to three decimal places
            ax[i].scatter(rank1, rank2, alpha=0.25, s=1, label='Empirical Data')  # Smaller dots
            ax[i].scatter(samples[:, 0], samples[:, 1], alpha=0.25, s=1, color='red', label='Copula Samples')  # Smaller dots
            ax[i].set_title(f'{name} Copula\nTheta: {formatted_parameters}, Tau: {formatted_tau}')
            ax[i].set_xlabel(f'{asset1} Rank')
            ax[i].set_ylabel(f'{asset2} Rank')
            ax[i].legend()
        except Exception as e:
            print(f"Error processing {name} copula for {asset1} and {asset2}: {e}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    # Save each figure with a unique name based on the asset pair
    filename = f"{pair_name.replace(' ', '_').replace('&', 'and').replace(',', '')}.png"
    plt.savefig(filename)


