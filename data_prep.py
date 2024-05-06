import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


# List all the csv files in the 'Portfolio 2' directory
csv_files = [f for f in os.listdir('Portfolio 2') if f.endswith('.csv')]

# For each csv file, load it into a pandas DataFrame
# Only include the 'Date' and 'Close' columns
# Rename the 'Close' column to the name of the stock
dfs = [pd.read_csv(os.path.join('Portfolio 2', f), usecols=['Date', 'Close']).rename(columns={'Close': f[:-4]}).set_index('Date') for f in csv_files]

# Join all the DataFrames on the 'Date' index
data = pd.concat(dfs, axis=1)

# Save the final DataFrame to a new csv file 'portfolio2.csv'
#data.to_csv('portfolio2.csv')

# print head data 
print(data.head())
print(data.tail())

