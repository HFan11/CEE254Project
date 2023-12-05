import pandas as pd
import numpy as np

# Load the Original Dataset
df = pd.read_csv("C:\\Users\\Haoch\\Desktop\\cs229-project\\Modified DataSet.csv")

# Adjust the values based on the population ranges with random mean values
sigma = 50
df.loc[(df['Population'] > 2000) & (df['Population'] < 3000), 'water consumption(m^3/hr)'] += np.random.normal(1000, sigma)
df.loc[(df['Population'] > 3000) & (df['Population'] < 3500), 'water consumption(m^3/hr)'] += np.random.normal(1500, sigma)
df.loc[(df['Population'] > 3500) & (df['Population'] < 4000), 'water consumption(m^3/hr)'] += np.random.normal(2000, sigma)
df.loc[df['Population'] > 4000, 'water consumption(m^3/hr)'] += np.random.normal(2500, sigma)

correlation = df[['Population', 'water consumption(m^3/hr)']].corr().iloc[0, 1]

# Introduce noise until the correlation is close to 0.6
while correlation > 0.75:
    noise = np.random.normal(0, 1000, len(df))  # adjust the magnitude of noise as needed
    df['water consumption(m^3/hr)'] += noise
    correlation = df[['Population', 'water consumption(m^3/hr)']].corr().iloc[0, 1]

# Save the modified data to a new CSV file
df.to_csv("C:\\Users\\Haoch\\Desktop\\cs229-project\\Modified DataSet New.csv", index=False)

