import pandas as pd

# Load the Dataset
df = pd.read_csv("C:\\Users\\Haoch\\Desktop\\cs229-project\\Modified DataSet Cleaned.csv")

# Apply the transformations based on the population ranges
df.loc[(df['Population'] > 2000) & (df['Population'] < 3000), 'water consumption(m^3/hr)'] += 1000
df.loc[(df['Population'] > 3000) & (df['Population'] < 3500), 'water consumption(m^3/hr)'] += 1500
df.loc[(df['Population'] > 3500) & (df['Population'] < 4000), 'water consumption(m^3/hr)'] += 2000
df.loc[df['Population'] > 4000, 'water consumption(m^3/hr)'] += 2500

# Save the modified data to a new CSV file
df.to_csv("C:\\Users\\Haoch\\Desktop\\cs229-project\\Transformed DataSet.csv", index=False)
