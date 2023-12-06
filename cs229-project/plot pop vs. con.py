import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("C:\\Users\\Haoch\\Desktop\\cs229-project\\Modified DataSet New.csv")

# Plotting
plt.figure(figsize=(10,6))
plt.scatter(df['Population'], df['water consumption(m^3/hr)'], alpha=0.5)
plt.title('Water Consumption vs Population')
plt.xlabel('Population')
plt.ylabel('Water Consumption (m^3/hr)')
plt.grid(True)
plt.show()
