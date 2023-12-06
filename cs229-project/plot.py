import pandas as pd
import matplotlib.pyplot as plt

# Load the Modified Dataset
df = pd.read_csv("C:\\Users\\Haoch\\Desktop\\cs229-project\\Modified DataSet.csv")

# Plot water consumption vs humidity
plt.figure(figsize=(12, 6))
plt.scatter(df['humidity'], df['water consumption(m^3/hr)'], alpha=0.6, edgecolors="w", linewidth=0.5)
plt.title("Water Consumption vs Humidity")
plt.xlabel("Humidity")
plt.ylabel("Water Consumption (m^3/hr)")
plt.show()

# Plot water consumption vs T avg C
plt.figure(figsize=(12, 6))
plt.scatter(df['T avg C'], df['water consumption(m^3/hr)'], alpha=0.6, edgecolors="w", linewidth=0.5)
plt.title("Water Consumption vs T avg C")
plt.xlabel("T avg C")
plt.ylabel("Water Consumption (m^3/hr)")
plt.show()
