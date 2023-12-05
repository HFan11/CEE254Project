import pandas as pd
import matplotlib.pyplot as plt

# Load the Modified Dataset
df = pd.read_csv("C:\\Users\\Haoch\\Desktop\\cs229-project\\Modified DataSet New.csv")

def remove_outliers(dataframe, column_name):
    Q1 = dataframe[column_name].quantile(0.25)
    Q3 = dataframe[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    # Return only rows where the value in the specified column is within the IQR boundaries
    return dataframe[(dataframe[column_name] >= lower_limit) & (dataframe[column_name] <= upper_limit)]

# Remove outliers for 'Population'
df_cleaned_pop = remove_outliers(df, 'Population')
# Remove outliers for 'water consumption(m^3/hr)' from the already cleaned dataframe
df_cleaned_water = remove_outliers(df_cleaned_pop, 'water consumption(m^3/hr)')

# Combine the indices of rows to drop
rows_to_drop = df.index.difference(df_cleaned_water.index)

# Drop those rows from the original dataframe
df_cleaned = df.drop(rows_to_drop)

# Save the cleaned data to a new CSV file
df_cleaned.to_csv("C:\\Users\\Haoch\\Desktop\\cs229-project\\Transform Modified DataSet New.csv", index=False)

# Scatter plot for the cleaned data
plt.scatter(df_cleaned['Population'], df_cleaned['water consumption(m^3/hr)'], alpha=0.5)
plt.title('Water Consumption vs Population (Cleaned Data)')
plt.xlabel('Population')
plt.ylabel('Water Consumption (m^3/hr)')
plt.grid(True)
plt.show()

