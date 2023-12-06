import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load your dataset
file_path = 'C:\\Users\\Haoch\\Desktop\\cs229-project\\Modified DataSet.csv'
data = pd.read_csv(file_path)

# One-hot encoding for the categorical variable 'Day of the Week'
data_encoded = pd.get_dummies(data.drop(columns=['Date']), columns=['Day of the Week'])

# Directory to save figures
save_dir = 'C:\\Users\\Haoch\\Desktop\\cs229-project\\data visualization\\'

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Function to create a valid file name
def create_valid_filename(title):
    for char in [' ', '(', ')', '/', '\\']:
        title = title.replace(char, '_')
    return title

# 1. Histograms/Boxplots for Key Features
key_features = ['Population', 'T avg C', 'D.P avg C', 'humidity', 'Average wind speed(Kmh)', 'Ppt avg (in)', 'water consumption(m^3/hr)']
for feature in key_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[feature], kde=True)
    title = f'Distribution of {feature}'
    plt.title(title)
    plt.savefig(save_dir + create_valid_filename(f'hist_{feature}.png'))  # Save histogram with a valid file name
    plt.show()

    plt.figure(figsize=(8, 4))
    sns.boxplot(x=data[feature])
    title = f'Boxplot of {feature}'
    plt.title(title)
    plt.savefig(save_dir + create_valid_filename(f'boxplot_{feature}.png'))  # Save boxplot with a valid file name
    plt.show()

# 2. Correlation Heatmap (using encoded data)
plt.figure(figsize=(12, 10))
sns.heatmap(data_encoded.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig(save_dir + 'correlation_heatmap.png')  # Save heatmap
plt.show()

# 3. Scatter Plots (excluding 'Day of the Week')
for feature in key_features:
    if feature != 'water consumption(m^3/hr)':  # Assuming 'water consumption(m^3/hr)' is your target variable
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=data[feature], y=data['water consumption(m^3/hr)'])
        title = f'Water Consumption vs {feature}'
        plt.title(title)
        plt.savefig(save_dir + create_valid_filename(f'scatter_{feature}_vs_consumption.png'))  # Save scatter plot with a valid file name
        plt.show()

