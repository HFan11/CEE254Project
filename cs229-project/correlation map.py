import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Modified Dataset
df = pd.read_csv("C:\\Users\\Haoch\\Desktop\\cs229-project\\Modified DataSet New.csv")

# Map "Day of the Week" to numerical values
day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
df['Day of the Week'] = df['Day of the Week'].map(day_mapping)
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['DayOfMonth'] = df['Date'].dt.day


selected_columns = ['Population', 'T avg C', 'D.P avg C', 'humidity', 'Average wind speed(Kmh)', 'Ppt avg (in)', 'water consumption(m^3/hr)','Day of the Week','Month','DayOfMonth']
df_selected = df[selected_columns]

correlation_matrix = df_selected.corr()

# Assuming 'data' is your DataFrame and 'consumption' is your target variable
features = ['Population', 'T avg C', 'D.P avg C', 'humidity', 'Average wind speed(Kmh)', 'Ppt avg (in)','Day of the Week','Month','DayOfMonth']  # Replace with your actual feature names

# Determine the layout of your subplots
n_features = len(features)
n_cols = 3  # You can adjust this based on your preference
n_rows = n_features // n_cols + (n_features % n_cols > 0)

plt.figure(figsize=(n_cols * 5, n_rows * 5))  # Adjust the size as needed

for i, feature in enumerate(features):
    plt.subplot(n_rows, n_cols, i + 1)
    plt.scatter(df[feature], df['water consumption(m^3/hr)'],color='black')
    plt.title(f'{feature} vs. water consumption')
    plt.xlabel(feature)
    plt.ylabel('water consumption(m^3/hr)')

plt.tight_layout()
plt.subplots_adjust(hspace=0.5, wspace=0.4)  # Adjust horizontal and vertical space between plots
plt.show()