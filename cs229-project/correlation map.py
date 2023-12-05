import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Modified Dataset
df = pd.read_csv("C:\\Users\\Haoch\\Desktop\\cs229-project\\Modified DataSet New.csv")

# Map "Day of the Week" to numerical values
day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
df['Day of the Week'] = df['Day of the Week'].map(day_mapping)

selected_columns = ['Population', 'T avg C', 'D.P avg C', 'humidity', 'Average wind speed(Kmh)', 'Ppt avg (in)', 'water consumption(m^3/hr)','Day of the Week']
df_selected = df[selected_columns]

correlation_matrix = df_selected.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.show()

