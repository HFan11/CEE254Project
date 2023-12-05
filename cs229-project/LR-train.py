import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load the Modified Dataset
df = pd.read_csv("C:\\Users\\Haoch\\Desktop\\cs229-project\\Transform Modified DataSet New.csv")

# Map "Day of the Week" to numerical values
day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
df['Day of the Week'] = df['Day of the Week'].map(day_mapping)

# Assuming df is your DataFrame
X = df[['Population', 'humidity','Average wind speed(Kmh)','Day of the Week']]
y = df['water consumption(m^3/hr)']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize Ridge Regression model
# The alpha parameter controls the strength of the regularization.
# Higher values of alpha mean more regularization. You can experiment with different values.
alpha_value = 0.5
model = Ridge(alpha=alpha_value)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# R^2 (Coefficient of Determination)
r2 = r2_score(y_test, y_pred)

# Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

print(f"R^2 (Coefficient of Determination): {r2}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")

coef = model.coef_[0]
print(f"Coefficient for Population: {coef}")

# Scatter plot
plt.scatter(y_pred, y_test, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # identity line

plt.title('Test Value vs Predicted Value')
plt.xlabel('Predicted Value')
plt.ylabel('Test Value')
plt.grid(True)
plt.show()


