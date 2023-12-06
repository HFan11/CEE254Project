import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the Modified Dataset
df = pd.read_csv("C:\\Users\\Haoch\\Desktop\\cs229-project\\Modified DataSet.csv")

# Assuming df is your DataFrame
X = df[['Population']]
y = df['water consumption(m^3/hr)']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Ridge Regression model
# The alpha parameter controls the strength of the regularization.
# Higher values of alpha mean more regularization. You can experiment with different values.
alpha_value = 1.0
model = Ridge(alpha=alpha_value)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# R^2 (Coefficient of Determination)
r2 = r2_score(y_test, y_pred)

# Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Squared Error for each data point
squared_errors = (y_test - y_pred)**2

print(f"R^2 (Coefficient of Determination): {r2}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Squared Errors for each data point: \n{squared_errors}")

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



