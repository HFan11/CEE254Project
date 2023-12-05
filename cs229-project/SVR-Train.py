import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the Modified Dataset
df = pd.read_csv("C:\\Users\\Haoch\\Desktop\\cs229-project\\Transform Modified DataSet New.csv")

# Selecting features and target
X = df[['Population', 'humidity','Average wind speed(Kmh)']]
y = df['water consumption(m^3/hr)']

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# It's usually a good idea to scale your data for SVM
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_Y.fit_transform(y_train.values.reshape(-1, 1))

# Initialize the SVM Regressor
model = SVR(kernel='rbf', C=100)  # You can experiment with the kernel and C value
model.fit(X_train_scaled, y_train_scaled.ravel())

# Predict on the test set
X_test_scaled = scaler_X.transform(X_test)
y_pred_scaled = model.predict(X_test_scaled)

# Transform the prediction back to original scale
y_pred = scaler_Y.inverse_transform(y_pred_scaled.reshape(-1, 1))

# R^2 (Coefficient of Determination)
r2 = r2_score(y_test, y_pred)

# Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

print(f"R^2 (Coefficient of Determination): {r2}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
# Scatter plot
plt.scatter(y_pred, y_test, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # identity line

plt.title('Test Value vs Predicted Value')
plt.xlabel('Predicted Value')
plt.ylabel('Test Value')
plt.grid(True)
plt.show()
