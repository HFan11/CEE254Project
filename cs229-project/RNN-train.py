import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
df = pd.read_csv("C:\\Users\\Haoch\\Desktop\\cs229-project\\Modified DataSet.csv")

# Applying .copy() to avoid SettingWithCopyWarning
df_no_outliers = df.copy()

# Outlier Removal
# Calculate standard deviation and mean for water consumption and humidity
consumption_std = df['water consumption(m^3/hr)'].std()
consumption_mean = df['water consumption(m^3/hr)'].mean()
humidity_std = df['humidity'].std()
humidity_mean = df['humidity'].mean()

# Define your outlier cutoff
cutoff = 3

# Calculate the bounds for what's considered an outlier
lower_bound_consumption = consumption_mean - (cutoff * consumption_std)
upper_bound_consumption = consumption_mean + (cutoff * consumption_std)
lower_bound_humidity = humidity_mean - (cutoff * humidity_std)
upper_bound_humidity = humidity_mean + (cutoff * humidity_std)

# Remove outliers
df_no_outliers = df_no_outliers[
    (df_no_outliers['water consumption(m^3/hr)'] > lower_bound_consumption) & 
    (df_no_outliers['water consumption(m^3/hr)'] < upper_bound_consumption) &
    (df_no_outliers['humidity'] > lower_bound_humidity) & 
    (df_no_outliers['humidity'] < upper_bound_humidity)
]

# Feature Engineering
df_no_outliers['Population_2'] = df_no_outliers['Population']
df_no_outliers['Ppt_avg_log'] = np.log(df_no_outliers['Ppt avg (in)'] + 1)
df_no_outliers['Ppt_avg_exp_decay'] = np.exp(-df_no_outliers['Ppt avg (in)'])
df_no_outliers['Ppt_avg_log1'] = np.log1p(df_no_outliers['Ppt avg (in)'])

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
df_no_outliers['Ppt_avg'] = df_no_outliers['Ppt avg (in)']
ppt_poly_features = poly.fit_transform(df_no_outliers[['Ppt_avg']])
ppt_poly_df = pd.DataFrame(ppt_poly_features, columns=poly.get_feature_names_out(['Ppt_avg']))
df_no_outliers = pd.concat([df_no_outliers, ppt_poly_df], axis=1)

# Binning and Interaction Terms
df_no_outliers['temp_bin'] = pd.cut(df_no_outliers['T avg C'], bins=5, labels=False)
df_no_outliers['temp_humidity_interaction'] = np.log(df_no_outliers['T avg C']+1) * df_no_outliers['humidity']**4

# Weekday/Weekend
weekend_days = ['Saturday', 'Sunday']
df_no_outliers['Weekday_Weekend'] = df_no_outliers['Day of the Week'].apply(lambda x: 1 if x in weekend_days else 0)

# Selecting features and target
features = ['Population_2', 'T avg C', 'temp_humidity_interaction', 'Ppt_avg_log', 'Ppt_avg_exp_decay', 'Ppt_avg_log1', 'temp_bin', 'Weekday_Weekend'] + list(ppt_poly_df.columns)
X = df_no_outliers[features]
y = df_no_outliers['water consumption(m^3/hr)']

# Check for NaN or infinite values in dataset
if X.isnull().values.any() or np.isinf(X.values).any():
    raise ValueError("Data contains NaN or infinite values.")

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Scaling
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_Y.fit_transform(y_train.values.reshape(-1, 1))

# Check for NaN or infinite values after scaling
if np.isnan(X_train_scaled).any() or np.isnan(y_train_scaled).any():
    raise ValueError("Scaling resulted in NaN values.")

# Building the RNN Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Linear activation for regression
])

# Compiling the model
model.compile(optimizer='adam', loss='mse')

# Training the model
history = model.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=32, validation_split=0.2)

# Check for NaN loss
if np.isnan(history.history['loss']).any():
    raise ValueError("Model training resulted in NaN values in loss.")

# Predict on the test set
X_test_scaled = scaler_X.transform(X_test)
y_pred_scaled = model.predict(X_test_scaled)

# Transforming the prediction back to original scale
y_pred = scaler_Y.inverse_transform(y_pred_scaled)

# Evaluation Metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Output and Plotting
print(f"R^2 (Coefficient of Determination): {r2}")
print(f"Mean Squared Error (MSE): {mse}")

plt.scatter(y_pred, y_test, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # identity line
plt.title('Test Value vs Predicted Value')
plt.xlabel('Predicted Value')
plt.ylabel('Test Value')
plt.grid(True)
plt.show()
