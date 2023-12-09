import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint
from sklearn.cluster import KMeans
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures
from xgboost import XGBRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.feature_selection import SelectFromModel
# Load the dataset
data = pd.read_csv('C:/Users/Haoch/Desktop/cs229-project/Modified Dataset.csv')

# Standard deviation method for water consumption
consumption_std = data['water consumption(m^3/hr)'].std()
consumption_mean = data['water consumption(m^3/hr)'].mean()

# Standard deviation method for humidity
humidity_std = data['humidity'].std()
humidity_mean = data['humidity'].mean()

# Define your outlier cutoff (commonly set to 2 or 3 standard deviations from the mean)
cutoff = 3

# Calculate the bounds for what's considered an outlier
lower_bound_consumption = consumption_mean - (cutoff * consumption_std)
upper_bound_consumption = consumption_mean + (cutoff * consumption_std)
lower_bound_humidity = humidity_mean - (cutoff * humidity_std)
upper_bound_humidity = humidity_mean + (cutoff * humidity_std)

# Remove outliers based on the standard deviation method
data_no_outliers = data[
    (data['water consumption(m^3/hr)'] > lower_bound_consumption) & 
    (data['water consumption(m^3/hr)'] < upper_bound_consumption) &
    (data['humidity'] > lower_bound_humidity) & 
    (data['humidity'] < upper_bound_humidity)
]

# Assuming 'Date' is a column with dates, convert it to datetime
#if 'Date' in data:
    #data['Date'] = pd.to_datetime(data['Date'])

if 'Date' in data_no_outliers.columns:
    data_no_outliers['Date'] = pd.to_datetime(data_no_outliers['Date'])

# Handle missing data by interpolating
#interpolated_data = data.interpolate()
interpolated_data = data_no_outliers.interpolate()

# Save the processed dataset
processed_dataset_path = 'C:/Users/Haoch/Desktop/cs229-project/Processed Dataset.csv'
interpolated_data.to_csv(processed_dataset_path, index=False)



data = pd.read_csv('C:/Users/Haoch/Desktop/cs229-project/Processed Dataset.csv')


# Existing feature engineering
data['Population_2'] = data['Population']
data['Ppt_avg_log'] = np.log(data['Ppt avg (in)'] + 1)
data['Ppt_avg_exp_decay'] = np.exp(-data['Ppt avg (in)'])
data['Ppt_avg_log1'] = np.log1p(data['Ppt avg (in)'])
data['Ppt_avg'] = data['Ppt avg (in)']
data['temp_bin'] = pd.cut(data['T avg C'], bins=5, labels=False)
data['temp_humidity_interaction'] = np.log(data['T avg C'] + 1) * data['humidity']
data['Average wind speed(Kmh)_adj'] = data['Average wind speed(Kmh)']


data['water_consumption_log'] = np.log1p(data['water consumption(m^3/hr)'])
# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month
data['DayOfMonth'] = data['Date'].dt.day

# Additional feature engineering
# Non-linear transformations
data['humidity_sqrt'] = np.sqrt(data['humidity'])
data['Population_log'] = np.log(data['Population'] + 1)

# Discretization of a continuous variable
discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
data['TempC_bins'] = discretizer.fit_transform(data[['T avg C']])

# Clustering as a feature
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(data[['Population', 'T avg C', 'humidity']])

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
ppt_poly_features = poly.fit_transform(data[['Ppt_avg']])
ppt_poly_df = pd.DataFrame(ppt_poly_features, columns=poly.get_feature_names_out(['Ppt_avg']))

# Drop the original 'Ppt_avg' column to avoid duplicate feature names
data.drop(columns=['Ppt_avg'], inplace=True)

data = pd.concat([data, ppt_poly_df], axis=1)

# Selecting features for the model
feature_columns = [col for col in data.columns if col not in ['water consumption(m^3/hr)','water_consumption_log','Date','Day of the Week']]
X = data[feature_columns]
y = data['water_consumption_log']


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

# Revert predictions and actual values back to original scale if they were log-transformed
exp_y_pred = np.expm1(y_pred)
exp_y_test = np.expm1(y_test)

# Calculate the R² score
r2 = r2_score(exp_y_test, exp_y_pred)

# Mean Squared Error (MSE)
mse = mean_squared_error(exp_y_test, exp_y_pred)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(exp_y_test, exp_y_pred)

print(f"R^2 (Coefficient of Determination): {r2}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")

plt.scatter(exp_y_pred, exp_y_test, alpha=0.5, color='black')  # Change color to black
plt.plot([min(exp_y_test), max(exp_y_test)], [min(exp_y_test), max(exp_y_test)], color='red')  # identity line
plt.xlabel('Predicted Value')
plt.ylabel('Test Value')
plt.show()

# Make predictions on the test set using the best model
first_80_percent = data.iloc[:int(len(data) * 0.8)]
time_series_data = data.iloc[int(len(data) * 0.8):]  # Last 20%

dates_first_80 = first_80_percent['Date']
observed_first_80 = np.expm1(first_80_percent['water_consumption_log'])
# Extract features, target and dates for the time series subset
X_time_series = time_series_data[feature_columns]
X_time_series_selected = scaler_X.transform(X_time_series)
y_time_series = time_series_data['water_consumption_log']
time_series_dates = time_series_data['Date']

# Predict on the time series subset
y_time_series_scaled= model.predict(X_time_series_selected)
y_time_series_pred= scaler_Y.inverse_transform(y_time_series_scaled)

# Revert predictions and actual values back to original scale if they were log-transformed
exp_y_time_series_pred = np.expm1(y_time_series_pred)
exp_y_time_series = np.expm1(y_time_series)

# Plotting observed vs. predicted values over time
plt.figure(figsize=(12, 6))
plt.plot(dates_first_80,observed_first_80,label='Observed (First 80%)', color='black', alpha=0.7 )
plt.plot(time_series_dates, exp_y_time_series, label='Observed (Last 20%)', color='black', alpha=0.7)
plt.plot(time_series_dates, exp_y_time_series_pred, label='Predicted (Last 20%)', color='red', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Water Consumption (m^3/hr)')
plt.legend()
plt.show()