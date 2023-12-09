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
#data.sort_values('Date', inplace=True)

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

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature selection using RandomForestRegressor
rf_selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))
X_train_selected = rf_selector.fit_transform(X_train, y_train)
X_test_selected = rf_selector.transform(X_test)

# Define the cross-validation strategy
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the hyperparameter space for RandomizedSearchCV
params = {
    "colsample_bytree": uniform(0.75, 0.15),
    "gamma": uniform(0, 0.2),
    "learning_rate": uniform(0.05, 0.1),
    "max_depth": randint(4, 6),
    "n_estimators": randint(150, 250),
    "reg_alpha": uniform(0.01, 0.1),
    "reg_lambda": uniform(0.01, 0.1)
}

# Initialize the RandomizedSearchCV with the XGBRegressor and the hyperparameter space
random_search = RandomizedSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_distributions=params,
    n_iter=100,  # Number of parameter settings that are sampled
    cv=cv_strategy,  # Cross-validation strategy
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='r2'
)

# Fit the RandomizedSearchCV to find the best model using selected features
random_search.fit(X_train_selected, y_train)    

# Print the best score found during the RandomizedSearchCV
print(f"Best R² score from cross-validation: {random_search.best_score_}")

# Make predictions on the test set using the best model
y_pred = random_search.best_estimator_.predict(X_test_selected)

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

# Plotting observed vs. predicted values
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
X_time_series_selected = rf_selector.transform(X_time_series)
y_time_series = time_series_data['water_consumption_log']
time_series_dates = time_series_data['Date']

# Predict on the time series subset
y_time_series_pred = random_search.best_estimator_.predict(X_time_series_selected)

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
# Split the dataset into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Refined hyperparameter optimization
#params = {
    #"colsample_bytree": uniform(0.75, 0.15), # refined based on previous results
    #"gamma": uniform(0, 0.2), # refined
    #"learning_rate": uniform(0.05, 0.1), # more focused range
    #"max_depth": randint(4, 6), # more focused depth
    #"n_estimators": randint(150, 250), # exploring higher number of trees
        # Adding L1 and L2 regularization
    #"reg_alpha": uniform(0.01, 0.1),
    #"reg_lambda": uniform(0.01, 0.1)
#}

# Initialize XGBoost with default parameters
#xgb_model = XGBRegressor(random_state=42)

# Fit the model on the training data
#xgb_model.fit(X_train, y_train)

# Make predictions on the test set
#y_pred = xgb_model.predict(X_test)

# Calculate the R² score
#r2 = r2_score(y_test, y_pred)
#print("R² score with default XGBoost parameters:", r2)

# Scatter plot for predicted vs actual values
#plt.figure(figsize=(10, 6))
#plt.scatter(y_test, y_pred, alpha=0.5)
#plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Diagonal line for reference
#plt.xlabel('Actual Water Consumption (m^3/hr)')
#plt.ylabel('Predicted Water Consumption (m^3/hr)')
#plt.title('Water Consumption: Actual vs Predicted')
#plt.show()