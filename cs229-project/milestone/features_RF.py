from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load the dataset
data = pd.read_csv('C:/Users/admin/Desktop/cs229-project/Modified Dataset.csv')

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
if 'Date' in data_no_outliers.columns:
    data_no_outliers['Date'] = pd.to_datetime(data_no_outliers['Date'])

# Handle missing data by interpolating
interpolated_data = data_no_outliers.interpolate()

# Save the processed dataset
processed_dataset_path = 'C:/Users/admin/Desktop/cs229-project/Processed Dataset.csv'
interpolated_data.to_csv(processed_dataset_path, index=False)



data = pd.read_csv('C:/Users/admin/Desktop/cs229-project/Processed Dataset.csv')


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
feature_columns = [col for col in data.columns if col not in ['water consumption(m^3/hr)', 'Date','Day of the Week']]
X = data[feature_columns]
y = data['water_consumption_log']




# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],  # More trees for a more stable average
    'max_depth': [None, 20, 30, 40],  # Testing deeper trees as well
    'min_samples_split': [2, 5, 10],  # Including lower values to allow more splitting
    'min_samples_leaf': [1, 2, 4],  # Testing the default value as well
    'max_features': ['auto', 'sqrt'],  # 'auto' uses all features
    'bootstrap': [True, False],  # Testing with and without bootstrap samples
}


# Initialize the Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Initialize the Grid Search with cross-validation
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           n_jobs=-1,
                           verbose=2,
                           scoring='r2')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Best parameters found by grid search
best_params = grid_search.best_params_

# Print the best parameters
print("Best parameters found: ", best_params)

# Initialize the best model based on grid search
best_rf = grid_search.best_estimator_

# Train the model
best_rf.fit(X_train, y_train)

# Predict using the best model
y_pred_rf = best_rf.predict(X_test)

# Calculate R^2 score
r2_rf = r2_score(y_test, y_pred_rf)

# Print R² score
print("R²:", r2_rf)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Diagonal line for reference
plt.xlabel('True Values [Water Consumption]')
plt.ylabel('Predictions [Water Consumption]')
plt.title(f'Water Consumption: True vs Predicted - R²: {r2_rf:.2f}')
plt.show()