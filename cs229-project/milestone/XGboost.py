from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
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
if 'Date' in data_no_outliers.columns:
    data_no_outliers['Date'] = pd.to_datetime(data_no_outliers['Date'])

# Handle missing data by interpolating
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
data['temp_humidity_interaction'] = np.log(data['T avg C'] + 1) * data['humidity']**6
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

# Refined hyperparameter optimization
params = {
    "colsample_bytree": uniform(0.75, 0.15), # refined based on previous results
    "gamma": uniform(0, 0.2), # refined
    "learning_rate": uniform(0.05, 0.1), # more focused range
    "max_depth": randint(4, 6), # more focused depth
    "n_estimators": randint(150, 250), # exploring higher number of trees
    # Adding L1 and L2 regularization
    "reg_alpha": uniform(0.01, 0.1),
    "reg_lambda": uniform(0.01, 0.1)
}

# Initialize XGBoost with default parameters
xgb_model = XGBRegressor(random_state=42)

# Fit the model on the training data
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Calculate the R² score
r2 = r2_score(y_test, y_pred)
print("R² score with default XGBoost parameters:", r2)

# Scatter plot for predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Diagonal line for reference
plt.xlabel('Actual Water Consumption (m^3/hr)')
plt.ylabel('Predicted Water Consumption (m^3/hr)')
plt.title('Water Consumption: Actual vs Predicted')
plt.show()