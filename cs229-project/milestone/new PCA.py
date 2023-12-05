import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Read the Dataset
file_path = 'C:\\Users\\Haoch\\Desktop\\cs229-project\\Modified Dataset.csv'
data = pd.read_csv(file_path)

# Step 2: Pre-process the Data
# Assuming your dataset is already clean and ready for PCA

# Dropping the target variable for PCA and any non-relevant or non-numeric columns
features = data.drop(columns=['water consumption(m^3/hr)', 'Date','Day of the Week'])

# Standardizing the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 3: Apply PCA
# Using the first three principal components as discussed
pca = PCA(n_components=3)
transformed_features = pca.fit_transform(scaled_features)

# Creating a DataFrame for the transformed features
transformed_df = pd.DataFrame(transformed_features, columns=['PC1', 'PC2', 'PC3'])

# Step 4: Prepare the Target Variable
y = data['water consumption(m^3/hr)']  # Target variable

# Now you can proceed with the split and model training as mentioned earlier
X = transformed_df  # Features from PCA
y = data['water consumption(m^3/hr)']  # Target variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing models
lr = LinearRegression()
svm = SVR()
rf = RandomForestRegressor()

# Training models
lr.fit(X_train, y_train)
svm.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Making predictions
y_pred_lr = lr.predict(X_test)
y_pred_svm = svm.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Evaluating models
mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_svm = mean_squared_error(y_test, y_pred_svm)
mse_rf = mean_squared_error(y_test, y_pred_rf)

r2_lr = r2_score(y_test, y_pred_lr)
r2_svm = r2_score(y_test, y_pred_svm)
r2_rf = r2_score(y_test, y_pred_rf)

# Print the evaluation results for each model
print("Linear Regression - MSE:", mse_lr, "R²:", r2_lr)
print("SVM - MSE:", mse_svm, "R²:", r2_svm)
print("Random Forest - MSE:", mse_rf, "R²:", r2_rf)