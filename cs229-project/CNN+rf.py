import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('C:/Users/Haoch/Desktop/cs229-project/Modified Dataset.csv')

# Preprocess the data
# One-hot encode categorical columns if necessary (e.g., 'Day of the Week')
categorical_features = ['Day of the Week']
numerical_features = ['Population', 'T avg C', 'D.P avg C', 'humidity', 'Average wind speed(Kmh)', 'Ppt avg (in)']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

X = preprocessor.fit_transform(data)
y = data['water consumption(m^3/hr)'].values

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for CNN (treating each data point as a 1D image)
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# CNN model for feature extraction
cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten()
])

# Add a dense layer for dimensionality reduction (if needed)
cnn_model.add(Dense(50, activation='relu'))

# Compile the model
cnn_model.compile(optimizer='adam', loss='mse')

# Train the model (you might need to adjust epochs and batch size)
cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=32)


# Extract features from the trained CNN
features_train = cnn_model.predict(X_train_cnn)
features_test = cnn_model.predict(X_test_cnn)

# Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train Random Forest on extracted features
rf.fit(features_train, y_train)



# Predict and evaluate
y_pred = rf.predict(features_test)
r2_score = r2_score(y_test, y_pred)

print(f'R^2 score: {r2_score}')

# Plotting Predicted vs Actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Diagonal line for reference
plt.xlabel('Actual Water Consumption (m^3/hr)')
plt.ylabel('Predicted Water Consumption (m^3/hr)')
plt.title('Water Consumption: Actual vs Predicted')
plt.show()