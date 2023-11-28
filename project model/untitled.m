% Input: training data, test data
% Output: n by 1 matrix
train_data=
test_data=

function pred_pm2d5 = Michael_pred_model(train_data, test_data)
% Create a new column 'time_numeric' to hold the numerical time feature
train_data.time_numeric = hour(train_data.time) + minute(train_data.time)/60;

% Randomly sample 10,000 rows for training and 2,000 for validation
rng(0); % Seed for reproducibility
total_rows = height(train_data);
indices = randperm(total_rows, 12000);
train_indices = indices(1:10000);
validation_indices = indices(10001:end);

% Split the data
train_data_subset = train_data(train_indices, :);
validation_data_subset = train_data(validation_indices, :);

% Extract features and response for training
X_train = train_data_subset{:, {'time_numeric', 'hmd', 'spd', 'tmp', 'lat', 'lon'}};
y_train = train_data_subset.pm2d5;

% Extract features and response for validation
X_validation = validation_data_subset{:, {'time_numeric', 'hmd', 'spd', 'tmp', 'lat', 'lon'}};
y_validation = validation_data_subset.pm2d5;

% Scale the features, since SVMs are sensitive to the scaling of the data
% Use zscore for scaling
%X_train_scaled = zscore(X_train);
%X_validation_scaled = zscore(X_validation);

% Train the SVM regression model on the training data
SVMModel = fitrsvm(X_train, y_train, 'Standardize', true, 'KernelFunction', 'rbf');

% Validate the model on the validation data
predicted_pm2d5_validation = predict(SVMModel, X_validation);


end