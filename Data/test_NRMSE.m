%% parameterized
width=30; % width of moving mean window
indexes=[2,3,4,5,6,7]; % indexes of columns averaged
dt = 30;% intervals of averaging
input_cols=[1,2,3,4,6,7]; % indexes of input columns
%features_to_standardize = [true, true,false,true];
test_cols=[1,2,3,4,5,6]; % indexes of test columns
%ker_func='rbf';
%train_file='separated_train_data_short_term_10_var.mat';
%test_file='test_data_short_term_10_var.mat';
% Random Forest parameters
numTrees = 1000; % Number of trees in the forest

maxDepth = 15; % Maximum depth of each tree
minLeafSize = 10; % Minimum number of samples per leaf
numVarsToSample = 'all'; % Options: 'all', 'log2', a fraction of the total, or an integer

raw_data=train_data;
raw_data=processAndSplitSensorData(raw_data);
x_input=[];
y_input=[];
x_test=[];
y_test=[];
    
%% preprocess

% Determine the split point for 80% training and 20% validation
numDataPoints = length(raw_data);
splitPoint = floor(numDataPoints * 0.8);

% Split the data into training and validation sets
train_data_raw = raw_data;
%val_data_raw = raw_data(splitPoint+1:splitPoint+100);

% Preprocess the training data
for i=1:length(train_data_raw)
    clear train_data_;
    train_data_=remove_outlier(train_data_raw{i});
    train_data_=averageByTimeInterval(train_data_,dt);
    train_data_=missing_filling(train_data_,dt);
    train_data_=move_mean(train_data_,width,indexes);
    for col=input_cols
        if col==1
            time{i}{col} = train_data_.time;
            x{i}{col}= (datenum(time{i}{col}))*24*60*60;
            
        else
            x{i}{col}=train_data_{:,col};
        end
    end
    x_input=cat(1,x_input,cat(2,x{i}{:}));
    y_{i}=train_data_{:,5};
    y_input=cat(1,y_input,y_{i});
end    
%% Implement the Prediction Model with Cross-Validation

% Number of folds for cross-validation
k = 5;

% Initialize variables to store the cross-validation results
cv_RMSE = zeros(k, 1);
cv_NRMSE = zeros(k, 1);

% Cross-validation indices
cv_indices = crossvalind('Kfold', size(x_input, 1), k);

for i = 1:k
    % Split data into training and validation for the current fold
    validationIdx = (cv_indices == i); 
    trainIdx = ~validationIdx;
    
    x_train = x_input(trainIdx, :);
    y_train = y_input(trainIdx);
    x_val = x_input(validationIdx, :);
    y_val = y_input(validationIdx);

    % Train Random Forest on the training set
    RFModel = TreeBagger(numTrees, x_train, y_train, 'Method', 'regression', ...
                     'MaxNumSplits', maxDepth, ...
                     'MinLeafSize', minLeafSize, ...
                     'NumVariablesToSample', numVarsToSample, ...
                     'OOBPrediction', 'on', 'OOBPredictorImportance', 'on');

    % Validate the model on the validation set
    y_pred_val = predict(RFModel, x_val);
    
    % Calculate RMSE and NRMSE for the validation set
    RMSE_val = rms(y_pred_val - y_val);
    normalizing_term_val = sqrt(mean(y_val.^2));
    NRMSE_val = RMSE_val / normalizing_term_val;

    % Store the results
    cv_RMSE(i) = RMSE_val;
    cv_NRMSE(i) = NRMSE_val;
end

% Calculate average RMSE and NRMSE across all folds
avg_RMSE = mean(cv_RMSE);
avg_NRMSE = mean(cv_NRMSE);

% Display the results
fprintf('Average RMSE across %d folds: %f\n', k, avg_RMSE);
fprintf('Average NRMSE across %d folds: %f\n', k, avg_NRMSE);

    
%% % testing
y_test=soln_data;    
clear x;
clear time;
for col=test_cols
    if col==1
       time{i}{col} = test_data.time;
       x{i}{col}= (datenum(time{i}{col}))*24*60*60;
        
    else
       x{i}{col}=test_data{:,col};
    end
end
x_test=cat(1,x_test,cat(2,x{i}{:}));
%y_pred_test=predict(SVMModel,x_test);
y_pred_test=predict(RFModel,x_test);
pred_pm2d5=y_pred_test;

%calculate the test error 
RMSE_test = rms(y_pred_test - y_test);
normalizing_term_test = sqrt(mean(y_test.^2));
NRMSE_test = RMSE_test / normalizing_term_test;