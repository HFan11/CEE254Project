function pred_pm2d5 = Michael_pred_model(train_data,test_data,problem_type)
%% parameterized
for problem_type = 1:3
    if problem_type == 1
        width=30; % width of moving mean window
        dt = 30;% intervals of averaging
        numTrees = 1000; % Number of trees in the forest
        maxDepth = 15; % Maximum depth of each tree
        minLeafSize = 10; % Minimum number of samples per leaf
        numVarsToSample = 'all'; % Options: 'all', 'log2', a fraction of the total, or an integer
    elseif problem_type == 2
        width=30; % width of moving mean window
        dt = 5;
        numTrees = 1000; % Number of trees in the forest
        maxDepth = 20; % Maximum depth of each tree
        minLeafSize = 10; % Minimum number of samples per leaf
        numVarsToSample = 'all'; % Options: 'all', 'log2', a fraction of the total, or an integer
    elseif problem_type == 3
        width=30;
        dt = 5;
        numTrees = 250; % Number of trees in the forest
        maxDepth = 15; % Maximum depth of each tree
        minLeafSize = 1; % Minimum number of samples per leaf
        numVarsToSample = 'all'; % Options: 'all', 'log2', a fraction of the total, or an integer
    end
end
indexes=[2,3,4,5,6,7]; % indexes of columns averaged
input_cols=[2,3]; % indexes of input columns
%features_to_standardize = [true, true,false,true];
test_cols=[2,3]; % indexes of test columns
%ker_func='rbf';
%train_file='separated_train_data_short_term_10_var.mat';
%test_file='test_data_short_term_10_var.mat';
train_data=processAndSplitSensorData(train_data);
x_input=[];
y_input=[];
x_test=[];
y_test=[];
%% preprocess
for i=1:length(train_data)
    clear train_data_;
    train_data_=remove_outlier(train_data{i});
    train_data_=averageByTimeInterval(train_data_,dt);
    train_data_ = missing_filling(train_data_,dt);
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
%% Shuffle the dataset
randIndices = randperm(size(x_input, 1));
x_input_shuffled = x_input(randIndices, :);
y_input_shuffled = y_input(randIndices, :);

% K-fold cross-validation
k = 5;
cv = cvpartition(size(x_input_shuffled, 1), 'KFold', k);

for i = 1:k
    x_train = x_input_shuffled(training(cv, i), :);
    y_train = y_input_shuffled(training(cv, i), :);
    x_val = x_input_shuffled(test(cv, i), :);
    y_val = y_input_shuffled(test(cv, i), :);

    % Train the Random Forest model
    RFModel = TreeBagger(numTrees, x_train, y_train, 'Method', 'regression', ...
                         'MaxNumSplits', maxDepth, ...
                         'MinLeafSize', minLeafSize, ...
                         'NumVariablesToSample', numVarsToSample, ...
                         'OOBPrediction', 'on', 'OOBPredictorImportance', 'on');

    % Predict on the validation set
    y_pred_val = predict(RFModel, x_val);
    
    % Calculate RMSE and NRMSE for the validation set
    RMSE_val = rms(y_pred_val - y_val);
    normalizing_term_val = sqrt(mean(y_val.^2));
    NRMSE_val = RMSE_val / normalizing_term_val;

    % Store the results
    cv_RMSE(i) = RMSE_val;
    cv_NRMSE(i) = NRMSE_val;
end
%% % testing
%y_test=soln_data;    
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
%RMSE_test = rms(y_pred_test - y_test);
%normalizing_term_test = sqrt(mean(y_test.^2));
%NRMSE_test = RMSE_test / normalizing_term_test;
end