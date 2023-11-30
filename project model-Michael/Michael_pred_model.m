function pred_pm2d5 = Michael_pred_model(train_file, test_file)
       clear all
%% parameterized
width=30; % width of moving mean window
indexes=[2,3,4,5,6,7]; % indexes of columns averaged
dt = 5;% intervals of averaging
input_cols=[1,2,3,4,6,7]; % indexes of input columns
%features_to_standardize = [true, true,false,true];
test_cols=[1,2,3,4,5,6]; % indexes of test columns
%ker_func='rbf';
%train_file='separated_train_data_short_term_10_var.mat';
%test_file='test_data_short_term_10_var.mat';
% Random Forest parameters
numTrees = 250; % Number of trees in the forest
    
data=load(train_file);
raw_data=data.train_data_static;

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
train_data_raw = raw_data(1:splitPoint);
%val_data_raw = raw_data(splitPoint+1:splitPoint+100);

% Preprocess the training data
for i=1:length(train_data_raw)
    clear train_data_;
    train_data_=remove_outlier(train_data_raw{i});
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
    
%% implement the prediction model
    
% Split the data into training and validation sets
numDataPoints = size(x_input, 1);
numTrain = floor(0.8 * numDataPoints);
numVal = numDataPoints - numTrain;




% Training set
x_train = x_input(1:numTrain, :);
y_train = y_input(1:numTrain);
    
% normalize the needed feature 

% Train the SVM regression model on the training data
%SVMModel = fitrsvm(x_train, y_train, 'Standardize', true, 'KernelFunction', ker_func);

% Validate the model on the validation data
%y_pred = predict(SVMModel, x_input);


% Train the Random Forest regression model on the training data
RFModel = TreeBagger(numTrees, x_train, y_train, 'Method', 'regression');

% Validate the model on the entire dataset
y_pred = predict(RFModel, x_input);
% Calculate RMSE and NRMSE for validation data
RMSE_val = rms(y_pred - y_input);
normalizing_term_val = sqrt(mean(y_input.^2));
NRMSE_val = RMSE_val / normalizing_term_val;
    
%% % testing
data=load(test_file);
test_data=data.test_data;
    
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
end