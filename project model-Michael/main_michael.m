clear all
%% parameterized
width=30; % width of moving mean window
indexes=[2,3,4,5,6,7]; % indexes of columns averaged
dt = 5;% intervals of averaging
input_cols=[1,2,3,4,5,6]; % indexes of input columns
test_cols=[1,2,3,4,5,6]; % indexes of test columns
ker_func='rbf';
train_file='separated_train_data_long_term_5_var.mat';
test_file='test_data_long_term_5_var.mat';
%x=data_static{5}.time;
%y=data_static{5}.pm2d5;
    %figure(1);
    %plot(x,y)
    
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
val_data_raw = raw_data(splitPoint+1:end);

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

% Validation set
for i = 1:length(val_data_raw)
    val_data_ = val_data_raw{i};
end

% Assuming val_data_raw is a cell array of tables/matrices
x_val = [];
y_val = [];

for i = 1:length(val_data_raw)
    val_data_ = val_data_raw{i}; % Get the i-th dataset

    % If the first column is time and needs to be converted
    if isdatetime(val_data_.time)
        time_val = (datenum(val_data_.time)) * 24 * 60 * 60;
    else
        time_val = val_data_.time; % If already in numeric format
    end

    % Extracting features - assuming features are in columns 1 to 6
    x_val_i = [time_val, val_data_{:, 2:6}];

    % Extracting target variable - assuming it's in column 5
    y_val_i = val_data_{:, 5};

    % Concatenating with previous data
    x_val = [x_val; x_val_i];
    y_val = [y_val; y_val_i];
end

% Train the SVM regression model on the training data
SVMModel = fitrsvm(x_train, y_train, 'Standardize', true, 'KernelFunction', ker_func);

% Validate the model on the validation data
y_pred_val = predict(SVMModel, x_val);

% Calculate RMSE and NRMSE for validation data
RMSE_val = rms(y_pred_val - y_val);
normalizing_term_val = sqrt(mean(y_val.^2));
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
y_pred_test=predict(SVMModel,x_test);
pred_pm2d5=y_pred_test