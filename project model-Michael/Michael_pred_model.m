function pred_pm2d5 = Michael_pred_model(train_file, test_file)
    clear all
    %% parameterized
    width=30; % width of moving mean window
    indexes=[2,3,4,5,6,7]; % indexes of columns averaged
    dt = 5;% intervals of averaging
    input_cols=[1,2,3,4,5,6]; % indexes of input columns
    test_cols=[1,2,3,4,5,6]; % indexes of test columns
    ker_func='rbf';
    train_file='P3_12.mat';
    test_file='T3_12.mat';
    ground_true_file='S3_12.mat';
    %x=data_static{5}.time;
    %y=data_static{5}.pm2d5;
    %figure(1);
    %plot(x,y)
    
    data=load(train_file);
    train_data=data.train_data;
    %train_data=data_mobile;
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
    
    %% implement the prediction model
    
    % Train the SVM regression model on the training data
    SVMModel = fitrsvm(x_input, y_input, 'Standardize', true, 'KernelFunction', ker_func);
    
    % Validate the model on the validation data
    y_pred = predict(SVMModel, x_input);
    
    RMSE_train=rms(y_pred-y_input);  
    normalizing_term = sqrt(mean(y_input.^2));
    NRMSE_train=RMSE_train/ normalizing_term
    
    
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
end

