function pred_pm2d5=grp_model(train_data,test_data,problem_type)
width=50; % width of moving mean window
indexes=[2,3,4,5,6,7]; % indexes of columns averaged
dt = 10;% intervals of averaging
input_cols=[1,2,6,7]; % indexes of input columns
test_cols=[1,2,5,6]; % indexes of test columns
ker_func='exponential';
if problem_type==3
    dt_test=5;
else
    dt_test=60;
end

if problem_type~=3

train_data=processAndSplitSensorData(train_data);
x_input=[];
y_input=[];
x_test=[];

% preprocess
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
[x_n,C,S] = normalize(x_input,'range');% normalize input x

% fitting
mdlgpr=fitrgp(x_n,y_input,'KernelFunction',ker_func);% fitting
y_pred=predict(mdlgpr,x_n);
RMSE=rms(y_pred-y_input);  

% testing
clear x;
clear time;

test_data=processAndSplitSensorData(test_data);
for i=1:length(test_data)
    clear test_data_;
    test_data_ = missing_filling(test_data{i},dt_test);
    for col=test_cols
        if col==1
            time{i}{col} = test_data_.time;
            x{i}{col}= (datenum(time{i}{col}))*24*60*60;
        
        else
            x{i}{col}=test_data_{:,col};
        end
    end
    x_test=cat(1,x_test,cat(2,x{i}{:}));
    
end

x_test_n=(x_test-C)./S;
pred_pm2d5=predict(mdlgpr,x_test_n);

% for interpolation problem
else


train_data=processAndSplitSensorData(train_data);
x_input=[];
y_input=[];
x_test=[];

% preprocess
for i=1:length(train_data)
    clear train_data_;
    train_data_=remove_outlier(train_data{i});
    train_data_=averageByTimeInterval(train_data_,dt);
    train_data_ = missing_filling_for_inter(train_data_,dt);
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
[x_n,C,S] = normalize(x_input,'range');% normalize input x

% fitting
mdlgpr=fitrgp(x_n,y_input,'KernelFunction',ker_func);% fitting
y_pred=predict(mdlgpr,x_n);
RMSE=rms(y_pred-y_input);  

% testing
clear x;
clear time;

test_data=processAndSplitSensorData_for_inter_test(test_data);
for i=1:length(test_data)
    clear test_data_;
    test_data_ = missing_filling_for_inter_test(test_data{i},dt_test);
    
    for col=test_cols
        if col==1
            time{i}{col} = test_data_.time;
            x{i}{col}= (datenum(time{i}{col}))*24*60*60;
        
        else
            x{i}{col}=test_data_{:,col};
        end
    end
    x_test=cat(1,x_test,cat(2,x{i}{:}));
    
end

x_test_n=(x_test-C)./S;
pred_pm2d5=predict(mdlgpr,x_test_n);
end

