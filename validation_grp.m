function nrmse=validation_grp(train_data,test_data,problem_type)
 
width=50; % width of moving mean window
indexes=[2,3,4,5,6,7]; % indexes of columns averaged
dt = 30;% intervals of averaging
input_cols=[1,2,6,7]; % indexes of input columns
test_cols=[1,2,5,6]; % indexes of test columns
ker_func='exponential';
val_thres=0.85;
if problem_type==3
    dt_test=5;
else
    dt_test=60;
end

train_data=processAndSplitSensorData(train_data);
x_input=[];
y_input=[];
x_val_input=[];
y_val_input=[];

% preprocess
for i=1:length(train_data)
    clear train_data_;
    val_data=train_data{i};
    train_data_=remove_outlier(train_data{i});
    train_data_=averageByTimeInterval(train_data_,dt);
    train_data_ = missing_filling(train_data_,dt);
    train_data_=move_mean(train_data_,width,indexes);
    thre_1=floor(val_thres*height(train_data_));
    thre_2=floor(val_thres*height(val_data));
    for col=input_cols
        if col==1
            time_t{i}{col} = train_data_.time(1:thre_1);
            x_t{i}{col}= (datenum(time_t{i}{col}))*24*60*60;
            time_val{i}{col} = val_data.time(thre_2:end);
            x_val{i}{col}= (datenum(time_val{i}{col}))*24*60*60;

        else
            x_t{i}{col}=train_data_{1:thre_1,col};
            x_val{i}{col}=val_data{thre_2:end,col};
        end
    end
    x_input=cat(1,x_input,cat(2,x_t{i}{:}));
    y_{i}=train_data_{1:thre_1,5};
    y_input=cat(1,y_input,y_{i});
    x_val_input=cat(1,x_val_input,cat(2,x_val{i}{:}));
    y_val{i}=val_data{thre_2:end,5};
    y_val_input=cat(1,y_val_input,y_val{i});
end
[x_n,C,S] = normalize(x_input,'range');% normalize input x

% fitting
mdlgpr=fitrgp(x_n,y_input,'KernelFunction',ker_func);% fitting


% validation

x_val_n=(x_val_input-C)./S;
val_pm2d5=predict(mdlgpr,x_val_n);
size(val_pm2d5)
size(y_val_input)
size(x_n)
RMSE=rms(val_pm2d5-y_val_input); 
normalizing_term=sqrt(mean(y_val_input.^2));
nrmse=RMSE/normalizing_term;
figure(1);
plot(y_input,'r')


