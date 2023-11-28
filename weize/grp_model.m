function pred_pm2d5=grp_model(train_data,test_data,problem_type)
%% 
width=30; % width of moving mean window
indexes=[2,3,4,5,6,7]; % indexes of columns averaged
dt = 5;% intervals of averaging
input_cols=[1,2,6,7]; % indexes of input columns
test_cols=[1,2,5,6]; % indexes of test columns
ker_func='exponential';
%x=data_static{5}.time;
%y=data_static{5}.pm2d5;
%figure(1);
%plot(x,y)


%train_data=data_mobile;
train_data=processAndSplitSensorData(train_data);
x_input=[];
y_input=[];
x_test=[];
y_test=[];
%% % preprocess

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
mdlgpr=fitrgp(x_n,y_input,'KernelFunction',ker_func);% fitting
y_pred=predict(mdlgpr,x_n);
RMSE=rms(y_pred-y_input);    
%% % testing

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
x_test_n=(x_test-C)./S;
y_pred_test=predict(mdlgpr,x_test_n);
data=load(ground_true_file);
y_ground=data.soln_data;
RMSE=rms(y_pred_test-y_ground);
normalizing_term=sqrt(mean(y_ground).^2);
nrmse=RMSE/normalizing_term;
%% visualize

figure(1);
plot(y_ground)
hold on
plot(y_pred_test)
legend('y ground','y pred test')
hold off
end