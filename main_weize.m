clear all;

problem_type=3;
var_level=10;

if problem_type == 1
    problem_name = 'short_term';
elseif problem_type == 2
    problem_name = 'long_term';
elseif problem_type == 3
    problem_name = 'interpolation';
end




train_file=['.\Data\test_phase\separated_train_data_',problem_name,'_',num2str(var_level),'_var.mat'];
test_file=['.\Data\test_phase\test_data_',problem_name,'_',num2str(var_level),'_var.mat'];
data=load(train_file);
train_data=data.train_data_static;
data=load(test_file);
test_data=data.test_data;


pred_pm2d5=grp_model(train_data,test_data,problem_type);
save([problem_name,'_',num2str(var_level),'.mat'],'pred_pm2d5');
%pred_pm2d5=validation_grp(train_data,test_data,problem_type);