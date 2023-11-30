% 2023/11/27
clear;
for problem_type = 1:3
    if problem_type == 1
        problem_name = 'short_term';
    elseif problem_type == 2
        problem_name = 'long_term';
    elseif problem_type == 3
        problem_name = 'interpolation';
    end

    for var_level = [0,5,10]
        train_data = load(['separated_train_data_',problem_name,'_',...
                          num2str(var_level),'_var.mat']).train_data_static;
        test_data = load(['test_data_',problem_name,'_',...
                           num2str(var_level),'_var.mat']).test_data;
        pred_pm2d5 = pm2d5_pred_model(train_data, test_data, problem_type);
        save([problem_name,'_',num2str(var_level),'.mat'],'pred_pm2d5');
        clear pred_pm2d5;
    end
end

