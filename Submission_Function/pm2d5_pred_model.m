function pred_pm2d5 = pm2d5_pred_model(train_data, test_data, problem_type)
    if problem_type == 1
        pred_pm2d5 = grp_model(train_data, test_data, 1);
    elseif problem_type == 2
        pred_pm2d5 = Michael_pred_model(train_data, test_data, 2);
    elseif problem_type == 3
        pred_pm2d5 = grp_model(train_data, test_data, 3);
    end
end