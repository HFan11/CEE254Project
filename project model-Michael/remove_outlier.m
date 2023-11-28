function data_clean = remove_outlier(train_data)
    % Initialize the cleaned data table with the first column (time)
    indices=1:height(train_data);
    % Get the variable names
    variableNames = train_data.Properties.VariableNames;
    
    % Loop through each column, skipping the first one (time)
    for i = 2:width(train_data)
        columnName = variableNames{i};
        
        % Calculate the IQR for the current column
        Q1 = quantile(train_data.(columnName), 0.25);
        Q3 = quantile(train_data.(columnName), 0.75);
        IQR = Q3 - Q1;

        % Define the outlier criteria
        upper_bound = Q3 + 1.5 * IQR;
        % Set the lowest possible value according to domain knowledge
        lower_bound = Q1 - 1.5 * IQR; 

        % Remove outliers and add the cleaned column to the 'data_clean' table
        non_outlier_indices{i-1} = find(train_data.(columnName) >= lower_bound & train_data.(columnName) <= upper_bound);
        indices=intersect(non_outlier_indices{i-1},indices);
    end
    data_clean=train_data(indices,:);
    % Optionally: save to a .mat file
    %cleaned_var_name = ['cleaned_', 'P2_1']; 
    %save([cleaned_var_name '.mat'], 'data_clean');
end