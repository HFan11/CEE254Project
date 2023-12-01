function averagedTable = averageByTimeInterval(inputTable, intervalMinutes)
    % Ensure the 'time' column is in datetime format
    if ~isdatetime(inputTable.time)
        inputTable.time = datetime(inputTable.time, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
    end
    
    % Validate the intervalMinutes input
    if ~isscalar(intervalMinutes) || intervalMinutes <= 0
        error('intervalMinutes must be a positive scalar value.');
    end

    % Round down the time to the nearest specified interval
    roundedTime = dateshift(inputTable.time, 'start', 'minute') - minutes(mod(minute(inputTable.time), intervalMinutes));
    inputTable.time = roundedTime;

    % Define the variable names to average
    varsToAverage = inputTable.Properties.VariableNames(2:end); % Exclude 'time'

    % Group the data by the rounded time and calculate the mean for each group
    % using the varfun function
    averagedData = varfun(@mean, inputTable, 'GroupingVariables', 'time', 'InputVariables', varsToAverage);

    % Remove the 'GroupCount' if present
    averagedData.Properties.VariableDescriptions{1} = ''; % Clean up description
    averagedData = removevars(averagedData, contains(averagedData.Properties.VariableNames, 'GroupCount'));
    
    % Rename the averaged variables to remove 'mean_' prefix
    for i = 1:length(varsToAverage)
        oldVarName = ['mean_' varsToAverage{i}];
        if ismember(oldVarName, averagedData.Properties.VariableNames)
            averagedData.Properties.VariableNames{oldVarName} = varsToAverage{i};
        end
    end

    % Return the averaged table
    averagedTable = averagedData;
end
