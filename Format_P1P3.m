clear
load Data\short_term_foshan_train_val.mat

staticTable = data_static{1};
for i = 2: length(data_static)
    staticTable = vertcat(staticTable, data_static{i});
end
mobileTable = data_mobile{1};
for i = 2: length(data_mobile)
    mobileTable = vertcat(mobileTable, data_mobile{i});
end
disp("Loaded both static and mobile data")
% Combine everythin into R x 7 matrix
totalTable = vertcat(staticTable, mobileTable);


%% Data for Problem one

DataSize = 0.9;
TestSize = 0.1;

% Have data from Oct8 to Oct 15
selectedDays = [8, 10, 12];
for i = 1:length(selectedDays)
    start_time = datetime(2018,10, selectedDays(i)) + hours(selectedDays(i)); % randomize so that the 3 days period starts at 8am, 10am, 12pm respectively
    end_time = start_time + days(3);
    in_range = (totalTable.time >= start_time) & (totalTable.time <= end_time);
    withinDateT=totalTable(in_range,:);
    randomRowIndices = randperm(size(withinDateT,1), int64(size(withinDateT,1)*DataSize));
    train_data = withinDateT(randomRowIndices, :);
    %train_data = table2array(train_data);
    save( ['Data\P1_', num2str(selectedDays(i)), '.mat'], "train_data")


    % Working on the test and soln data
    start_time = end_time;
    end_time = start_time + hours(3); % 3 hour Data to predict
    in_range = (totalTable.time >= start_time) & (totalTable.time <= end_time);
    withinDateT=totalTable(in_range,:);
    randomRowIndices = randperm(size(withinDateT,1), int64(size(withinDateT,1)*TestSize));
    test_data = withinDateT(randomRowIndices, :);

    % Copy from HW1 of CEE254. To group data points by each hour
    date = datestr(test_data.time,'mm/dd/yyyy'); % Get only the date mmddyyyy from the 'time' variable
    [dayNumber,dayName] = weekday(datenum(date,'mm/dd/yyyy'));
    hour = datestr(test_data.time,'HH');
    data = addvars(test_data, date, dayName, hour,dayNumber, 'Before','time',...
    'NewVariableNames',{'date','dayName','hour','dayNumber'});
    data_gb_datehr = grpstats(data,{'date','hour'},{'mean'},'DataVars',{'hmd','spd','tmp','pm2d5','lat','lon'});
    data_gb_datehr = sortrows(data_gb_datehr);
    backToDatetime = datetime(data_gb_datehr.date, "InputFormat","MM/dd/yyyy");
    backToDatetime = backToDatetime + hours(str2num(data_gb_datehr.hour));
    data_gb_datehr = addvars(data_gb_datehr, backToDatetime, 'Before','date', 'NewVariableNames','datetime');
    data_gb_datehr.Properties.RowNames = {};
    test_data = removevars(data_gb_datehr,{'date','hour','GroupCount'});
    test_data = renamevars(test_data,["datetime","mean_hmd","mean_spd","mean_tmp","mean_pm2d5","mean_lat","mean_lon"], ...
                 ["time","hmd","spd","tmp","pm2d5","lat","lot"]);
    soln_data = test_data.pm2d5;
    save( ['Data\S1_', num2str(selectedDays(i)), '.mat'], "soln_data")
    test_data = removevars(test_data,{'pm2d5'});
    save( ['Data\T1_', num2str(selectedDays(i)), '.mat'], "test_data")


end

%% Data for Problem three
DataSize = 0.99;

% Have data from Oct8 to Oct 15
selectedDays = [8, 10, 12];
for i = 1:length(selectedDays)
    start_time = datetime(2018,10, selectedDays(i));
    end_time = start_time + days(3);
    in_range = (totalTable.time >= start_time) & (totalTable.time <= end_time);
    withinDateT=totalTable(in_range,:);

    start_time = start_time + days(1) + hours(11) + minutes(30);
    end_time = start_time + minutes(60);
    toDelete = (withinDateT.time >= start_time) & (withinDateT.time <= end_time);
    deletedData = withinDateT(toDelete,:);
    withinDateT(toDelete,:) = [];

    randomRowIndices = randperm(size(withinDateT,1), int64(size(withinDateT,1)*DataSize));
    train_data = withinDateT(randomRowIndices, :);
    save( ['Data\P3_', num2str(selectedDays(i)), '.mat'], "train_data")

    % Working on the test and soln data
    withinDateT = deletedData;
    randomRowIndices = randperm(size(withinDateT,1), int64(size(withinDateT,1)*TestSize));
    test_data = withinDateT(randomRowIndices, :);

    % Copy from HW1 of CEE254. To group data points by each 5 minute
    % TODO NOT implemented.

end

%% End of Program
disp("End of Program")
