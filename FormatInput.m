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

dataSize = [ 50000, 150000, 500000];


%% Data for Problem one
start_time = datetime(2018,10,8);
end_time = datetime(2018,10,11);
in_range = (totalTable.time >= start_time) & (totalTable.time <= end_time);

withinDateT=totalTable(in_range,:);

for i = 1:length(dataSize)
    randomRowIndices = randperm(size(withinDateT,1), dataSize(i));
    train_data = withinDateT(randomRowIndices, :);
    save( ['Data\P1_1008_', num2str(dataSize(i)), '.mat'], "train_data")
end

%% Data for Problem one
start_time = datetime(2018,10,8);
end_time = datetime(2018,10,11);
in_range = (totalTable.time >= start_time) & (totalTable.time <= end_time);

withinDateT=totalTable(in_range,:);

for i = 1:length(dataSize)
    randomRowIndices = randperm(size(withinDateT,1), dataSize(i));
    train_data = withinDateT(randomRowIndices, :);
    save( ['Data\P1_1008_', num2str(dataSize(i)), '.mat'], "train_data")
end

start_time = datetime(2018,10,10);
end_time = datetime(2018,10,12);
in_range = (totalTable.time >= start_time) & (totalTable.time <= end_time);

withinDateT=totalTable(in_range,:);

for i = 1:length(dataSize)
    randomRowIndices = randperm(size(withinDateT,1), dataSize(i));
    train_data = withinDateT(randomRowIndices, :);
    save( ['Data\P1_1010_', num2str(dataSize(i)), '.mat'], "train_data")
end

%% Data for Problem two
start_time = datetime(2018,10,8);
end_time = datetime(2018,10,11);
in_range = (totalTable.time >= start_time) & (totalTable.time <= end_time);

withinDateT=totalTable(in_range,:);

for i = 1:length(dataSize)
    randomRowIndices = randperm(size(withinDateT,1), dataSize(i));
    train_data = withinDateT(randomRowIndices, :);
    save( ['Data\P1_1008_', num2str(dataSize(i)), '.mat'], "train_data")
end

%% Data for Problem three
start_time = datetime(2018,10,8);
end_time = datetime(2018,10,11);
in_range = (totalTable.time >= start_time) & (totalTable.time <= end_time);

withinDateT=totalTable(in_range,:);

for i = 1:length(dataSize)
    randomRowIndices = randperm(size(withinDateT,1), dataSize(i));
    train_data = withinDateT(randomRowIndices, :);
    save( ['Data\P1_1008_', num2str(dataSize(i)), '.mat'], "train_data")
end

%% End of Program
disp("End of Program")
