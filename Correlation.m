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

%% Check correlation for static Table
staticTable{:,"Numtime"} = datenum(staticTable{:,"time"});
staticTable= removevars(staticTable,{'time'});
staticTable = movevars(staticTable,"Numtime",'Before',"humidity");

%corrplot(staticTable)

%% Check correlation for mobile Table
mobileTable{:,"Numtime"} = datenum(mobileTable{:,"time"});
mobileTable= removevars(mobileTable,{'time'});
mobileTable = movevars(mobileTable,"Numtime",'Before',"humidity");

%corrplot(mobileTable)

%% Check correlation for all Table
totalTable = vertcat(staticTable, mobileTable);
corrplot(totalTable)
