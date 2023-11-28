function missing_filled = missing_filling_for_inter_test(inputTable, intervalMinutes)
% This is a function for filling missing data
num_time=datenum(inputTable.time);
h=height(inputTable);
inputTable=table2timetable(inputTable);


first_day=floor(num_time(1));
low_bound=first_day*24*60*60+11*60*60+30*60;
up_bound =first_day*24*60*60+12*60*60+25*60;
dates=linspace(low_bound,up_bound,h);
dates=datestr(dates/24/3600,'yyyy-mm-dd HH:MM:SS');
dates=datetime(dates);
missing_filled = retime(inputTable,dates,'pchip');
dates=low_bound:(intervalMinutes*60):up_bound;
dates=datestr(dates/24/3600,'yyyy-mm-dd HH:MM:SS');
dates=datetime(dates);
missing_filled = retime(missing_filled,dates,'pchip');
missing_filled = timetable2table(missing_filled);

end