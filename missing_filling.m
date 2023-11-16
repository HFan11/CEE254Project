function missing_filled = missing_filling(inputTable, intervalMinutes)
% This is a function for filling missing data
intervalMinutes=minutes(intervalMinutes);
inputTable=table2timetable(inputTable);
missing_filled = retime(inputTable,'regular','pchip','TimeStep',intervalMinutes);
missing_filled = timetable2table(missing_filled);
end