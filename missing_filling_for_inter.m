function missing_filled = missing_filling_for_inter(inputTable, intervalMinutes)
% This is a function for filling missing data
intervalMinutes=minutes(intervalMinutes);
inputTable=table2timetable(inputTable);
missing_filled = retime(inputTable,'regular','pchip','TimeStep',intervalMinutes);
missing_filled = timetable2table(missing_filled);
num_time=datenum(missing_filled.time);
%leave out the interpolation period
first_day=floor(num_time(1));
low_bound=first_day*24*60*60+24*60*60+11*60*60+30*60;
up_bound =first_day*24*60*60+24*60*60+12*60*60+25*60;
num_time=num_time*24*60*60;
del_index1=find(num_time>low_bound);
del_index2=find(num_time<up_bound);
del_index=intersect(del_index1,del_index2);
missing_filled(del_index,:)=[];

end