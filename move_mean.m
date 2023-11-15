function y=move_mean(data,width,columns)
% this is a function that do moving mean, columns is a an array of indexes
% of the columns need doing means
for index_column =1:length(columns)
column=data{:,columns(index_column)};
data{:,columns(index_column)}=movmean(column,width);
end
y=data;
end



