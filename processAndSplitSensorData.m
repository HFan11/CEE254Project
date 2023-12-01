function sensorDataCells = processAndSplitSensorData(train_data)
    % This function processes the 'train_data' table by counting the number of 
    % unique sensors based on their latitude and longitude, and then splitting 
    % the data for each sensor. It returns a cell array where each element
    % directly contains the data for a specific sensor.

    % Round the position data for grouping
    train_data.lat=round(train_data.lat,3);
    train_data.lon=round(train_data.lon,3);
    
    % Extract latitude and longitude
    latitudes = train_data.lat;
    longitudes = train_data.lon;

    % Find unique combinations of latitude and longitude
    uniqueLocations = unique([latitudes, longitudes], 'rows');

    % Count the number of sensors
    numSensors = size(uniqueLocations, 1);

    % Initialize cell array to hold data for each sensor
    sensorDataCells = cell(1, numSensors);

    % Split the data for each unique location
    for i = 1:numSensors
        lat = uniqueLocations(i, 1);
        lon = uniqueLocations(i, 2);
        sensorDataCells{i} = train_data(latitudes == lat & longitudes == lon, :);
    end
end