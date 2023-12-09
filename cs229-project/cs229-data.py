import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Start date (1st of November 2021 at 12:00)
start_date = datetime(2021, 11, 1, 12)

# Time (hr) in days
days = 30
hours_per_day = 24

# Actual Flow: Sine wave to simulate water demand
amplitude = 600 # Difference between the peak and mean
mean_demand = 1000 # Average demand
actual_flow = mean_demand + amplitude * np.sin(2 * np.pi * np.arange(0, days, 1/hours_per_day))

# Generating the time labels in "yyyy-mm-dd-hh" format
time_labels = [(start_date + timedelta(hours=i)).strftime('%Y-%m-%d-%H') for i in range(days * hours_per_day)]

# Creating a DataFrame
df = pd.DataFrame({
    'Time (yyyy-mm-dd-hh)': time_labels,
    'Actual Flow (m^3/hr)': actual_flow
})

# Save to CSV
df.to_csv('water_consumption.csv', index=False)

