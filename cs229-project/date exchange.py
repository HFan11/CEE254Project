import pandas as pd

# Load the dataset
file_path = r"C:\Users\Haoch\Desktop\cs229-project\Modified DataSet.csv"
df = pd.read_csv(file_path)

# Generate date range from '01/01/2016' to '12/31/2022'
date_range = pd.date_range(start='01/01/2016', end='12/31/2022')

# Ensure the dataset's length matches the date range
if len(df) != len(date_range):
    print(f"Warning: Length of dataset ({len(df)}) doesn't match expected length ({len(date_range)}). Ensure you have data for each date!")
else:
    # Fill in the Date and Day of the Week columns
    df['Date'] = date_range
    df['Day of the Week'] = date_range.day_name()

    # Save the modified dataframe back to the CSV file
    output_path = r"C:\Users\Haoch\Desktop\cs229-project\Modified DataSet.csv"
    df.to_csv(output_path, index=False)

    print("Modification complete and saved to:", output_path)





