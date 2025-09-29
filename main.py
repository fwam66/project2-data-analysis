import pandas as pd

# Read in CSV files
household = pd.read_csv('household_vista_2023_2024.csv')
trips = pd.read_csv('trips_vista_2023_2024.csv')

# Removes missing data that won't contribute to analysis
household = household[household['hhinc_group'].notna()]

# Split hhinc columns into Weekly and Annual income
# Easier to read and understand
household[['weekly_hhinc_group', 'annual_hhinc_group']] = household['hhinc_group'].str.split(' ', n=1, expand=True)
household['annual_hhinc_group'] = household['annual_hhinc_group'].str.strip('()')
household = household.drop(columns=['hhinc_group'])

# Label encode income groups
print(household['weekly_hhinc_group'].unique())


