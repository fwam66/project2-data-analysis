import pandas as pd

household = pd.read_csv('household_vista_2023_2024.csv', index_col = 0)
trips = pd.read_csv('trips_vista_2023_2024.csv', index_col = 0)

## Data Cleaning ##
# Removes missing data and data that won't contribute to analysis
household = household[household['hhinc_group'].notna()]
household = household.drop(columns = ['surveyperiod'])

trips = trips.drop(columns = [
        'tripno', 'linkmode', 'dist1', 'dist2', 'dist3','dist4', 
        'dist5', 'dist6', 'dist7', 'dist8', 'dist9', 'starthour', 
        'arrhour', 'origplace2', 'origpurp2', 'destplace2', 'destpurp2',
        'time1', 'time2', 'time3', 'time4', 'time5', 'time6', 'time7',
        'time8', 'time9'
        ])

print(trips['mode2'].unique())

## Feature Engineering ##
# Split hhinc columns into Weekly and Annual income (easier to read and understand)
household[['weekly_hhinc_group', 'annual_hhinc_group']] = household['hhinc_group'].str.split('(', n=1, expand=True)
household['annual_hhinc_group'] = household['annual_hhinc_group'].str.strip('()')
household['weekly_hhinc_group'] = household['weekly_hhinc_group'].str.strip(' ')
household = household.drop(columns=['hhinc_group'])

# Categorise trip's main mode of transport as either Public, Private, Active, Hired or Other
mode_map = {
    'Vehicle Driver' : 'Private', 'Vehicle Passenger' : 'Private', 
    'Walking' : "Active", 'Bicycle' : "Active", 'School Bus' : 'Public',
    'Rideshare Service' : 'Hired', 'Motorcycle': 'Private', 'Plane' : 'Other', 
    'Taxi': 'Hired', 'Other': 'Other', 'Running/jogging' : "Active",
    'Mobility Scooter': 'Private', 'Tram':'Public', 'e-Scooter': "Active", 
    'Public Bus':'Public', 'Train':'Public'
    }

mode_columns = [
    'mode1', 'mode2', 'mode3', 'mode4', 'mode5', 'mode6',
    'mode7', 'mode8','mode9'
    ]

for col in mode_columns:
    trips[col] = trips[col].map(mode_map)

main_modes = ['Public', 'Hired', 'Private', 'Active', 'Other']

def find_main_mode(row):
    for mode in main_modes:
        if mode in row.values:
            return mode
    return 'nan'

trips['mainmode'] = trips[mode_columns].apply(find_main_mode, axis=1)
trips = trips.drop(columns=mode_columns)


## Encoding ##
# Label encode income groups
weekly_encode = {
    '$1-$149' : 0,  '$150-$299' : 1,  '$300-$399' : 2,
    '$400-$499' : 3,  '$500-$649' : 4,  '$650-$799' : 5,
    '$800-$999' : 6,  '$1,000-$1,249' : 7,  '$1,250-$1,499' : 8,
    '$1,500-$1,749' : 9,  '$1,750-$1,999' : 10, '$2,000-$2,499' : 11,
    '$2,500-$2,999' : 12, '$3,000-$3,499' : 13, '$3,500-$3,999' : 14,
    '$4,000-$4,499' : 15, '$4,500-$4,999' : 16, '$5,000-$5,999' : 17,
    '$6,000-$7,999' : 18, '$8,000 or more' : 19
    }

annual_encode = { 
    '$1-$7,799' : 0,  '$7,800-$15,599' : 1,  '$15,600-$20,799' : 2,
    '$20,800-$25,999' : 3,  '$26,000-$33,799' : 4,  '$33,800-$41,599' : 5,
    '$41,600-$51,999' : 6,  '$52,000-$64,999' : 7,  '$65,000-$77,999' : 8,
    '$78,000-$90,999' : 9,  '$91,000-$103,999' : 10, '$104,000-$129,999' : 11,
    '$130,000-$155,999' : 12, '$156,000-$181,999' : 13, '$182,000-$207,999' : 14,
    '$208,000-$233,999' : 15, '$234,000-$259,999' : 16, '$260,000-$311,999' : 17,
    '$312,000-$415,999' : 18, '$416,000 or more' : 19
    }

household['annual_hhinc_group'] = household['annual_hhinc_group'].map(annual_encode)
household['weekly_hhinc_group'] = household['weekly_hhinc_group'].map(weekly_encode)

weight_columns = [
    'hhpoststratweight', 'hhpoststratweight_GROUP_1' , 'hhpoststratweight_GROUP_2',
    'hhpoststratweight_GROUP_3', 'hhpoststratweight_GROUP_4',	
    'hhpoststratweight_GROUP_5', 'hhpoststratweight_GROUP_6',
    'hhpoststratweight_GROUP_7', 'hhpoststratweight_GROUP_8',
    'hhpoststratweight_GROUP_9', 'hhpoststratweight_GROUP_10'
    ]

trip_columns = [
    'trippoststratweight', 'trippoststratweight_GROUP_1' , 'trippoststratweight_GROUP_2',
    'trippoststratweight_GROUP_3', 'trippoststratweight_GROUP_4',	
    'trippoststratweight_GROUP_5', 'trippoststratweight_GROUP_6',
    'trippoststratweight_GROUP_7', 'trippoststratweight_GROUP_8',
    'trippoststratweight_GROUP_9', 'trippoststratweight_GROUP_10'   
    ]


# Second dataframe to hold household and trip weights
household_weights = household[weight_columns]
household = household.drop(columns=weight_columns)

trip_weights = trips[trip_columns]
trips = trips.drop(columns=trip_columns)


# Write cleaned data into CSV format
household.to_csv('cleaned_household_vista.csv')
household_weights.to_csv('cleaned_household_weights_vista.csv', index =False)
trips.to_csv('cleaned_trips.csv')
