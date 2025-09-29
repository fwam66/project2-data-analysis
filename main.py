import pandas as pd

# Read in CSV files
household = pd.read_csv('household_vista_2023_2024.csv', index_col = 0)
trips = pd.read_csv('trips_vista_2023_2024.csv', index_col = 0)

# Removes missing data and data that won't contribute to analysis
household = household[household['hhinc_group'].notna()]
household = household.drop(columns = ['surveyperiod'])

# Split hhinc columns into Weekly and Annual income (easier to read and understand)
household[['weekly_hhinc_group', 'annual_hhinc_group']] = household['hhinc_group'].str.split('(', n=1, expand=True)
household['annual_hhinc_group'] = household['annual_hhinc_group'].str.strip('()')
household['weekly_hhinc_group'] = household['weekly_hhinc_group'].str.strip(' ')
household = household.drop(columns=['hhinc_group'])

# Label encode income groups
weekly_encode = {'$1-$149' : 0,  '$150-$299' : 1,  '$300-$399' : 2,
                 '$400-$499' : 3,  '$500-$649' : 4,  '$650-$799' : 5,
                 '$800-$999' : 6,  '$1,000-$1,249' : 7,  '$1,250-$1,499' : 8,
                 '$1,500-$1,749' : 9,  '$1,750-$1,999' : 10, '$2,000-$2,499' : 11,
                '$2,500-$2,999' : 12, '$3,000-$3,499' : 13, '$3,500-$3,999' : 14,
                '$4,000-$4,499' : 15, '$4,500-$4,999' : 16, '$5,000-$5,999' : 17,
                '$6,000-$7,999' : 18, '$8,000 or more' : 19}

annual_encode = { '$1-$7,799' : 0,  '$7,800-$15,599' : 1,  '$15,600-$20,799' : 2,
                 '$20,800-$25,999' : 3,  '$26,000-$33,799' : 4,  '$33,800-$41,599' : 5,
                 '$41,600-$51,999' : 6,  '$52,000-$64,999' : 7,  '$65,000-$77,999' : 8,
                 '$78,000-$90,999' : 9,  '$91,000-$103,999' : 10, '$104,000-$129,999' : 11,
                '$130,000-$155,999' : 12, '$156,000-$181,999' : 13, '$182,000-$207,999' : 14,
                '$208,000-$233,999' : 15, '$234,000-$259,999' : 16, '$260,000-$311,999' : 17,
                '$312,000-$415,999' : 18, '$416,000 or more' : 19}

household['annual_hhinc_group'] = household['annual_hhinc_group'].map(annual_encode)
household['weekly_hhinc_group'] = household['weekly_hhinc_group'].map(weekly_encode)

weight_columns = ['hhpoststratweight', 'hhpoststratweight_GROUP_1' , 'hhpoststratweight_GROUP_2',
                'hhpoststratweight_GROUP_3', 'hhpoststratweight_GROUP_4',	
                'hhpoststratweight_GROUP_5', 'hhpoststratweight_GROUP_6',
                'hhpoststratweight_GROUP_7', 'hhpoststratweight_GROUP_8',
                'hhpoststratweight_GROUP_9', 'hhpoststratweight_GROUP_10']


# Second dataframe to hold household weights
household_weights = household[weight_columns]
household = household.drop(columns=weight_columns)

print(household[['dwelltype', 'owndwell']].drop_duplicates())

