import pandas as pd
import requests as r
import time


# Function to convert json response from API into pandas dataframe with only pm2.5 and datetime
def to_dataframe(data: list):
    dfs = []
    for k, point in enumerate(data):
        s1 = pd.Series({'pm25': point['value'], 'date': point['date']['utc']})
        dfs.append(s1)

    output = pd.concat(dfs, join='inner', axis=1).transpose()
    return output


# Define set of AQ stations
# 725 = Santiago, Chile ~ 94.46k data points
# 2009 = San Francisco, USA ~ 82.34k data points
# 2549 = Chennai, India ~ 31.79k data points
# 3455 = Toulouse, France ~ 73.82k data points
# 10753 = Melbourne, Australia ~ 30.50k data points
# 9764 = Accra, Ghana ~ 32.75k data points
aq_stations = [[725, 'Santiago'], [2009, 'SF'], [2549, 'Chennai'], [3455, 'Toulouse'],
               [10753, 'Melbourne'], [9764, 'Accra']]

for el in aq_stations:
    # Define API query parameters
    query = {'date_from': '2015-01-01T00:00:00+00:00', 'date_to': '2021-09-01T00:00:00+00:00', 'parameter': 'pm25',
             'location_id': el[0], 'limit': 100000, 'sort': 'asc'}

    # Query API and retry if 1st query fails
    response = r.get("https://u50g7n0cbj.execute-api.us-east-1.amazonaws.com/v2/measurements", params=query)
    if response.status_code != 200:
        time.sleep(10.0)
        response = r.get("https://u50g7n0cbj.execute-api.us-east-1.amazonaws.com/v2/measurements", params=query)

    # Parse API response
    response = response.json()['results']

    # Convert API response as a dataframe
    if response:
        df_temp = to_dataframe(response)

    # Save dataframe
    df_temp.to_pickle(f'dataset/air_quality_data/{el[1]}.pkl.zip')

    time.sleep(0.05)
