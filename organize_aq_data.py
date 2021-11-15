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
# 725 = Santiago, Chile
# 2135 = Oakland, USA
# 8415 = Dhaka, Bangladesh
# 2051 = Newcastle, UK
# 10753 = Melbourne, Australia
# 9762 = Abidjan, Ivory Coast
# 5300 = Sao Paulo, Brazil
# 8101 = Cagliari, Italy
aq_stations = [[725, 'Santiago'], [2135, 'Oakland'], [8415, 'Dhaka'], [10753, 'Melbourne'],
               [9762, 'Abidjan'], [4369, 'Prague'], [293, 'Anchorage'], [5300, 'Sao Paulo']]

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
    df_temp.to_pickle(f'dataset/air_quality_data/{el[1]}.pkl')

    time.sleep(0.5)

df_temp = pd.read_excel('dataset/air_quality_data/Olifantsfontein.xlsx')
df_temp = df_temp.iloc[:-10, :]
dlist = []
for i in df_temp.iloc[:, 0].tolist():
    i = i.replace('24:00', '00:00')
    dlist.append(pd.Timestamp(i))

df_temp = pd.DataFrame(data={'pm25': df_temp['PM2.5'].to_numpy(), 'date': dlist})
df_temp.to_pickle(f'dataset/air_quality_data/Thembisa.pkl')
