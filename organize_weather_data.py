import numpy as np
import pandas as pd
import os


files = os.listdir('dataset/daily_weather_preprocessed/')

toulouse = []
chennai = []
melbourne = []
santiago = []
accra = []
sf = []
columns_to_drop = ['STATION', 'NAME', 'TEMP_ATTRIBUTES', 'DEWP_ATTRIBUTES', 'SLP_ATTRIBUTES', 'STP_ATTRIBUTES',
                   'VISIB_ATTRIBUTES', 'WDSP_ATTRIBUTES', 'MAX_ATTRIBUTES', 'MIN_ATTRIBUTES', 'PRCP_ATTRIBUTES']

column_rename = {'DEWP': 'dewpoint', 'SLP': 'sea level pressure', 'STP': 'local pressure', 'VISIB': 'visibility',
                 'WDSP': 'windspeed', 'MXSPD': 'max windspeed', 'MAX': 'max temp', 'MIN': 'min temp',
                 'PRCP': 'precipitation', 'FRSHTT': 'weather'}

for f in files:
    city, year = f.replace('.', '-').split('-')[:2]
    temp = pd.read_csv('dataset/daily_weather_preprocessed/' + f, parse_dates=[1])
    temp = temp.drop(columns=columns_to_drop)
    temp = temp.rename(columns=column_rename)

    if city == 'Accra':
        accra.append(temp)
    elif city == 'Toulouse':
        toulouse.append(temp)
    elif city == 'Chennai':
        chennai.append(temp)
    elif city == 'SF':
        sf.append(temp)
    elif city == 'Chile':
        santiago.append(temp)
    elif city == 'Melbourne':
        melbourne.append(temp)
    else:
        continue

toulouse = pd.concat(toulouse, axis=0, ignore_index=True).sort_values(by='DATE').reset_index(drop=True)
sf = pd.concat(sf, axis=0, ignore_index=True).sort_values(by='DATE').reset_index(drop=True)
accra = pd.concat(accra, axis=0, ignore_index=True).sort_values(by='DATE').reset_index(drop=True)
melbourne = pd.concat(melbourne, axis=0, ignore_index=True).sort_values(by='DATE').reset_index(drop=True)
chennai = pd.concat(chennai, axis=0, ignore_index=True).sort_values(by='DATE').reset_index(drop=True)
santiago = pd.concat(santiago, axis=0, ignore_index=True).sort_values(by='DATE').reset_index(drop=True)

na1 = dict.fromkeys(['TEMP', 'dewpoint', 'sea level pressure', 'local pressure', 'max temp', 'min temp'], 9999.9)
na2 = dict.fromkeys(['visibility', 'windspeed', 'max windspeed', 'gust', 'SNDP'], 999.9)
na3 = dict.fromkeys(['precipitation'], 99.99)
for element in [na1, na2, na3]:
    toulouse = toulouse.replace(element, np.nan)
    sf = sf.replace(element, np.nan)
    accra = accra.replace(element, np.nan)
    melbourne = melbourne.replace(element, np.nan)
    chennai = chennai.replace(element, np.nan)
    santiago = santiago.replace(element, np.nan)

toulouse = toulouse.dropna(how='all')
sf = sf.dropna(how='all')
accra = accra.dropna(how='all')
melbourne = melbourne.dropna(how='all')
chennai = chennai.dropna(how='all')
santiago = santiago.dropna(how='all')

toulouse.to_pickle('dataset/daily_weather_postprocessed/toulouse.pkl')
sf.to_pickle('dataset/daily_weather_postprocessed/sf.pkl')
accra.to_pickle('dataset/daily_weather_postprocessed/accra.pkl')
melbourne.to_pickle('dataset/daily_weather_postprocessed/melbourne.pkl')
chennai.to_pickle('dataset/daily_weather_postprocessed/chennai.pkl')
santiago.to_pickle('dataset/daily_weather_postprocessed/santiago.pkl')
