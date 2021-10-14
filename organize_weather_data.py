import numpy as np
import pandas as pd
import os


files = os.listdir('dataset/daily_weather_preprocessed/')

munich = []
chennai = []
melbourne = []
santiago = []
abidjan = []
oakland = []
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

    if city == 'Abidjan':
        abidjan.append(temp)
    elif city == 'Munich':
        munich.append(temp)
    elif city == 'Chennai':
        chennai.append(temp)
    elif city == 'Oakland':
        oakland.append(temp)
    elif city == 'Santiago':
        santiago.append(temp)
    elif city == 'Melbourne':
        melbourne.append(temp)
    else:
        continue

munich = pd.concat(munich, axis=0, ignore_index=True).sort_values(by='DATE').reset_index(drop=True)
oakland = pd.concat(oakland, axis=0, ignore_index=True).sort_values(by='DATE').reset_index(drop=True)
abidjan = pd.concat(abidjan, axis=0, ignore_index=True).sort_values(by='DATE').reset_index(drop=True)
melbourne = pd.concat(melbourne, axis=0, ignore_index=True).sort_values(by='DATE').reset_index(drop=True)
chennai = pd.concat(chennai, axis=0, ignore_index=True).sort_values(by='DATE').reset_index(drop=True)
santiago = pd.concat(santiago, axis=0, ignore_index=True).sort_values(by='DATE').reset_index(drop=True)

na1 = dict.fromkeys(['TEMP', 'dewpoint', 'sea level pressure', 'local pressure', 'max temp', 'min temp'], 9999.9)
na2 = dict.fromkeys(['visibility', 'windspeed', 'max windspeed', 'GUST', 'SNDP'], 999.9)
na3 = dict.fromkeys(['precipitation'], 99.99)
for element in [na1, na2, na3]:
    munich = munich.replace(element, np.nan)
    oakland = oakland.replace(element, np.nan)
    abidjan = abidjan.replace(element, np.nan)
    melbourne = melbourne.replace(element, np.nan)
    chennai = chennai.replace(element, np.nan)
    santiago = santiago.replace(element, np.nan)

munich.to_pickle('dataset/daily_weather_postprocessed/Munich.pkl')
oakland.to_pickle('dataset/daily_weather_postprocessed/Oakland.pkl')
abidjan.to_pickle('dataset/daily_weather_postprocessed/Abidjan.pkl')
melbourne.to_pickle('dataset/daily_weather_postprocessed/Melbourne.pkl')
chennai.to_pickle('dataset/daily_weather_postprocessed/Chennai.pkl')
santiago.to_pickle('dataset/daily_weather_postprocessed/Santiago.pkl')
