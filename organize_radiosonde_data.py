import numpy as np
import pandas as pd
import requests as r
from bs4 import BeautifulSoup
from calendar import Calendar
import time
import os
from scipy.interpolate import CubicSpline


def preprocess_arrays():
    data_list = [['Abidjan', 65578, 2020], ['Santiago', 85586, 2015], ['Oakland', 72493, 2016],
                 ['Melbourne', 94866, 2020], ['Munich', 10868, 2016], ['Dhaka', 41923, 2016]]
    for station, station_number, initial_date in data_list:
        columns = ['PRES', 'HGHT', 'TEMP', 'DWPT', 'RELH', 'MIXR', 'DRCT', 'SKNT', 'THTA', 'THTE', 'THTV']
        df = pd.DataFrame()
        df[columns] = np.zeros(len(columns))
        for i in range(initial_date, 2022):
            for j in range(1, 13):
                day = list(filter(lambda a: a != 0, Calendar().itermonthdays(year=i, month=j)))[-1]
                url = f'http://weather.uwyo.edu/cgi-bin/sounding?region=samer&TYPE=TEXT%3ALIST&YEAR={i}&MONTH={str(j).zfill(2)}&FROM=0100&TO={day}12&STNM={station_number}'

                html = r.get(url).text

                # soup = BeautifulSoup(html, 'html.parser')
                soup = BeautifulSoup(html, 'lxml')

                header = soup.find_all('h2')
                data = soup.find_all('pre')[0::2]

                if not data:
                    continue

                for points, info in zip(data, header):
                    header_string = info.text[-15:].split(' ')
                    hour = int(header_string[0].split('Z')[0])
                    day = int(header_string[1])
                    date = pd.Timestamp(year=i, month=j, day=day, hour=hour)

                    lines = points.text.split('\n')
                    for elements in lines[5:-1]:
                        elements = elements.split(' ')
                        elements = list(filter(None, elements))
                        df_temp = pd.DataFrame(data=dict(zip(df.columns, elements)), columns=df.columns, index=[
                            date], dtype=float)
                        df = df.append(df_temp)

                time.sleep(2)

        df = df.drop(labels=['DWPT', 'RELH', 'MIXR', 'DRCT', 'SKNT', 'THTA', 'THTE'], axis=1)
        df = df[df['HGHT'] <= 15000]
        df.to_pickle(f'dataset/radiosonde_preprocessed/{station}.pkl')


def postprocess_arrays():
    def get_consecutive(arr):
        return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)

    files = os.listdir('dataset/radiosonde_preprocessed/')
    for f in files:
        data = pd.read_pickle(f'dataset/radiosonde_preprocessed/{f}')
        df_temp = pd.DataFrame()
        for m in data.index.unique():
            if data.loc[m, 'HGHT'].size <= 1:
                continue

            z = np.arange(0, 15000, step=100)
            idx = np.where(np.diff(data.loc[m, 'HGHT'].values) > 0)[0]
            if idx.size == 0:
                continue
            else:
                idx = get_consecutive(idx)[0]

            h = data.loc[m, 'HGHT'].values[idx]
            pressure = data.loc[m, 'PRES'][idx].fillna(method='ffill').fillna(method='bfill')
            temperature = data.loc[m, 'TEMP'][idx].fillna(method='ffill').fillna(method='bfill')
            theta_v = data.loc[m, 'THTV'][idx].fillna(method='ffill').fillna(method='bfill')
            if pressure.isna().any() or temperature.isna().any() or theta_v.isna().any():
                continue

            pressure = CubicSpline(h, pressure)
            temperature = CubicSpline(h, temperature)
            theta_v = CubicSpline(h, theta_v)

            pressure_columns = ['PRES_vs_z_' + str(s) for s in z]
            temperature_columns = ['TEMP_vs_z_' + str(s) for s in z]
            theta_v_columns = ['THTV_vs_z_' + str(s) for s in z]

            columns = pressure_columns + temperature_columns + theta_v_columns
            values = pressure(z).tolist() + temperature(z).tolist() + theta_v(z).tolist()

            df_temp = df_temp.append(pd.DataFrame(dict(zip(columns, values)), index=[m]))

        df_temp.to_pickle(f'dataset/radiosonde_postprocessed/{f}')


# preprocess_arrays()
postprocess_arrays()
