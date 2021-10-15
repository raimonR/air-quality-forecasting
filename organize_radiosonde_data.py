import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests as r
from bs4 import BeautifulSoup
from calendar import Calendar
import time
import os


def preprocess_graphs():
    # data_list = [['Santiago', 85586, 2015], ['Oakland', 72493, 2016], ['Abidjan', 65578, 2020],
    #              ['Melbourne', 94866, 2020], ['Munich', 10868, 2016], ['Chennai', 43279, 2018]]
    data_list = [['Chennai', 43279, 2018]]
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

        df.to_pickle(f'dataset/radiosonde_preprocessed/graphs/{station}.pkl')


def preprocess_variables():
    # data_list = [['Santiago', 85586, 2015], ['Oakland', 72493, 2016], ['Abidjan', 65578, 2020],
    #              ['Melbourne', 94866, 2020], ['Munich', 10868, 2016], ['Chennai', 43279, 2018]]
    data_list = [['Chennai', 43279, 2018]]
    for station, station_number, initial_date in data_list:
        columns = ['showalter index', 'cape', 'ci', 'ml theta']
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
                data = soup.find_all('pre')[1::2]

                if not data:
                    continue

                for points, info in zip(data, header):
                    header_string = info.text[-15:].split(' ')
                    hour = int(header_string[0].split('Z')[0])
                    day = int(header_string[1])
                    date = pd.Timestamp(year=i, month=j, day=day, hour=hour)

                    lines = points.text.split('\n')
                    cols = ['showalter', 'Convective Available Potential Energy', 'Convective Inhibition',
                            'mean mixed layer potential temperature']
                    rows = []
                    for unit in cols:
                        row_size = len(rows)
                        rows.append([s for s in lines if unit.lower() in s.lower()])
                        if len(rows) - row_size == 0:
                            rows.append(0)

                    value = []
                    for elements in rows:
                        if len(elements) == 0:
                            value.append(0)
                        else:
                            elements = elements[0].split(' ')
                            elements = list(filter(None, elements))
                            value.append(elements[-1])

                    df_temp = pd.DataFrame(data=dict(zip(df.columns, value)), columns=df.columns, index=[
                        date], dtype=float)
                    df = df.append(df_temp)

                time.sleep(2)

        df.to_pickle(f'dataset/radiosonde_preprocessed/variables/{station}.pkl')


def postprocess_graphs():
    def index_loc(arr, value):
        array = np.asarray(arr)
        idx = (np.abs(array - value)).argmin()
        return idx

    def find_ml_height(h_arr, temp_arr):
        lr = -9.8/1e3
        surface_temp = np.mean(temp_arr[:2])
        points = lr*h_arr + surface_temp
        intersection = np.isclose(temp_arr, points, atol=2)
        if np.all(intersection == False):
            return 0
        else:
            intersection = np.argmax(intersection)
            return h_arr[intersection]

    files = os.listdir('dataset/radiosonde_preprocessed/graphs/')
    alt = 4000
    for f in files:
        df = []
        data = pd.read_pickle(f'dataset/radiosonde_preprocessed/graphs/{f}')
        measurements = data.index.unique()
        for index, m in enumerate(measurements.to_list()):
            i = index_loc(data.loc[m, 'HGHT'], alt)
            height = data.loc[m, 'HGHT'].iloc[:i + 2]
            temperature = data.loc[m, 'TEMP'].iloc[:i + 2]
            theta_v = data.loc[m, 'THTV'].iloc[:i + 2]
            theta_a = data.loc[m, 'THTA'].iloc[:i + 2]
            ws = data.loc[m, 'SKNT'].iloc[:i + 2]

            if height.min() >= 100:
                continue

            if np.all(height.isna()) or np.all(temperature.isna()) or np.all(theta_v.isna()) or np.all(ws.isna()):
                continue

            if theta_v.isna().sum()/theta_v.size > 0.3:
                continue

            tv_lapse_rate = theta_v[~np.isnan(theta_v)][:2].diff()[1]/height[~np.isnan(theta_v)][:2].diff()[1]*1e3
            temp_lapse_rate = temperature[~np.isnan(temperature)][:2].diff()[1]/height[~np.isnan(theta_v)][:2].diff()[
                1]*1e3
            if tv_lapse_rate < 0:
                ml_height = find_ml_height(height, temperature)
            else:
                if np.isnan(data.loc[m, 'THTV'][0]):
                    alpha = 9.81/data.loc[m, 'THTV'][1]
                else:
                    alpha = 9.81/data.loc[m, 'THTV'][0]
                br_num = alpha*np.diff(theta_v)*np.diff(height)/np.diff(ws)**2
                br_num = np.nan_to_num(br_num, nan=0, posinf=0, neginf=0)
                j = np.where(br_num > 1)[0]
                if j.size == 0:
                    ml_height = 0
                else:
                    j = j[0]
                    ml_height = height[1:][j]

            df.append(pd.DataFrame(data=[[tv_lapse_rate, temp_lapse_rate, ml_height]],
                columns=['theta_v lapse rate', 'temperature lapse rate', 'mixing layer height'], index=[m]))

        df = pd.concat(df)
        df.to_pickle(f'dataset/radiosonde_postprocessed/graphs/{f}')


def postprocess_variables():
    files = os.listdir('dataset/radiosonde_preprocessed/variables/')
    for f in files:
        data = pd.read_pickle(f'dataset/radiosonde_preprocessed/variables/{f}')
        data.to_pickle(f'dataset/radiosonde_postprocessed/variables/{f}')


preprocess_graphs()
preprocess_variables()
postprocess_graphs()
postprocess_variables()
