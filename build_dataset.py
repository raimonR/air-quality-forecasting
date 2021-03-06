import numpy as np
import pandas as pd


def build_dataset():
    # Get datasets for each city
    for city in ['Abidjan.pkl', 'Dhaka.pkl', 'Melbourne.pkl', 'Santiago.pkl', 'Oakland.pkl', 'Anchorage.pkl',
                 'Thembisa.pkl', 'Prague.pkl', 'Sao Paulo.pkl']:
        ds = ['air_quality_data', 'daily_weather_postprocessed', 'drought_postprocessed', 'radiosonde_postprocessed']
        merged_df = []
        for f in ds:
            if f == 'air_quality_data':
                aq_data = pd.read_pickle(f'./dataset/{f}/{city}')
                aq_data = aq_data.set_index('date')
                aq_data.index = pd.to_datetime(aq_data.index, utc=False)
                aq_data = aq_data.asfreq(freq='1H')
                aq_data.index = aq_data.index.tz_localize(None)
                aq_data['pm25'] = pd.to_numeric(aq_data['pm25'])
                aq_data[aq_data < 0] = np.nan
                aq_data = aq_data.interpolate(method='time', limit=4)

                merged_df.append(aq_data)

            elif f == 'daily_weather_postprocessed':
                weather_data = pd.read_pickle(f'./dataset/{f}/{city}')
                weather_data = weather_data.set_index('DATE')
                weather_data.index = pd.to_datetime(weather_data.index, utc=False)
                weather_data.loc[weather_data['GUST'].isna(), 'GUST'] = 0
                weather_data.loc[weather_data['SNDP'].isna(), 'SNDP'] = 0
                weather_data.loc[weather_data['sea level pressure'].isna(), 'sea level pressure'] = 1013.25
                weather_data.loc[weather_data['precipitation'].isna(), 'precipitation'] = 0
                for cols in weather_data.columns:
                    for month in range(1, 13):
                        intersection = weather_data[cols].isna() & (weather_data.index.month == month)
                        weather_data.loc[intersection, cols] = weather_data.loc[
                            weather_data.index.month == month, cols].mean()

                weather_data = weather_data.asfreq(freq='1H', method='ffill')
                weather_data.index = weather_data.index.tz_localize(None)

                merged_df.append(weather_data)

            elif f == 'drought_postprocessed':
                drought_data = pd.read_pickle(f'./dataset/{f}/{city}')
                drought_data = drought_data.set_index('Date')
                drought_data.index = pd.to_datetime(drought_data.index, utc=False)
                drought_data = drought_data.asfreq(freq='1H', method='ffill')
                drought_data.index = drought_data.index.tz_localize(None)

                merged_df.append(drought_data)

            elif f == 'radiosonde_postprocessed':
                radiosonde_data = pd.read_pickle(f'./dataset/{f}/{city}')
                radiosonde_data = radiosonde_data.asfreq(freq='1H', method='ffill')
                radiosonde_data.index = radiosonde_data.index.tz_localize(None)

                merged_df.append(radiosonde_data)

        for i in range(1, len(merged_df)):
            merged_df[0] = merged_df[0].merge(merged_df[i], how='inner', left_index=True, right_index=True)

        merged_df = merged_df[0]
        merged_df.to_pickle(f'./dataset/merged/{city.split(".")[0]}_merged_dataset.pkl')


build_dataset()
