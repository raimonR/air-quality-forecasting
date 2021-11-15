import xarray as xr
import pandas as pd
import os


files = os.listdir('dataset/drought_preprocessed')

munich = []
dhaka = []
melbourne = []
santiago = []
abidjan = []
oakland = []

set_list = [['Oakland', [-122.5, 37.5]], ['Santiago', [-70.5, -33.5]], ['Prague', [50.5, 14.5]],
            ['Abidjan', [-3.5, 5.5]], ['Dhaka', [90.5, 23.5]], ['Melbourne', [144.5, -37.5]],
            ['Sao Paulo', [-23.5, -46.5]], ['Anchorage', [61.5, -149.5]], ['Thembisa', [-25.5, 28.5]]]
for element in set_list:
    drought_data = []
    for f in files:
        date = f.split('.')[0]
        ds = xr.open_dataset(f'dataset/drought_preprocessed/{f}', engine='netcdf4', decode_times=False)
        ds = ds.to_dataframe()
        try:
            ds = ds['di_01']
        except KeyError:
            ds = ds['var1']

        lat_l, lat_r = element[1][0] - 2, element[1][0] + 2
        lon_l, lon_r = element[1][1] - 2, element[1][1] + 2
        ds = ds.loc[(lat_l, lon_l):(lat_r, lon_r)].mean()
        ds = pd.DataFrame(data=[[pd.Timestamp(date), ds]], columns=['Date', 'Drought Index'])
        drought_data.append(ds)

    drought_data = pd.concat(drought_data).reset_index(drop=True)
    drought_data.to_pickle(f'dataset/drought_postprocessed/{element[0]}.pkl')

