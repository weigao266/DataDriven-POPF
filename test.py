df_4 = pd.read_csv('./wind_station_datasets/wind_station_4/95485-2008.csv', skiprows=3)
wind_data_4 = df_4['power (MW)']
wind_data_4 = wind_data_4.values
wind_data_4 = wind_data_4.reshape(-1, 1)