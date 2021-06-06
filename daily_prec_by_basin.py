# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

#######Process INMET Automatic Station Data###########
"""
After pre-processing INMET hourly weather data is compiled for each year. The target is to map the stations
to a state-basin curve, and then get the aggregated daily mean for the curves.
@author: Gabriel Marinho
1st Task - Process an individual folder, and loop in all folders, creating a database with station
location, precipitation, and date/hour information    
2nd Task - Map the station to points in the NMME moldes available from NOAA    
"""

stations = pd.read_excel(r'C:\Users\mariga\OneDrive - Verisk Analytics\Projeto Final\Data\weather_stations_inmet_with_basins.xlsx')
stations = stations[stations.Station_Ty =='Automatic']
all_dfs = []
data_folder = 'C:/Users/mariga/Documents/GitHub/Projeto Final/Data Wrangling Scripts/Year_Data'
for file in os.listdir(data_folder):
    df = pd.read_csv(data_folder+'/'+file, sep = ';', encoding='latin-1')
    df.precipitation = df.precipitation.apply(lambda x: x if x>=0 else 0)
    mapped_data = pd.merge(df, stations,how='left', left_on=['station_code'], right_on=['Inmet_Code'])
    mapped_data = mapped_data[['hour', 'day', 'month', 'year', 'precipitation', 'unit', 'state',
           'region', 'station_code', 'station_name', 'full_year_flag','hydro_shape']]
    mapped_data.groupby(by = ['day', 'month', 'year', 'precipitation', 'unit', 'state','region', 'station_code', 'station_name', 'full_year_flag','hydro_shape'], as_index=False).sum() 
    daily_data = mapped_data.groupby(by = ['day', 'month', 'year', 'precipitation', 'unit', 'state','region', 'station_code', 'station_name', 'full_year_flag','hydro_shape'], as_index=False).sum()
    daily_data.drop(columns = ['hour'], inplace=True)
    daily_data = daily_data.groupby(by = ['day', 'month', 'year', 'unit', 'hydro_shape'], as_index=False).mean()
    daily_data = daily_data[daily_data.hydro_shape.apply(lambda x: False if x.find('IGNORE') != -1 else True)]
    all_dfs.append(daily_data)

final_data = pd.concat(all_dfs)
final_data.reset_index(drop = True, inplace = True)