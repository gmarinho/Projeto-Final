# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

#######Process INMET Automatic Station Data###########
"""
INMET Weather data is available in folders for each year, containing a csv file 
for each automatic station available 
@author: Gabriel Marinho
1st Task - Process an individual folder, and loop in all folders, creating a database with station
location, precipitation, and date/hour information    
2nd Task - Map the station to points in the NMME moldes available from NOAA    
"""
"""The csv files can be divided in two parts, the first 7 rows containing the station metadata
and the rest, containing the observations.
The first part is following the patern below:
    
Region:	(CO,N,NE,S,S)
State_Abbr:	XX
Station_City:	e.g: BRASILIA
WMO Code:	AXXX
Latitude:	-15,78944444
Longitude:	-47,92583332
Height:	1159,54
Foundation date (YYYY-MM-DD):	YYYY-MM-DD
"""
"""
The Second part contains the following variables in observations:
    ['DATA (YYYY-MM-DD)', 'HORA (UTC)', 'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)',
       'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)',
       'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)',
       'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)',
       'RADIACAO GLOBAL (W/m²)',
       'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)',
       'TEMPERATURA DO PONTO DE ORVALHO (°C)',
       'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)',
       'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)',
       'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)',
       'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)',
       'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)',
       'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)',
       'UMIDADE RELATIVA DO AR, HORARIA (%)',
       'VENTO, DIREÇÃO HORARIA (gr) (° (gr))', 'VENTO, RAJADA MAXIMA (m/s)',
       'VENTO, VELOCIDADE HORARIA (m/s)']

null values are replaced by -9999

Only the variable Total Precipitation is going to be used at the first moment"""


parent_folder='C:/Users/mariga/OneDrive - Verisk Analytics/Projeto Final/Data/Weather/INMET'
for year_folder in os.listdir(parent_folder):
    if int(year_folder) < 2019:
        continue
        date_column_name = 'DATA (YYYY-MM-DD)'
        hour_column_name = 'HORA (UTC)'
    else:
        date_column_name = 'Data'
        hour_column_name = 'Hora UTC'        
    year_dfs = []
    for file in os.listdir(parent_folder+'/'+year_folder):
        split = file.split(sep = '_')
        region = split[1]
        state = split[2]
        station_code = split[3]
        station_name = split[4]
        date_begin = split[5]
        if('01-01-'+str(year_folder)!=date_begin):
            full_year = False
        else:
            full_year = True
        df = pd.read_csv(parent_folder+'/'+year_folder+'/'+file, encoding='latin-1', skiprows=8, sep=';', decimal=',')
        df = df[[date_column_name, hour_column_name, 'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)']]
        df['date'] = pd.to_datetime(df[date_column_name])
        df['hour'] = df[hour_column_name].apply(lambda x: int(x[:2]))
        df['day'] = df.date.dt.day
        df['month'] = df.date.dt.month
        df['year'] = df.date.dt.year
        df['precipitation'] = df['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)']
        df['unit'] = 'mm'
        df.drop(columns = [date_column_name, hour_column_name, 'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)','date'], inplace=True)
        df['state'] = state
        df['region'] = region
        df['station_code'] = station_code
        df['station_name'] = station_name
        df['full_year_flag'] = full_year
        year_dfs.append(df)
    year_df = pd.concat(year_dfs)
    year_df.reset_index(inplace = True, drop = True)
    year_df.to_csv(year_folder+'.csv', sep = ';', index = False, encoding='latin-1')
