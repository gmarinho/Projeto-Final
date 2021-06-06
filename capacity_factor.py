# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 17:35:21 2020

@author: mariga
"""
import pandas as pd
import datetime
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from sklearn import metrics
from sklearn import preprocessing
import xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import PredefinedSplit


def map_state_hydro(x, state_dict):
    try:
        return state_dict[x]
    except:
        return 'IGNORE'
    
def get_7_days_prec(x, df, basin_df):
    ######## 7 days prec for the hydro shape ###########
    date = x.date
    last_week = date - datetime.timedelta(7)
    prec = df[np.logical_and(df.date<date, df.date>=last_week)].precipitation.sum()
    x['prec_7d'] = prec
    ######## 7 Days prec for the full basin ######################################
    prec_basin = basin_df[np.logical_and(basin_df.date<date, basin_df.date>=last_week)].precipitation.sum()
    x['prec_basin_7d'] = prec_basin
    ################ Date preciptation for the basin ####################
    date_prec_basin = basin_df[basin_df.date == date].precipitation.sum()
    x['prec_basin'] = date_prec_basin
    return x
    
"""
Capacity vectors by basin-state in Brazil

After mapping all the power plants basin, state, river, capacity and operation start date. The information
is compiled to create capacity vectors by basin-state, the decision to drop the river as an aggregation
column is due to the impossibility of mapping weather stations to a river.
The capacity is calculated for every day and hydro curve(basin-state) in the time range selected
"""
start_date = '1/1/2010'
end_date = '31/12/2020'
aurora_resources = pd.read_excel(r'C:\Users\mariga\OneDrive - Verisk Analytics\Projeto Final\Data\hydro_power_plants_v3.xlsx')
#Getting the available basin-state compbinations within the dataset
aurora_resources = aurora_resources[aurora_resources.Ignore==False]
aurora_resources = aurora_resources[aurora_resources.plant_type=='ror']
aurora_resources.drop(columns = ['plant_type'], inplace = True)
aurora_resources['hydro_shape'] = aurora_resources['state'] + ' ' +aurora_resources['basin']
date_rng = pd.date_range(start =start_date, end = end_date)
shapes = aurora_resources.hydro_shape.unique()
#Dictionary with capacity per day per region, converting from MW to GW
hydro_region_capacity = {region : {date : aurora_resources[np.logical_and(aurora_resources['start_date']<=date,aurora_resources['hydro_shape'] == region)]['capacity'].sum()/1000 for date in date_rng } for region in shapes}
#brazil_solar_installed_capacity = [brazil_resources_solar[brazil_resources_solar['Resource Begin Date']<=date]['Installed_Capacity'].sum() for date in date_rng]
df_hydro_region_capacity = pd.DataFrame(hydro_region_capacity)
df_hydro_region_capacity.index.name = 'date'
#Creating a new index, so the date can be used as a regular column
df_hydro_region_capacity.reset_index(inplace = True)
#Total column for reference
#Adding the unit information
df_hydro_region_capacity['unit'] = 'GW'

df_hydro_region_capacity = df_hydro_region_capacity.melt(id_vars = ['date', 'unit'], value_name ='capacity', var_name ='region')
 
df_hydro_region_capacity['day'] = df_hydro_region_capacity.date.apply(lambda x: x.day).astype(int)
df_hydro_region_capacity['month'] = df_hydro_region_capacity.date.apply(lambda x: x.month).astype(int)
df_hydro_region_capacity['year'] = df_hydro_region_capacity.date.apply(lambda x: x.year).astype(int)
 

""""
Generation data:
The generation data was acquired using a crawler in the ONS webpage.
Now, to create the capacity factors, both generation and capacity are going to be aggregated by
basin-state and joined together to create the capacity factor for each curve.

"""
df_generation = pd.read_csv(r'C:\Users\mariga\OneDrive - Verisk Analytics\Projeto Final\Data\generation_data.csv', encoding='latin-1', sep=';')
df_generation = df_generation.iloc[1:]
df_generation.Year = df_generation.Year.astype(int)
df_generation.Month = df_generation.Month.astype(int)
df_generation.Day = df_generation.Day.astype(int)
df_generation.Generation = df_generation.Generation.astype(float)
#
reduced_resources = aurora_resources.copy()[['name_ons','state','hydro_shape']].drop_duplicates()
df_joined = pd.merge(df_generation, reduced_resources, left_on=['Name', 'State'], right_on=['name_ons', 'state'], how = 'inner')
df_joined.drop(columns = ['state', 'name_ons','Name', 'Reporting Region',
       'Energy_Source','Unit', 'Data_Source','State'], inplace = True)
df_joined = df_joined.groupby(by=['Year', 'Month', 'Day', 'hydro_shape'],as_index=False).sum()
df_capacity_factor = pd.merge(df_joined, df_hydro_region_capacity, left_on=['Year','Month','Day','hydro_shape'], right_on =[ 'year', 'month','day', 'region'], how = 'left' )
df_capacity_factor['CF'] = df_capacity_factor.Generation/(df_capacity_factor.capacity*24)
#####Due to issues in the operation date, some regions have capacity factor higher than 1 in given days
df_capacity_factor.CF = df_capacity_factor.CF.apply(lambda x: x if x<=1 else 1)
df_capacity_factor = df_capacity_factor[['date','year', 'month','day','unit',
       'region','CF']]


##################Joining capacity factor and weather###########################################
weather_data = pd.read_csv(r"C:\Users\mariga\OneDrive - Verisk Analytics\Documents\GitHub\Projeto Final\Data Wrangling Scripts\daily_precipitation_avg_per_basin.csv", encoding='latin-1')
##############Creating the day - 1 logic to associate with capacity factors
df_cf_weather = pd.merge(df_capacity_factor, weather_data,
                         left_on=['year','month','day','region'],
                         right_on=['year','month','day','hydro_shape'], how = 'left')
#Some data points will not have a station associated with a basin-state curve
#for those, a state average will be used to fill the nas
df_nan = df_cf_weather.copy()[df_cf_weather.precipitation.isnull()]
df_not_null = df_cf_weather.copy()[df_cf_weather.precipitation.notnull()]
state_map = aurora_resources.drop(columns =['id',  'name_ons', 'name_id', 'region', 'energy_source',
       'data_source', 'basin', 'river', 'capacity', 'ceg_code', 'start_date',
       'Ignore']).drop_duplicates().set_index('hydro_shape').to_dict()['state']
df_nan['state'] = df_nan.region.apply(lambda x: state_map[x])
weather_data['state'] = weather_data.hydro_shape.apply(lambda x: map_state_hydro(x,state_map))
weather_data = weather_data.drop(columns=['hydro_shape', 'full_year_flag'])
weather_data = weather_data.groupby(by = ['day', 'month', 'year', 'unit',
       'state'], as_index = False).mean()
df_nan.drop(columns = [ 'unit_y',
       'hydro_shape', 'precipitation', 'full_year_flag'], inplace = True)
df_nan = pd.merge(df_nan, weather_data,left_on = ['year', 'month', 'day','state'], right_on=['year','month','day', 'state'], how = 'left')
df_nan.drop(columns =['unit_x','state','unit'], inplace = True)
df_not_null.drop(columns= ['unit_x','unit_y','hydro_shape', 'full_year_flag'], inplace = True)
df_final = pd.concat([df_nan, df_not_null],ignore_index=True)
######### Mapping Basin - Hydro Curve for final dataframe ##############
shape_basin_map = aurora_resources.copy()[['basin', 'hydro_shape']]
shape_basin_map.drop_duplicates(inplace = True)
df_final = pd.merge(df_final, shape_basin_map, left_on = 'region', right_on='hydro_shape')
###### Getting the precipitation from the 7 previous days in the shape and basin #####################
dfs_region = []
for region in df_final.region.unique():
    df_region = df_final[df_final.region == region]
    basin = df_region.basin.unique()[0]
    df_basin = df_final[df_final.basin == basin]
    df_region = df_region.apply(lambda x: get_7_days_prec(x, df_region, df_basin),axis=1)
    dfs_region.append(df_region)

df_final = pd.concat(dfs_region)
df_final.reset_index(inplace = True, drop = True)

#############################################################################

################## Daily Test ######################
df_final = pd.read_csv(r"C:\Users\mariga\OneDrive - Verisk Analytics\Documents\GitHub\Projeto Final\Data Wrangling Scripts\formatted_data_for_ML.csv", encoding = 'latin-1')
df_daily = pd.concat([df_final,pd.get_dummies(df_final['region'],dummy_na=False)],axis=1).drop(['region'],axis=1)
df_daily.drop(columns = ['date', 'basin', 'hydro_shape'], inplace = True)
df_daily = df_daily[df_daily.year<2020]
#Using data until 2017 as train set
df_train = df_daily[df_daily.year<2018]
#Using data from 2018 onwards as test set
df_test =df_daily[df_daily.year>2017]
indexes = df_daily.year.apply(lambda x: -1 if x<2018 else 0)

regions = list(df_final.region.unique())
y_train = df_train.CF
y_train = y_train.iloc[:]
X_train = df_train.drop(columns = ['CF'])
X_train = X_train.iloc[:,:]
X_test = df_test.drop(columns = ['CF'])
y_test = df_test.CF
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
scalerxgb = preprocessing.StandardScaler().fit(df_daily.drop(columns = ['CF']))
X = scalerxgb.transform(df_daily.drop(columns = ['CF']).iloc[:,:])
Y = df_daily.CF
########## Simple Neural Net Model

model = Sequential()
model.add(Dense(100, activation='sigmoid', input_shape=(42,)))
model.add(Dense(500, activation='sigmoid'))
model.add(Dense(1000, activation='sigmoid'))
model.add(Dense(2000, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam',
              metrics=['mse'])
callback = EarlyStopping(monitor = 'loss', patience = 2)
model.fit(X_train, y_train,epochs=100, batch_size=100, verbose=1, callbacks =[callback] )
model = load_model(r"C:\Users\mariga\OneDrive - Verisk Analytics\Documents\GitHub\Projeto Final\Data Wrangling Scripts\trained_neural_model.h5")
y_pred_ann = model.predict(X_test)
print("RMSE:",sqrt(metrics.mean_squared_error(y_test, y_pred_ann)))
#######XGBoost regressor ############
#Grid Search used to configure the XGBoost Regressor
#ps = PredefinedSplit(indexes.to_numpy())
#xb = xgboost.XGBRegressor()
#xgb_params = {
#        'min_child_weight': [1, 5, 10],
#        'gamma': [0.5, 1, 1.5, 2, 5],
#        'subsample': [0.6, 0.8, 1.0],
#        'colsample_bytree': [0.6, 0.8, 1.0],
#        'max_depth': [3, 4, 5]
#        }
#gridcv = GridSearchCV(xb, xgb_params, verbose = 3, scoring = 'neg_mean_squared_error', n_jobs = -1, cv = ps)
#gridcv.fit(X, Y)
#xb = xgboost.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#             colsample_bynode=1, colsample_bytree=0.6, gamma=1, gpu_id=-1,
#             importance_type='gain', interaction_constraints='',
#             learning_rate=0.300000012, max_delta_step=0, max_depth=3,
#             min_child_weight=1, monotone_constraints='()',
#             n_estimators=100, n_jobs=0, num_parallel_tree=1,
#             objective='reg:squarederror', random_state=0, reg_alpha=0,
#             reg_lambda=1, scale_pos_weight=1, subsample=0.8,
#             tree_method='exact', validate_parameters=1, verbosity=None)
xb = xgboost.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.6, gamma=1, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.300000012, max_delta_step=0, max_depth=6,
             min_child_weight=1,  monotone_constraints='()',
             n_estimators=100, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=0.8, tree_method='exact',
             validate_parameters=1, verbosity=None)
xb.fit(X_train, y_train)
y_pred_xgb = xb.predict(X_test)
print("RMSE:",sqrt(metrics.mean_squared_error(y_test, y_pred_xgb)))
#######SVM regressor ############
from sklearn.svm import LinearSVR
svr = LinearSVR(C=1, epsilon=0.005, dual = False, loss = 'squared_epsilon_insensitive')
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)
print("RMSE:",sqrt(metrics.mean_squared_error(y_test, y_pred_svr)))
#####################################################
####### Random Forest Regressor ############
from sklearn.ensemble import RandomForestRegressor
ps_rf = PredefinedSplit(indexes.to_numpy())
rf_params = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)],
               'max_features': ['auto', 'sqrt'],
               'max_depth':  [int(x) for x in np.linspace(10, 110, num = 11)],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'bootstrap': [True, False]}
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
gridsearch_rf = RandomizedSearchCV(rf, rf_params,n_jobs = -1, cv = ps_rf, n_iter=150, verbose = 3)
gridsearch_rf.fit(X,Y)
y_pred_rf = rf.predict(X_test)
print("RMSE:",sqrt(metrics.mean_squared_error(y_test, y_pred_rf)))
#####################################################


y_pred_xgb = pd.DataFrame(y_pred_xgb)
y_pred_ann = pd.DataFrame(y_pred_ann)
df_predictions = pd.concat([y_test.reset_index(drop = True), y_pred_xgb, y_pred_ann], axis = 1, ignore_index=True)
df_predictions.columns = ['real','XGBoost', 'ANN']
df_predictions.plot()
path = 'C:/Users/mariga/OneDrive - Verisk Analytics/Projeto Final/Results/'
mapping = pd.read_csv(path+'state_list.csv')
####Plotting the curve for all regions
results = {'Region':[],'MLP':[],'XGBoost':[],'RF':[],'SVR':[]}
for region in regions:
    #### Filtering the data points for the region
    state = mapping[mapping.Region == region].State.iloc[0]
    basin = mapping[mapping.Region == region].Basin.iloc[0]
    X_test = df_test.drop(columns = ['CF'])[df_test[region]==1]
    y_test = df_test[df_test[region]==1].CF   
    X_test = scaler.transform(X_test)
    #### Predictions
    y_pred_svr = svr.predict(X_test)
    y_pred_xgb = xb.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    y_pred_mlp = model.predict(X_test)
    #### Converting predictions to pandas dataframes
    y_pred_xgb = pd.DataFrame(y_pred_xgb)
    y_pred_svr = pd.DataFrame(y_pred_svr)
    y_pred_rf = pd.DataFrame(y_pred_rf)
    y_pred_mlp = pd.DataFrame(y_pred_mlp)
    #### Joining all dfs
    df_predictions = pd.concat([y_test.reset_index(drop = True), y_pred_xgb, y_pred_rf,y_pred_svr,y_pred_mlp], axis = 1, ignore_index=True)
    df_predictions.columns = ['real','XGBoost', 'RF', 'SVR','MLP']
    #### Getting the original data to have year and month information
    X_original = scaler.inverse_transform(X_test)
    test_final = pd.DataFrame(X_original)
    test_final.columns = df_test.drop(columns = ['CF']).columns
    test_final = test_final[['year','month','day']]
    test_final['SVR'] = y_pred_svr
    test_final['XGBoost'] = y_pred_xgb
    test_final['RF'] = y_pred_rf
    test_final['MLP'] = y_pred_mlp
    test_final['real'] = y_test.reset_index(drop = True)
    ### RMSE Message for the chart
    rmse_info = "RMSE XGBoost: " +str(sqrt(metrics.mean_squared_error(y_test, y_pred_xgb))) + "\n" + "RMSE SVR: "+str(sqrt(metrics.mean_squared_error(y_test, y_pred_svr))) + "\n" + "RMSE RF: "+str(sqrt(metrics.mean_squared_error(y_test, y_pred_rf))) + "\n" + "RMSE MLP: "+str(sqrt(metrics.mean_squared_error(y_test, y_pred_mlp)))
    ### Groupping the predictions by month, to reduce the noise for daily data
    test_final = test_final.drop(columns = ['day']).groupby(by = ['year','month'], as_index = False).mean()
    figure = test_final.drop(columns=['year','month']).plot(title =state + ' - ' + basin).get_figure()
    figure.text(0.0,0.9,rmse_info, fontsize = 5)
    plt.ylabel('Capacity Factor')
    plt.xlabel('Months')
    figure.savefig(path+'simulation_'+region+'.jpg')
    plt.close(figure)
    results['Region'].append(region)
    results['XGBoost'].append(sqrt(metrics.mean_squared_error(y_test, y_pred_xgb)))
    results['RF'].append(sqrt(metrics.mean_squared_error(y_test, y_pred_rf)))
    results['MLP'].append(sqrt(metrics.mean_squared_error(y_test, y_pred_mlp)))
    results['SVR'].append(sqrt(metrics.mean_squared_error(y_test, y_pred_svr)))



