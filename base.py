#!/usr/bin/python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import quandl
import urllib
import csv
import pickle
import requests
import bs4 as bs
import time
import dateutil
import os
import glob
import lxml.html as lh
import re
import yfinance as yf
import datetime as dt
import xgboost as xgb
import statsmodels.api as sm
import seaborn as sns
import scipy
import sklearn
import mlxtend
import statsmodels.api as sm
import matplotlib.ticker as ticker
import joblib
import tensorflow as tf
import warnings

from numpy.random import rand
from matplotlib import style
from heapq import nlargest
from pandas import DataFrame
from datetime import datetime
from aiohttp import ClientSession
from random import random
from math import exp, log, sqrt
from datetime import date
from scipy.stats import norm,shapiro
from pylab import *
from matplotlib import style
from heapq import nlargest
from pandas.plotting import register_matplotlib_converters,scatter_matrix
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.externals import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from tpot import TPOTRegressor,TPOTClassifier
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from tpot.builtins import StackingEstimator
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from mlxtend.regressor import StackingRegressor
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.preprocessing import PolynomialFeatures
from numpy.random import rand
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, \
    RobustScaler, PowerTransformer, QuantileTransformer, Normalizer
from scipy.stats import trim_mean
from statsmodels.tsa.stattools import acf, pacf
from mpl_finance import candlestick2_ohlc
from scipy.signal import hilbert, chirp

from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout, Activation
from keras.models import load_model, model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore',category=FutureWarning)

pd.options.mode.chained_assignment = None 

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

params_date = { 'start_date': '2018-11-21',
                'end_date': '2019-11-20',
}

params_strat = { 'ticker': '^GDAXI', # GC=F ^GDAXI
                 'strat': 'ls',
                 'window': 20,
                 'init_cap': 1e5,
                 'num_pos': 100,
}

params = {
    'check_shape': True,
    'test_size': 0.1, # used in the train/validation split
    'regrs': 'lin', # [ 'lin', 'tree', 'forest', 'xgbr', 'nn', 'comb', 'tpot' ]
    'grid_search': False,
    'loss_eval': False, # for xgbr
    'filename': 'model.pkl',
    'with_thrs': True,
    'threshold': 10, # if with_thrs = True
    'in_sample_ratio': 0.75, # used for the split between the out-of-sample and in-sample data
    'fill_method': 'ffill',
}

params_lstm = {
    'lstm': True, # when False use the above parameters
    'train_size': 0.8,
    'test_size': 0.2,
    'batch_size': 2,
    'epochs': 100,
    'time_steps': 1,
    'lr': 0.0010000,
    'dropout': 0.0,
    'rec_dropout': 0.01,
    'stateful': True,
    'kernel': 'random_uniform',
}

def get_headers(df):
    return(df.columns.values)

def read_data(path):
    df = pd.read_csv(path,sep=',')
    return(df)

def fill_data(df,methd):
    methods = ['ffill','bfill','mean','zero']
    if(methd == 'ffill'):
        df.fillna(method='ffill',inplace=True)
        return(df)
    elif(methd == 'bfill'):
        df.fillna(method='bfill',inplace=True)
        return(df)
    elif(methd == 'mean'):
        df.fillna(df.mean(),inplace=True)
        return(df)
    elif(methd == 'zero'):
        df.fillna(0,inplace=True)
        return(df)
    else:
        raise ValueError("valid filling methods are ffill,bfill,mean or zero")
    
def view_data(df):
    print(df.head(20))
  
def get_info(df):  
    df.info()
    df.describe()
    
def format_output(df):
    df.columns = [''] * len(df.columns)
    df = df.to_string(index=False)
    return(df)
   
def check_lin(df):
    cols = [ col for col in df.columns if params['label'] not in col ]
    for el in cols:
        fig = plt.figure() 
        plt.scatter(df[str(el)], df[params['label']], color='red')
        plt.xlabel(str(el), fontsize=14)
        plt.ylabel(params['label'], fontsize=14)
        plt.grid(True)
        fig.savefig("figs/label_vs_"+str(el), bbox_inches='tight')
        plt.close()
      
def check_features(df,flag):
    cols = [ col for col in df.columns if str(flag) in col ] + [params['label']]
    scatter_matrix(df[cols], figsize=(15, 10), diagonal='kde')
    plt.show()

def get_cmtx_test(df):
    corr_matrix = df.corr()
    labels_ex = ['mid_price','vwap_price','trade_price','sum_trade_1s']

    f, ax = plt.subplots(2,2,figsize=(11, 9))
    f.autofmt_xdate(rotation=45)

    cmtx= [ corr_matrix[str(el)].sort_values(ascending=False) for el in labels_ex ]

    ax[0, 0].plot(cmtx[0],label=labels_ex[0])
    ax[0, 0].legend()

    ax[0, 1].plot(cmtx[1],label=labels_ex[1])
    ax[0, 1].legend()

    ax[1, 0].plot(cmtx[2],label=labels_ex[2])
    ax[1, 0].legend()

    ax[1, 1].plot(cmtx[3],label=labels_ex[3])
    ax[1, 1].legend()

    plt.show()
    
def show_cmtx(df):
    corr_matrix = df.corr()
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_matrix, cmap=cmap, center=0, square=True, linewidths=0.5)
    plt.show()

def plot_dist(df,feat):
    arr = np.array(df[str(feat)])
    raw_plt = sns.distplot(arr)
    # sqr_plt = sns.distplot(np.sqrt(arr))
    log_plt = sns.distplot(np.log(arr))
    plt.show()    
    
def check_shape(X_train, X_test, y_train, y_test):
    print('X_train.shape, X_test.shape = ', X_train.shape, X_test.shape)
    print('y_train.shape, y_test.shape = ', y_train.shape, y_test.shape)

def get_in_out_sample(df,flag):
    # the data is split into 3 parts:
    # (1) training (2) testing (3) validation
    # the first two are based on in-sample data
    # the last is based on out-of-sample data
    # for this data is taken out completely
    #------------------------------------
    #| 80% of data (1+2) | 20% taken out|              
    #------------------------------------ 
    df_in_sample = df.head(int(len(df)*(params['in_sample_ratio'])))
    df_out_sample = df.iloc[len(df_in_sample):]

    if(flag=='in'):
        return(df_in_sample)
    else:
        return(df_out_sample)
        
def ml_train_test(df,top_features):
    cols = top_features + [params['label']]
    df = df[cols]

    df_in_sample = get_in_out_sample(df,'in')
    
    X = np.array(df_in_sample.drop([params['label']],axis=1))
    y = np.array(df_in_sample[params['label']])     
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=params['test_size'],random_state=47253)
    if(params['check_shape']):
        check_shape(X_train, X_test, y_train, y_test)
    return(X_train, X_test, y_train, y_test)

def regressors(regrs):
    if(regrs == 'lin'):
        reg = LinearRegression(n_jobs=-1)
    elif(regrs == 'tree'):
        reg = DecisionTreeRegressor(max_features='auto', max_depth=30, max_leaf_nodes=270, min_samples_leaf=9, min_samples_split=3)
    elif(regrs == 'forest'):
        reg = RandomForestRegressor(n_estimators=10,max_depth=2,min_samples_split=4,min_samples_leaf=1,n_jobs=-1) 
    elif(regrs == 'xgbr'):
        reg = XGBRegressor(learning_rate=0.81, max_depth=4, min_child_weight=16, n_estimators=27,\
                           subsample=0.45, n_jobs=-1)
    elif(regrs == 'nn'):
        reg = MLPRegressor(hidden_layer_sizes=(32,1), activation='relu', solver='adam')
    elif(regrs == 'comb'):
        xgbr = XGBRegressor(learning_rate=0.81, max_depth=4, min_child_weight=16, n_estimators=27,\
                           subsample=0.45, n_jobs=-1)
        dtr = DecisionTreeRegressor(max_features='auto', max_depth=30, max_leaf_nodes=270, min_samples_leaf=9, min_samples_split=3)
        reg = StackingRegressor(regressors=[xgbr,dtr],meta_regressor=xgbr)
    elif(regrs == 'tpot'):
        reg = TPOTRegressor(generations=10,verbosity=2,scoring='r2',n_jobs=-1,random_state=254361)
    return(reg)

def set_grid_search(regrs,X_train,y_train,reg):
    if(regrs=='tree'):
        random_grid = build_grid_tree()
        prms = grid_search(reg,X_train,y_train,random_grid)
        reg_prms = DecisionTreeRegressor(max_features=prms['max_features'], max_depth=prms['max_depth'], \
                                         min_samples_split=prms['min_samples_split'], max_leaf_nodes=prms['max_leaf_nodes'], \
                                         min_samples_leaf=prms['min_samples_leaf'])
    elif(regrs=='forest'):
        random_grid = build_grid_rf()
        prms = grid_search(reg,X_train,y_train,random_grid)
        reg_prms = RandomForestRegressor(n_estimators=prms['n_estimators'],max_features=prms['max_features'], \
                                         max_depth=prms['max_depth'], min_samples_split=prms['min_samples_split'], \
                                         min_samples_leaf=prms['min_samples_leaf'], n_jobs=-1) 
    elif(regrs=='xgbr'):
        random_grid = build_grid_xgbr()
        prms = grid_search(reg,X_train,y_train,random_grid)
        reg_prms = XGBRegressor(learning_rate=prms['learning_rate'], max_depth=prms['max_depth'], \
                                min_child_weight=prms['min_child_weight'], n_estimators=prms['n_estimators'],\
                                subsample=prms['subsample'], n_jobs=-1)
    elif(regrs=='nn'):
        random_grid = build_grid_nn()
        prms = grid_search(reg,X_train,y_train,random_grid)
        reg_prms = MLPRegressor(hidden_layer_sizes=prms['hidden_layer_sizes'],activation=prms['activation'],solver=prms['solver'],\
                                alpha=prms['alpha'],learning_rate_init=prms['learning_rate_init'],learning_rate=prms['learning_rate'],\
                                max_iter=prms['max_iter'],tol=prms['tol'],momentum=prms['momentum'],beta_1=prms['beta_1'],\
                                beta_2=prms['beta_2'],n_iter_no_change=prms['n_iter_no_change'])
    return(reg_prms)

def loss_eval(X_train,X_test,y_train,y_test,reg):
    eval_set = [(X_train, y_train), (X_test, y_test)]
    reg.fit(X_train, y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=False)

    results = reg.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)

    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='test')
    ax.legend()
    plt.ylabel('log-loss')
    plt.title('XGBoost log-loss')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='train')
    ax.plot(x_axis, results['validation_1']['error'], label='test')
    ax.legend()
    plt.ylabel('error')
    plt.title('XGBoost error')
    plt.show()
    
    return(reg)
    
def model(X_train,X_test,y_train,y_test):
    if(params['grid_search']):
        reg_init = regressors(params['regrs'])
        reg = set_grid_search(params['regrs'],X_train,y_train,reg_init)
    else:
        reg = regressors(params['regrs'])
       
    if(params['loss_eval']):
        reg = loss_eval(X_train,X_test,y_train,y_test,reg)
    else:
        reg.fit(X_train,y_train)
        
    predictions = reg.predict(X_test)
    joblib.dump(reg,params['filename'])

    accuracy = r2_score(y_test,predictions)
    print('r2 score in-sample = ', accuracy)
    # print(reg.feature_importances_)
    
def scaler_def(index):
    if(index == 0):
        scl = StandardScaler()
    elif(index == 1):
        scl = MinMaxScaler()
    elif(index == 2):
        scl = MaxAbsScaler()
    elif(index == 3):
        scl = RobustScaler(quantile_range=(25, 75))
    elif(index == 4):
        scl = PowerTransformer(method='yeo-johnson')
    elif(index == 5):
        scl = PowerTransformer(method='box-cox')
    elif(index == 6):
        scl = QuantileTransformer(output_distribution='normal')
    elif(index == 7):
        scl = QuantileTransformer(output_distribution='uniform')
    elif(index == 8):
        scl = Normalizer()
    else:
        raise ValueError('not a correct scaler defined')
    return(scl)
    
def plot_ml(test,pred):
    plt.plot(test,'r*-')
    plt.plot(pred,'bo-')
    plt.xlabel(params['label'] + ' true values')
    plt.ylabel(params['label'] + ' predicted values')
    plt.show()

def model_tpot(X_train,X_test,y_train,y_test):
    reg = regressors(params['regrs'])       
    reg.fit(X_train,y_train)        

    predictions = reg.predict(X_test)       
    joblib.dump(reg,params['filename'])

    accuracy = r2_score(y_test,predictions)
    print(accuracy)

def build_grid_tree():
    max_features = ['auto','sqrt']
    max_depth = [3,9,10,30,90,270]
    max_leaf_nodes=[3,9,10,30,90,270]
    min_samples_split = [3,9,10,30,90,270]
    min_samples_leaf = [3,9,10,30,90,270]

    grid = {'max_features': max_features,
            'max_depth': max_depth,
            'max_leaf_nodes': max_leaf_nodes,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
    }
    return(grid)
    
def build_grid_rf():
    n_estimators = [20, 40, 80, 160]
    max_features = ['auto', 'sqrt']
    max_depth = [1, 3, 9]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]

    grid = {'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf
    }
    return(grid)

def build_grid_xgbr():
    learning_rate = [0.03, 0.09, 0.27, 0.81]
    n_estimators = [ 3, 9, 27, 81 ]
    max_depth = [ 2, 4, 8, 16 ]
    min_child_weight = [ 2, 4, 8, 16 ]
    subsample = [ 0.05, 0.15, 0.45 ]
    
    grid = {'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'subsample': subsample,
    }    
    return(grid)
  
def build_grid_nn():
    hidden_layer_sizes = [ (16,16), (8,8), (4,4) ]
    activation = ['relu', 'tanh']
    solver = ['adam']
    alpha = [ 0.001, 0.003, 0.01, 0.03, 0.1, 0.3 ]
    learning_rate_init = [ 0.001, 0.003, 0.01, 0.03, 0.1, 0.3 ]
    learning_rate = ['constant', 'adaptive'] 
    max_iter = [ 100, 300, 900 ]
    tol = [0.0001]
    momentum = [0.1,0.5,0.9]
    beta_1=[0.1,0.3,0.5,0.9]
    beta_2=[0.1,0.5,0.9]
    n_iter_no_change=[3,9,27]

    grid = {'hidden_layer_sizes': hidden_layer_sizes,
            'activation': activation,
            'solver': solver,
            'alpha': alpha,
            'learning_rate_init': learning_rate_init,
            'learning_rate': learning_rate,           
            'max_iter': max_iter,
            'tol': tol,
            'momentum': momentum,
            'beta_1': beta_1,
            'beta_2': beta_2,
            'n_iter_no_change': n_iter_no_change
    }
    return(grid)

def grid_search(reg,X_train,y_train,random_grid):
    print('#############################################')
    print('parameters in use before grid search')
    print('#############################################')

    pprint(reg.get_params())
    prms = reg.get_params()
    
    reg_random = RandomizedSearchCV(estimator=reg,param_distributions=random_grid,n_iter=10,cv=10, \
                                    verbose=0,random_state=42,n_jobs=-1)
    print("Randomized search..")
    search_time_start = time.time()
    reg_random.fit(X_train,y_train)
    print("Randomized search time:", time.time() - search_time_start)

    prms = reg_random.best_params_

    print('#############################################')
    print('best parameters after grid search')
    print('#############################################')
    pprint(prms)

    return(prms)

def check_missing_data(df):
    res = df.isnull().sum().sort_values(ascending=False)
    print(res)
    
    if(sum(res.values) != 0):
        kv_nz = omit_by(res)
        for el in kv_nz.keys():        
            print(df[df[str(el)].isnull()])

def get_nans(df):
    nulls = df.isnull().sum()
    nulls[nulls > 0]
    print(nulls)
            
def get_skew(df):
    skew_feats=df.skew().sort_values(ascending=False)
    return(skew_feats)

def write_to(df,name,flag):
    try:
        if(flag=="csv"):
            df.to_csv(str(name)+".csv")
        elif(flag=="html"):
            df.to_html(str(name)+"html")
    except:
        print("No other types supported")
        
def get_cmtx_label(df,corr_thrs,plotting):
    corr_matrix = df.corr()
    cmtx = corr_matrix[params['label']].sort_values(ascending=False)    
    dct = dict(zip(list(cmtx.keys()),cmtx.values))

    # possibility to select features based
    # on their correlation selected through a
    # threshold value specified as a parameter
    if(params['with_thrs']):
        dct_select = dict((k, v) for k, v in dct.items() if v >= float(corr_thrs)/100.0 and k != params['label'])
    else:
        dct_select = dict((k, v) for k, v in dct.items() if k != params['label'])

    if(plotting):
        fig, ax = plt.subplots(figsize=(11,9))
        plt.subplots_adjust(hspace = 0.5)
        fig.autofmt_xdate(rotation=45)

        ax.plot(list(dct_select.keys()),list(dct_select.values()),label=str(corr_thrs)+"% threshold")
        ax.set_title('Correlation of features with ' + str(params['label']))
        ax.legend()
        plt.show()

    return(list(dct_select.keys()))

def predict_oosample(df,top_features):
    cols = top_features + [params['label']]
    df = df[cols]
    
    df_out_sample = get_in_out_sample(df,'out')
    label_true_vals = list(df_out_sample[params['label']].values)
    
    # removing the label to apply the trained model
    df_out_sample.drop(columns=[params['label']],inplace=True)
    arr = df_out_sample.to_numpy()
    
    model = joblib.load(params['filename'])
    predictions = [ model.predict(arr[el].reshape(1,-1)).ravel()[0] for el in range(arr.shape[0]) ]
    accuracy = r2_score(label_true_vals,predictions)

    print('r2 score out-sample = ', accuracy)

    #############################################
    # here models are based on Neural Networks
    #############################################

def get_optimizer(flag):
    if(flag=='rms'):
        return(optimizers.RMSprop(lr=params_lstm["lr"]))
    elif(flag=='sgd'):
        return(optimizers.SGD(lr=params_lstm["lr"], decay=1e-6, momentum=0.9, nesterov=True))
    elif(flag=='adam_tn'):
        return(optimizers.Adam(lr=params_lstm["lr"]))
    elif(flag==''):
        return('adam')
    
def model_lstm(df,top_features,plotting):
    train_cols = top_features + [params['label']]
    df = df[train_cols]

    df_in_sample = get_in_out_sample(df,'in')

    df_train, df_test = train_test_split(df_in_sample, train_size=params_lstm['train_size'], \
                                         test_size=params_lstm['test_size'], shuffle=False)
    # print("Train and Test size", len(df_train), len(df_test))

    # scale the feature MinMax, build array
    x = df_train.loc[:,train_cols].values
    scaler = scaler_def(1)
    x_train = scaler.fit_transform(x)
    x_test = scaler.transform(df_test.loc[:,train_cols])
    
    idx_out = len(train_cols)-1
    x_t, y_t = build_timeseries(x_train, idx_out)
    x_t, y_t = trim_dataset(x_t, params_lstm["batch_size"]), trim_dataset(y_t, params_lstm["batch_size"]) 
    x_temp, y_temp = build_timeseries(x_test, idx_out)
    x_val, x_test_t = np.split(trim_dataset(x_temp, params_lstm["batch_size"]),2)
    y_val, y_test_t = np.split(trim_dataset(y_temp, params_lstm["batch_size"]),2)
   
    # (batch_size, timesteps, data_dim)
    print("######################")
    print(x_test_t.shape)
    print("######################")

    lstm_model = Sequential()
    lstm_model.add(LSTM(4, batch_input_shape=(params_lstm["batch_size"], params_lstm["time_steps"], x_t.shape[2]),
                        dropout=params_lstm['dropout'], recurrent_dropout=params_lstm['rec_dropout'], stateful=params_lstm['stateful'],
                        return_sequences=True, kernel_initializer=params_lstm['kernel']))
    lstm_model.add(LSTM(4, activation='relu', return_sequences=False))
    lstm_model.add(Dense(1, activation='tanh'))
    optmz = get_optimizer('')
    lstm_model.compile(loss='mse',optimizer=optmz)   

    lstm_model.fit(x_t, y_t, epochs=params_lstm['epochs'], verbose=2, batch_size=params_lstm["batch_size"], \
                   shuffle=False, validation_data=(trim_dataset(x_val, params_lstm["batch_size"]), \
                                                   trim_dataset(y_val, params_lstm["batch_size"])))
    
    model_json = lstm_model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    lstm_model.save_weights("model.h5")
    print("Saved model to disk")
    
    y_pred = lstm_model.predict(trim_dataset(x_test_t, params_lstm["batch_size"]), batch_size=params_lstm["batch_size"])
    y_pred = y_pred.flatten()
    y_test_t = trim_dataset(y_test_t, params_lstm["batch_size"])
    mse = mean_squared_error(y_test_t, y_pred)
    r2 = r2_score(y_test_t, y_pred)
    print("mse = ", mse)
    print("r2 = ", r2)  

    # convert the predicted value to range of real data
    y_pred_org = (y_pred * scaler.data_range_[idx_out]) + scaler.data_min_[idx_out]
    # scaler.inverse_transform(y_pred)
    y_test_t_org = (y_test_t * scaler.data_range_[idx_out]) + scaler.data_min_[idx_out]
    # scaler.inverse_transform(y_test_t)

    # Visualize the training data
    if(plotting):
        # plt.figure(figsize=(10.0,7.5))
        # plt.subplot(1, 2, 1)
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('Model loss')
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'], loc='upper left')

        plt.subplot(1,2,2)
        plt.plot(y_pred_org)
        plt.plot(y_test_t_org)
        plt.title('Prediction vs Real')
        plt.ylabel('Price')
        plt.xlabel('Days')
        plt.legend(['Prediction', 'Real'], loc='upper left')

        plt.show()
      
def build_timeseries(mat, y_col_index):
    # y_col_index is the index of column that would act as output column
    # total number of time-series samples would be len(mat) - params_lstm["time_steps"]
    dim_0 = mat.shape[0] - params_lstm["time_steps"]
    dim_1 = mat.shape[1]

    x = np.zeros((dim_0, params_lstm["time_steps"], dim_1))
    y = np.zeros((dim_0,))
    
    for i in range(dim_0):
        x[i] = mat[i:params_lstm["time_steps"]+i]
        y[i] = mat[params_lstm["time_steps"]+i, y_col_index]
    # print("length of time-series i/o",x.shape,y.shape)
    return x, y    

def trim_dataset(mat, batch_size):
    # trims dataset to a size that's divisible by params_lstm["batch_size"]
    no_of_rows_drop = mat.shape[0]%batch_size
    if(no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat

def predict_oosample_lstm(df,top_features):
    cols = top_features + [params['label']]
    df = df[cols]
    
    df_out_sample = get_in_out_sample(df,'out')
    label_true_vals = list(df_out_sample[params['label']].values)
    
    # removing the label to apply the trained model
    df_out_sample.drop(columns=[params['label']],inplace=True)
    arr = df_out_sample.to_numpy()

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    
    predictions = [ loaded_model.predict(arr[el].reshape(1,-1)).ravel()[0] for el in range(arr.shape[0]) ]
    accuracy = r2_score(label_true_vals,predictions)
    
    print('r2 score out-sample = ', accuracy)

def check_norm(df):
    fig = plt.figure(figsize=(16,9))
    ax1 = fig.add_subplot(121)
    z = lambda x: (x - x.mean()) / x.std()
    ax1.hist(z(df['daily_return']), bins=50, label='daily return')
    ax1.hist(z(df['log_return']), bins=50, label='log return')
    plt.title('Distributions')

    ax2 = fig.add_subplot(122)  
    sm.qqplot(df['daily_return'],ax=ax2,color='r')
    sm.qqplot(df['log_return'],ax=ax2,color='b')
    plt.title('Distributions')
    plt.show()

    # shapiro test
    p_value = scipy.stats.shapiro(df['log_return'].dropna())[1]
    if p_value <= 0.05:
        print("Null hypothesis of normality is rejected.")
    else:
        print("Null hypothesis of normality is accepted.")
    
def processing(df,dy):
    # moving average
    df['ma_'+str(dy//2)] = df['Adj Close'].rolling(window=dy//2).mean()
    df['ma_'+str(dy)] = df['Adj Close'].rolling(window=dy).mean()
    df['ma_'+str(dy*2)] = df['Adj Close'].rolling(window=dy*2).mean()

    df['ema_'+str(dy//2)] = df['Adj Close'].ewm(span=dy//2, adjust=False).mean()
    df['ema_'+str(dy)] = df['Adj Close'].ewm(span=dy, adjust=False).mean()
    df['ema_'+str(dy*2)] = df['Adj Close'].ewm(span=dy*2, adjust=False).mean()

    # bollinger
    df['std_'+str(dy//2)] = df['Adj Close'].rolling(window=dy//2).std()
    df['std_'+str(dy)] = df['Adj Close'].rolling(window=dy).std()
    df['std_'+str(dy*2)] = df['Adj Close'].rolling(window=dy*2).std()
    
    df['boll_up_'+str(dy)] = df['ma_'+str(dy)] + 2.0 * df['std_'+str(dy)]
    df['boll_dn_'+str(dy)] = df['ma_'+str(dy)] - 2.0 * df['std_'+str(dy)]   

    # return
    df['daily_return'] = df['Adj Close'].pct_change()
    df['log_return'] = np.log(1.0+df['daily_return'])

    df['vol_ann'] = df['daily_return'].rolling(window=252).std()*np.sqrt(252)
    df['vol_'+str(dy//2)] = df['daily_return'].rolling(window=dy//2).std()
    df['vol_'+str(dy)] =   df['daily_return'].rolling(window=dy).std()
    df['vol_'+str(dy*2)] = df['daily_return'].rolling(window=dy*2).std()

    df['vol_ann_log'] = df['log_return'].rolling(window=252).std()*np.sqrt(252)
    df['vol_'+str(dy//2)+'_log'] = df['log_return'].rolling(window=dy//2).std()
    df['vol_'+str(dy)+'_log'] =   df['log_return'].rolling(window=dy).std()
    df['vol_'+str(dy*2)+'_log'] = df['log_return'].rolling(window=dy*2).std()

    # aggregation
    df['cum_ret'] = (1.0+df['daily_return']).cumprod()
    df['vwap'] = (np.cumsum(df['Volume'] * df['Adj Close']) / np.cumsum(df['Volume']))
    # 'monthly_return' = df.resample('M').apply(lambda x: x[-1]).pct_change()
    return(df)

def sharpe_ratio(df,rf=0.05,days=252):
    volatility = df['log_return'].std() * np.sqrt(days) 
    sharpe_ratio = (df['log_return'].mean() - rf) / volatility
    return(sharpe_ratio)

def information_ratio(df,benchmark_returns=0.15,days=252):
    return_difference = df['log_return'] - benchmark_returns 
    volatility = return_difference.std() * np.sqrt(days) 
    information_ratio = return_difference.mean() / volatility
    return(information_ratio)

def modigliani_ratio(df, benchmark_returns=0.15, rf=0.05, days=252):
    volatility = df['log_return'].std() * np.sqrt(days) 
    sharpe_ratio = (df['log_return'].mean() - rf) / volatility 
    benchmark_volatility = benchmark_returns.std() * np.sqrt(days)
    m2_ratio = (sharpe_ratio * benchmark_volatility) + rf
    return(m2_ratio)

def get_signals(df,dy):
    signals = pd.DataFrame(index=df.index)
    signals[['Open','High','Low','Close','Adj Close']] = df[['Open','High','Low','Close','Adj Close']]
    
    # candlestick based
    signals['signal_hammer'] = 0
    conditions = [ (df['Close'].values > df['Open'].values) \
                   & (df['Open'].values > df['Low'].values) \
                   & (df['High'].values == df['Close'].values), \
                   (df['Close'].values < df['Open'].values) \
                   & (df['Close'].values > df['Low'].values) \
                   & (df['High'].values == df['Open'].values) ]
    choices = [ 1, 0 ]
    signals['signal_hammer'] = np.select(conditions, choices)
    signals['pos_hammer'] = signals['signal_hammer'].diff()
    
    signals['signal_bull_bear'] = 0
    conditions = [ (df['High'].values > df['Close'].values) \
                   & (df['Close'].values > df['Open'].values) \
                   & (df['Open'].values > df['Low'].values), \
                   (df['High'].values > df['Open'].values) \
                   & (df['Open'].values > df['Close'].values) \
                   & (df['Close'].values > df['Low'].values) ]
    choices = [ 1, 0 ]
    signals['signal_bull_bear'] = np.select(conditions, choices)
    signals['pos_bull_bear'] = signals['signal_bull_bear'].diff()
    
    # bollinger based
    signals['signal_boll'] = 0
    signals['Adj Close'] = df['Adj Close']

    signals['ma_'+ str(dy)] = df['Adj Close'].rolling(window=dy).mean()
    signals['std_'+str(dy)] = df['Adj Close'].rolling(window=dy).std()  
    signals['boll_up_'+ str(dy)] = signals['ma_'+str(dy)] + 2 * signals['std_'+str(dy)] 
    signals['boll_dn_'+ str(dy)] = signals['ma_'+str(dy)] - 2 * signals['std_'+str(dy)]

    conditions  = [ signals['Adj Close'][dy:] <= signals['boll_dn_'+ str(dy)][dy:], 
                    signals['Adj Close'][dy:] >= signals['boll_up_'+ str(dy)][dy:] ]
    choices     = [ 1, 0 ]
    signals['signal_boll'][dy:] = np.select(conditions, choices)
    signals['pos_boll'] = signals['signal_boll'].diff()
    
    # long/short (double cross-over) based
    signals['signal_ls'] = 0
    signals['sh_ma'] = df['Adj Close'].rolling(window=dy).mean()
    signals['lg_ma'] = df['Adj Close'].rolling(window=dy*3).mean()

    conditions  = [ signals['sh_ma'][dy:] > signals['lg_ma'][dy:],\
                    signals['sh_ma'][dy:] < signals['lg_ma'][dy:]] 
    choices     = [ 1, 0 ]
    signals['signal_ls'][dy:] = np.select(conditions, choices)
    signals['pos_ls'] = signals['signal_ls'].diff()
    
    # long/short (triple cross-over) based   
    signals['signal_triple'] = 0
    signals['sh_ma'] = df['Adj Close'].rolling(window=dy).mean()
    signals['lg_ma'] = df['Adj Close'].rolling(window=dy*3).mean()
    signals['vl_ma'] = df['Adj Close'].rolling(window=dy*9).mean()

    conditions  = [ (signals['sh_ma'][dy:] > signals['lg_ma'][dy:]) \
                    & (signals['sh_ma'][dy:] > signals['vl_ma'][dy:]) \
                    & (signals['lg_ma'][dy:] > signals['vl_ma'][dy:]),\
                    (signals['sh_ma'][dy:] > signals['lg_ma'][dy:]) \
                    & (signals['sh_ma'][dy:] < signals['vl_ma'][dy:]) \
                    & (signals['lg_ma'][dy:] < signals['vl_ma'][dy:])
                    ]
    choices     = [ 1, 0 ]
    signals['signal_triple'][dy:] = np.select(conditions, choices)
    signals['pos_triple'] = signals['signal_triple'].diff()
    
    # vwap based
    signals['signal_vwap'] = 0

    conditions = [ df['Adj Close'].values > df['vwap'].values, 
                   df['Adj Close'].values < df['vwap'].values ]
    choices     = [ 1, 0 ]
    signals['signal_vwap'] = np.select(conditions, choices)
    signals['pos_vwap'] = signals['signal_vwap'].diff()
    
    # gather all signals
    # signals = signals[['Adj Close','pos_hammer','pos_bull_bear','pos_boll','pos_ls','pos_vwap']]
    # signals.dropna(inplace=True)

    # signals['signal_all'] = 0
    # conditions = [ signals['signal_ls'] == signals['signal_vwap'],
    #                signals['signal_ls'] != signals['signal_vwap'] ]
    # choices     = [ 1, -1 ]
    # signals['signal_all'] = np.select(conditions, choices)
    
    return(signals)

def strat_bktst(signals):
    initial_capital= params_strat['init_cap']
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions[params_strat['ticker']] = params_strat['num_pos']*signals['signal_'+params_strat['strat']]   
    portfolio = positions.multiply(signals['Adj Close'], axis=0)
    pos_diff = positions.diff()
    portfolio['holdings'] = (positions.multiply(signals['Adj Close'], axis=0)).sum(axis=1)
    portfolio['cash'] = initial_capital - (pos_diff.multiply(signals['Adj Close'], axis=0)).sum(axis=1).cumsum()   
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    portfolio['returns'] = portfolio['total'].pct_change()

    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')
    portfolio['total'].plot(ax=ax1, lw=2.)
    ax1.plot(portfolio.loc[signals['pos_'+params_strat['strat']] == 1.0].index, 
             portfolio.total[signals['pos_'+params_strat['strat']] == 1.0],
             '^', markersize=10, color='m')
    ax1.plot(portfolio.loc[signals['pos_'+params_strat['strat']] == -1.0].index, 
             portfolio.total[signals['pos_'+params_strat['strat']] == -1.0],
             'v', markersize=10, color='k')
    plt.show()    
    return(portfolio)

def plot_signals(df,dy):
    # def mydate(x,pos):
    #     try:
    #         return xdate[int(x)]
    #     except IndexError:
    #         return ''

    # # hammer based
    # fig_hammer = plt.figure(figsize=(16,9))
    # ax_hammer = fig_hammer.add_subplot(111,ylabel='Price in $')

    # df.reset_index(inplace=True)   
    # candlestick2_ohlc(ax_hammer,df['Open'],df['High'],df['Low'],df['Close'],width=0.8)
    # xdate = [ i.date() for i in df['Date'] ]

    # df[['Open','High','Low','Close']].plot(ax=ax_hammer,lw=1)

    # ax_hammer.xaxis.set_major_locator(ticker.MaxNLocator(10))
    # ax_hammer.xaxis.set_major_formatter(ticker.FuncFormatter(mydate))
    # fig_hammer.autofmt_xdate()
    # fig_hammer.tight_layout()
    
    # # plotting the sell signals
    # ax_hammer.plot(df.loc[df.pos_hammer == -1.0].index, 
    #                df.pos_hammer[df.pos_hammer == -1.0],
    #                'v', markersize=5, color='k')
    # # plotting the buy signals
    # ax_hammer.plot(df.loc[df.pos_hammer == 1.0].index, 
    #                df.pos_hammer[df.pos_hammer == 1.0],
    #                '^', markersize=5, color='m')
    # plt.show()

    # long/short based
    fig_ls = plt.figure(figsize=(16,9))
    ax = fig_ls.add_subplot(111,ylabel='Price in $')
    df['Adj Close'].plot(ax=ax, color='r', lw=1)
   
    # plot the short and long lookback moving averages
    df[['sh_ma', 'lg_ma']].plot(ax=ax, lw=1)
    # plotting the sell signals
    ax.plot(df.loc[df.pos_ls == -1.0].index, 
            df.sh_ma[df.pos_ls == -1.0],
            'v', markersize=10, color='k')
    # plotting the buy signals
    ax.plot(df.loc[df.pos_ls == 1.0].index, 
            df.sh_ma[df.pos_ls == 1.0],
            '^', markersize=10, color='m')
    plt.show()

def get_features(df):
    features = pd.DataFrame(index=df.index)
    lst = ['Open','High','Low','Close','Adj Close','Volume','daily_return','log_return']
    features[lst] = df[lst]

    features['f01'] = features['daily_return']
    features['f02'] = features['Volume'].apply(np.log) # log of daily volume
    features['f03'] = features['Volume'].apply(np.sqrt) # log of daily volume
    features['f04'] = features['Volume'].diff() # change since prior day
    features['f05'] = features['Volume'].diff(50)
    
    # log of 5 day moving average of volume
    features['f06'] = features['Volume'].rolling(5).mean().apply(np.log)

    # daily volume vs. 200 day moving average
    features['f07'] = features['Volume']/features['Volume'].rolling(200).mean()-1

    # daily closing price vs. 50 day exponential moving avg
    features['f08'] = features['Adj Close']/features['Adj Close'].ewm(span=50).mean()-1

    # zscore
    zscore_fun = lambda x: (x - x.mean()) / x.std()
    features['f09'] = (features['Adj Close'] - features['Adj Close'].mean()) / features['Adj Close'].std()
    features['f09'].plot.kde(title='Z-Scores (not quite accurate)')
    
    features['f10'] = (features['Adj Close'] - features['Adj Close'].rolling(window=200, min_periods=20).mean())/ \
                        features['Adj Close'].rolling(window=200, min_periods=20).std()
    features['f10'].plot.kde(title='Z-Scores (accurate)')
    
    # returns the percentile rank (from 0.00 to 1.00) of traded
    # volume for each value as compared to a trailing 200 day period.
    features['f11'] = features['Volume'].rolling(200,min_periods=20).apply(lambda x: pd.Series(x).rank(pct=True)[0])
    features['f12'] = features['f05'].apply(np.sign)

    # how many days in a row a value has increased (or decreased)    
    features['f13'] = features['f12'].rolling(20).sum()
    features.dropna(inplace=True)
    return(features)

def create_outcomes(df):
    outcomes = pd.DataFrame(index=df.index)
    func_one_day_ahead = lambda x: x.pct_change(-1)
    outcomes['close_1'] = df['Close'].apply(func_one_day_ahead)
    func_five_day_ahead = lambda x: x.pct_change(-5)
    outcomes['close_5'] = prices.groupby(level='symbol').close.apply(func_five_day_ahead)
    return(outcomes)   

def get_rsi(df,dy):
    window_length = dy
    close = df['Adj Close']
    delta = close.diff()
    delta = delta[1:]

    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate the EWMA
    roll_up1 = up.ewm(span=window_length,adjust=False).mean()
    roll_down1 = down.abs().ewm(span=window_length,adjust=False).mean()
    # Calculate the RSI based on EWMA
    RS1 = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))

    # Calculate the SMA
    roll_up2 = up.rolling(window=window_length).mean()
    roll_down2 = down.abs().rolling(window=window_length).mean()
    # Calculate the RSI based on SMA
    RS2 = roll_up2 / roll_down2
    RSI2 = 100.0 - (100.0 / (1.0 + RS2))

    # Compare graphically
    plt.figure()
    RSI1.plot()
    RSI2.plot()
    plt.legend(['RSI via EWMA', 'RSI via SMA'])
    plt.show()

def get_macd(df):
    exp1 = df['Adj Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Adj Close'].ewm(span=26, adjust=False).mean()
    macd = exp1-exp2
    exp3 = macd.ewm(span=9,adjust=False).mean()

    macd.plot(label='MACD', color = '#EBD2BE')
    exp3.plot(label='Signal Line', color='#E5A4CB')
    plt.legend(loc='upper left')
    plt.show()   

def get_tesla_headlines(page):
    html = requests.get(page).text
    soup = BeautifulSoup(html)    
    headlines = soup.find_all("a", { "target" : "_self" })
    headlines.pop(0)
    dates = soup.findAll('small')
    dates.pop(0)
    return [i.text.strip() for i in headlines], [i.text.strip().split()[0] for i in dates]
    
def viz_shade(df,dy):
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    x_axis = df.index.get_level_values(0)
    
    ax.fill_between(x_axis, df['boll_up_'+str(dy)], df['boll_dn_'+str(dy)], color='grey')
    ax.plot(x_axis, df['Close'], color='blue', lw=2)
    plt.show()

def viz_data(df,xtitle,ytitle):
    df.plot(
        x=str(xtitle), 
        y=str(ytitle), 
        kind='bar', 
        legend=False, 
        color='blue',
        width=0.8,
        figsize=(16,9)
    )

    plt.xlabel(str(xtitle))
    plt.ylabel(str(ytitle))
    plt.gca().yaxis.grid(linestyle=':')
    plt.xticks(rotation='vertical', fontsize = 8)
    plt.savefig('dax.pdf')
    plt.show()

def get_trim_dat(df):
    # Mean after discarding top and  
    # bottom 10 % values eliminating outliers 
    dat_no_trim = df['Adj Close'].mean()
    dat_trim = trim_mean(df['Adj Close'], 0.1)
    return(dat_no_trim,dat_trim)
    
def get_dax_spot():
    data = yf.download(params_strat['ticker'], start=params_date['start_date'], end=params_date['end_date'])
    return(data)

def get_acf_pacf(df):
    lag_acf = acf(df['Adj Close'], nlags=300)
    lag_pacf = pacf(df['Adj Close'], nlags=30, method='ols')

    fig = plt.figure(figsize=(16,9))

    ax1 = fig.add_subplot(211)
    ax1.plot(lag_acf,marker='+')
    ax1.axhline(y=0,linestyle='--',color='gray')
    ax1.axhline(y=-1.96/np.sqrt(len(lag_acf)),linestyle='--',color='gray')
    ax1.axhline(y=1.96/np.sqrt(len(lag_acf)),linestyle='--',color='gray')
    plt.title('ACF')
    plt.xlabel('number of lags')
    plt.ylabel('correlation')
    plt.grid(True)
    plt.tight_layout()    

    ax2 = fig.add_subplot(212)
    ax2.plot(lag_pacf,marker='+')
    ax2.axhline(y=0,linestyle='--',color='blue')
    ax2.axhline(y=-1.96/np.sqrt(len(lag_pacf)),linestyle='--',color='blue')
    ax2.axhline(y=1.96/np.sqrt(len(lag_pacf)),linestyle='--',color='blue')
    plt.title('PACF')
    plt.xlabel('number of lags')
    plt.ylabel('correlation')
    plt.grid(True)
    plt.tight_layout()    
    plt.show()
   
def execute():
    regrs = params['regrs']

    # (0) load/view raw data
    df_raw = get_dax_spot()
    
    # (1) check/count any missing data
    # print(check_missing_data(df_raw))

    # (2) filling dataframe
    # df = fill_data(df_raw,params['fill_method'])

    # (3) process columns to create additional features
    dy = params_strat['window']
    df = processing(df_raw,dy)   
    
    feats = get_features(df)
    view_data(feats)
    
    # sigs = get_signals(df,dy)   
    # strat_bktst(sigs)
    # plot_signals(sigs,dy)
    
    # (4) plot correlation matrix
    # show_cmtx(df)

    # (5) check on different potential labels
    # get_cmtx_test(df)
    
    # (6) trade_price is chosen as label
    # plot its correlation w.r.t. other
    # with features at different threshold
    # to see the most significant features
    top_features = [ 'Adj Close' ] # get_cmtx_label(df,params['threshold'],False)
    # print(top_features)

    # (7) plot features scatterplot
    # for i,el in enumerate(features):
    #     check_features(df,features[i])

    # if(not params_lstm['lstm']):
    #     # (8) basic ml regressors
    #     X_train, X_test, y_train, y_test = ml_train_test(df,top_features)
    #     model(X_train,X_test,y_train,y_test)

    #     # (9) use the model trained above (stored as model.pkl)
    #     # to get its performance on the out-of-sample part of data
    #     # prediction made with the standard ML model (no lstm)
    #     predict_oosample(df,top_features)

    # else:
    #     # (10) lstm model
    #     model_lstm(df,top_features,False)
    #     # (11) use the model trained above (stored as model.json and model.h5)
    #     # to get its performance on the out-of-sample part of data
    #     # prediction made with the lstm model
    #     predict_oosample_lstm(df,top_features)
    
    # (12) optional: test auto ML using Tpot
    # just for reference
    # X_train, X_test, y_train, y_test = ml_train_test(df,top_features)
    # model_tpot(X_train,X_test,y_train,y_test)
 
if __name__ == '__main__':
    execute()
    

    















