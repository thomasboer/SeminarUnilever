# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 21:30:46 2022

@author: Daphne, Luigi, Mathilde, Thomas
"""

# =============================================================================
# 1. Import libraries
# =============================================================================
import copy 

from datetime import datetime
from dateutil import rrule

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from numpy.fft import fft, rfft, rfftfreq
import numpy as np
from numpy import arange

import pandas as pd
from pandas import Series
from pmdarima.arima import auto_arima
from pmdarima.arima.utils import ndiffs

import scipy.signal as signal
from scipy.fftpack import fft, fftfreq

from sklearn.linear_model import Lasso

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import kpss
from statsmodels import api
import statsmodels.api as sm

import time

# =============================================================================
# 2. Definition of self-made functions
# =============================================================================

def detrending2(data):
    y = pd.DataFrame(signal.detrend(data['interest'], type = 'constant'))
    y.columns = ['interest']

    x1 = np.ones(len(y['interest']))
    x2 = np.arange(0,len(y['interest']))
    xols = np.vstack((x1,x2)).T
    model_ols = api.OLS(np.array(y['interest']), xols).fit()

    p_value_trend = model_ols.pvalues[1]
    slope = 0
    if model_ols.pvalues[1] >= 0.01:
        detrended = y
        detrended = pd.concat([detrended, data['date']], axis = 1)
        detrended = pd.DataFrame(np.array(detrended), columns = ['interest', 'date'])
        
    else:    
        df2 = y.reset_index() #This doesn't work for time indexes, so we need to reset
        data.columns = ['date','interest']
        coeff = np.polyfit(df2.index,df2.interest,deg=1)
        slope = coeff[0]
        model = np.poly1d(coeff)
        y['trend'] = model(df2.index) # Remember, this model uses the the integer x axis, not time
        y['residual'] = y['interest'] - y['trend']
        detrended = y['residual']
        pd.DataFrame(np.array(detrended), columns = ['interest'])
        detrended = pd.concat([detrended, data['date']], axis = 1)
        detrended = pd.DataFrame(np.array(detrended), columns = ['interest', 'date'])
    
    return detrended, p_value_trend, slope

def sample_first_prows(data, perc=0.75):
    return data.head(int(len(data)*(perc)))

def data_keyword_index(all_data, index: int, country: str):
    '''
        To obtain data corresponding to a specific dish given by the index
    '''
    data_nl = all_data[all_data['countryCode'] == country] #Filter data on a country (NL)
    unique_keywords = np.unique(data_nl['keyword']) #Create list with all different keywords
    data_nl = data_nl[['date', 'interest', 'keyword']]
    data_nl['date'] = pd.DataFrame(pd.to_datetime(data_nl['date'], format = '%d-%m-%Y'))
    data_nl = data_nl.sort_values(by = 'date')
    data_nl = data_nl.reset_index(drop=True) #Filter data on specific keyword
    keyword_data = data_nl[data_nl['keyword'] == unique_keywords[index]].reset_index(drop=True) 

    return keyword_data, unique_keywords

def data_keyword_str(all_data, keyword: str, country: str):
    '''
        To obtain data corresponding to a specific dish given by the string
    '''
    
    data_nl = all_data[all_data['countryCode'] == country] #Filter data on a country (NL)
    unique_keywords = np.unique(data_nl['keyword']) #Create list with all different keywords
    index = list(unique_keywords).index(keyword)
    data_nl = data_nl[['date', 'interest', 'keyword']]
    data_nl = data_nl.reset_index(drop=True) #Filter data on specific keyword
    keyword_data = data_nl[data_nl['keyword'] == unique_keywords[index]].reset_index(drop=True) 
 
    # Data chronologisch sorteren
    keyword_data['date'] = pd.DataFrame(pd.to_datetime(keyword_data['date'], format = '%d-%m-%Y'))
    keyword_data = keyword_data.sort_values(by = 'date')
    keyword_data = keyword_data.reset_index(drop = True) 

    return keyword_data #Filter data on specific keyword

def add_holidays(data):
    data['date'] = pd.to_datetime(data['date'], format = '%d-%m-%Y')
    dates = pd.DataFrame(pd.date_range("2018-01-01", periods=260, freq='W'), columns = ['date'])

    # holiday dummies
    pasen = pd.DataFrame({
    'holiday': 'pasen',
    'ds' : pd.to_datetime(['2018-04-01', '2018-04-02', 
                            '2019-04-21', '2019-04-22', 
                            '2020-04-12', '2020-04-12', 
                            '2021-04-04', '2021-04-05' , 
                            '2022-04-17', '2022-04-18']),
    },
    index = [0,1,2,3,4,5,6,7,8,9],
    )

    hemelvaart = pd.DataFrame({
    'holiday': 'hemelvaart',
    'ds': pd.to_datetime(['2018-05-10', '2018-05-11', '2018-05-12', '2018-05-13', 
                            '2019-05-30', '2019-05-31', '2019-06-01', '2019-06-02',
                            '2020-05-21', '2020-05-22','2020-05-23','2020-05-24',
                            '2021-05-13', '2021-05-14', '2021-05-15', '2021-05-16',
                            '2022-05-26', '2022-05-27', '2022-05-28', '2022-05-29']),
    },
    index = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
    )

    pinksteren = pd.DataFrame({
    'holiday': 'pinksteren',
    'ds': pd.to_datetime(['2018-05-19', '2018-05-20', '2018-05-21', 
                            '2019-06-08', '2019-06-09', '2019-06-10',
                            '2020-05-30', '2020-05-31', '2020-06-01', 
                            '2021-05-22', '2021-05-23', '2021-05-24',
                            '2022-06-04', '2022-06-05', '2022-06-06']),
    },
    index = [30,31,32,33,34,35,36,37,38,39,40,41,42,43,44],
    )

    kerst = pd.DataFrame({
    'holiday': 'kerst',
    'ds': pd.to_datetime(['2018-12-24', '2018-12-25', '2018-12-26', 
                            '2019-12-24', '2019-12-25', '2019-12-26',
                            '2020-12-24', '2020-12-25', '2020-12-26', 
                            '2021-12-24', '2021-12-25', '2021-12-26',
                           '2022-12-24', '2022-12-25', '2022-12-26']),
    },
    index = [45,46,47,48,49,50,51,52,53,54,55,56,57,58,59],
    )

    # Create 3 extra (empty) columns in the 'data' dataframe. Those will be filled with seasonal dummies
    frames = [pasen, hemelvaart, pinksteren, kerst]
    holidays = pd.concat(frames)
    holidays = pd.DataFrame(holidays)
    columns = ['pasen', 'hemelvaart', 'pinksteren', 'kerst']
    data_empty = pd.DataFrame(np.zeros((len(dates),4)))
    data_add = pd.DataFrame(data = data_empty, columns = columns)
    data = pd.concat([dates, data['interest'], data_add], axis = 1)
    
    holidaydate = 0
    for index, row in data.iterrows():
        for holidaydate in range(0,len(holidays)):
            if holidaydate <=9:
                if index == 0:
                    lowdate = datetime(2018,1,1)
                    highdate = row['date']
                    if lowdate < holidays.loc[holidaydate]['ds'] <= highdate:
                        data.loc[index,'pasen'] = 1


                elif index == data.shape[0] - 1:
                    lowdate = row['date']
                    highdate = datetime(2022,12,31)
                    if lowdate < holidays.loc[holidaydate]['ds'] <= highdate:
                        data.loc[index-1,'pasen'] = 1


                else:
                    lowdate = data.loc[index-1]['date']
                    highdate = data.loc[index]['date']
                    if lowdate < holidays.loc[holidaydate]['ds'] <= highdate:
                        data.loc[index-1,'pasen'] = 1

            if 9 < holidaydate <= 29:
                if index == 0:
                    lowdate = datetime(2018,1,1)
                    highdate = row['date']
                    if lowdate < holidays.loc[holidaydate]['ds'] <= highdate:
                        data.loc[index,'hemelvaart'] = 1


                elif index == data.shape[0] - 1:
                    lowdate = row['date']
                    highdate = datetime(2022,12,31)
                    if lowdate < holidays.loc[holidaydate]['ds'] <= highdate:
                        data.loc[index-1,'hemelvaart'] = 1


                else:
                    lowdate = data.loc[index-1]['date']
                    highdate = data.loc[index]['date']
                    if lowdate < holidays.loc[holidaydate]['ds'] <= highdate:
                        data.loc[index-1,'hemelvaart'] = 1

            if 29 < holidaydate <= 44:
                if index == 0:
                    lowdate = datetime(2018,1,1)
                    highdate = row['date']
                    if lowdate < holidays.loc[holidaydate]['ds'] <= highdate:
                        data.loc[index,'pinksteren'] = 1


                elif index == data.shape[0] - 1:
                    lowdate = row['date']
                    highdate = datetime(2022,12,31)
                    if lowdate < holidays.loc[holidaydate]['ds'] <= highdate:
                        data.loc[index-1,'pinksteren'] = 1


                else:
                    lowdate = data.loc[index-1]['date']
                    highdate = data.loc[index]['date']
                    if lowdate < holidays.loc[holidaydate]['ds'] <= highdate:
                        data.loc[index-1,'pinksteren'] = 1

            if 45 < holidaydate <= 59:
                if index == 0:
                    lowdate = datetime(2018,1,1)
                    highdate = row['date']
                    if lowdate < holidays.loc[holidaydate]['ds'] <= highdate:
                        data.loc[index,'kerst'] = 1


                elif index == data.shape[0] - 1:
                    lowdate = row['date']
                    highdate = datetime(2022,12,31)
                    if lowdate < holidays.loc[holidaydate]['ds'] <= highdate:
                        data.loc[index-1,'kerst'] = 1


                else:
                    lowdate = data.loc[index-1]['date']
                    highdate = data.loc[index]['date']
                    if lowdate < holidays.loc[holidaydate]['ds'] <= highdate:
                        data.loc[index-1,'kerst'] = 1
    return data

def train_test_split(data, perc):
    train = sample_first_prows(data, perc)
    test = data.iloc[max(train.index)+1:]
    return train, test

def ordered_freq_apl(y_orig):
    _val_orig_psd = abs(rfft(y_orig))[1:]
    _val_freqs = rfftfreq(len(y_orig))[1:]
    imaginary_ampl = rfft(y_orig)[1:] 

    indexes = list(range(len(_val_orig_psd)))
    indexes.sort(key=lambda i: _val_orig_psd[i])
    indexes.reverse()

    ordered_ampl = np.zeros(len(indexes))
    ordered_freq = np.zeros(len(indexes))
    ordered_imag = []

    for i in range(len(indexes)):

        ordered_imag.append(imaginary_ampl[indexes[i]])
        ordered_ampl[i] = _val_orig_psd[indexes[i]]
        ordered_freq[i] = _val_freqs[indexes[i]]
    
    return ordered_freq, ordered_ampl, ordered_imag, imaginary_ampl, _val_freqs


def filter_freq_apl(ordered_freq, ordered_imag, f_thres, ampl_thres):
    low_pass_freq = []
    filtered_freq = []
    low_pass_imag = []
    filtered_imag = []

    for i in range(len(ordered_freq)):
        if ordered_freq[i] < f_thres:
            low_pass_freq.append(ordered_freq[i])
            low_pass_imag.append(ordered_imag[i])
    for j in range(len(low_pass_freq)):
        if np.absolute(low_pass_imag[j]) > ampl_thres:
            filtered_freq.append(low_pass_freq[j])
            filtered_imag.append(low_pass_imag[j])
            
    return filtered_freq, filtered_imag

def fourier_residSARIMA(key_word, y_SARIMA_train, y_SARIMA_pred, y_train, y_test, seasonal_order, coef_SARIMA, par):
    y_SARIMA_total = np.concatenate((y_SARIMA_train, y_SARIMA_pred))
    y_true_total = np.concatenate((y_train, y_test))
    y_SARIMA_skippedzeros_train = np.array(y_SARIMA_train[seasonal_order:])
    y_true_len_SARIMA = np.array(y_train[seasonal_order:])

    resid_SARIMA_skippedzeros = y_true_len_SARIMA - y_SARIMA_skippedzeros_train
    y_SARIMA_total_resid = y_true_total - y_SARIMA_total
    
    _val_orig_psd = abs(rfft(resid_SARIMA_skippedzeros))[1:]
    _val_freqs = rfftfreq(len(resid_SARIMA_skippedzeros))[1:]
    imaginary_ampl = rfft(resid_SARIMA_skippedzeros)[1:] 

    indexes = list(range(len(_val_orig_psd)))
    indexes.sort(key=lambda i: _val_orig_psd[i])
    indexes.reverse()
    
    ordered_ampl = np.zeros(len(indexes))
    filtered_freq = np.zeros(len(indexes))
    filtered_imag = []

    for i in range(len(indexes)):

        filtered_imag.append(imaginary_ampl[indexes[i]])
        filtered_freq[i] = _val_freqs[indexes[i]] 
    
    N = len(filtered_freq)
    t = np.arange(seasonal_order, len(y_SARIMA_total))
    n_train = len(t) - len(y_SARIMA_pred) 
    n_train_plus_predict = len(y_SARIMA_total)
    OLS_input_features = np.empty((N, len(t))) # n rows = Amount of selected frequencies, cols = Train + Predictions
    for i in range(N):
    
        ampli = np.absolute(filtered_imag[i]) / (len(y_SARIMA_skippedzeros_train)/2) 
        phase = np.angle(filtered_imag[i])
        OLS_input_features[i] = np.cos(2*np.pi * filtered_freq[i] * t + phase)
    
    yearly_inphase = np.zeros(N)
    indexes_to_drop = []
    for i in range(N):
        if round(OLS_input_features[i][0],5) == round(OLS_input_features[i][52], 5):
            yearly_inphase[i] = 1
        else:
            indexes_to_drop.append(i)

    interesting_freq = []
    interesting_imag = []
    for i in range(N):
        if yearly_inphase[i]==1:
            interesting_freq.append(filtered_freq[i])
            interesting_imag.append(filtered_imag[i])
    
    OLS_input_features = pd.DataFrame(OLS_input_features)
    OLS_input_features.drop(index = indexes_to_drop, inplace = True)
    
    matrix = pd.DataFrame(OLS_input_features)
    
    OLS_x_matrix = matrix.reset_index().fillna(0).drop(['index'], axis=1).T
    
    allAIC_lasso, minAIC_lasso = Lasso_AIC(OLS_x_matrix, resid_SARIMA_skippedzeros, key_word, n_train, par)
    locs_lasso = minAIC_lasso['locs']
    
    nr_par_LASSO = len(locs_lasso)
    y_train_lasso, y_pred_lasso = y_predict_L(resid_SARIMA_skippedzeros, OLS_x_matrix, n_train, n_total, locs_lasso)
    number_coeff_lasso = nr_par_LASSO + coef_SARIMA

    dummies = []
    AIC_criteria = 0
    length = len(interesting_freq)
    locs, lowest_AIC, allAIC = best_N_selection_2(dummies, AIC_criteria, OLS_x_matrix, n_train, length, y_true_len_SARIMA)
    indexlow = allAIC.index(min(allAIC))
    locs = locs[:(indexlow+1)]

    Fourier_SARIMA_resid = api.OLS(resid_SARIMA_skippedzeros.astype(float), OLS_x_matrix[0:n_train].loc[:,locs].astype(float)).fit() #Fit model based on first 156 observations in order to make a model      
    number_coeff = len(Fourier_SARIMA_resid.params) + coef_SARIMA

    resid_SARIMA_fourier_train = Fourier_SARIMA_resid.predict(OLS_x_matrix[0:n_train].loc[:,locs])
    resid_SARIMA_fourier_pred = Fourier_SARIMA_resid.predict(OLS_x_matrix[n_train::].loc[:,locs])
    
    return resid_SARIMA_skippedzeros, resid_SARIMA_fourier_train, resid_SARIMA_fourier_pred, y_SARIMA_total_resid[seasonal_order:], number_coeff, number_coeff_lasso, y_train_lasso, y_pred_lasso

def predictions(n_train, n_total, y_est_pred, dates):
    
    y_est_pred = np.array(y_est_pred)
    predictions = pd.DataFrame(np.zeros((len(y_est_pred), 2)))
    predictions.columns = ['predictions', 'date']

    for i in range(len(y_est_pred)):
        predictions.loc[i,'predictions'] = y_est_pred[i]
        predictions.loc[i,'date'] = dates.loc[n_train+shift+i][0].date()
            
    return predictions

def baseline_model(y_train, shift, n_train, n_total, order_test, xexog):
    xexog_train = xexog[0:n_train].loc[:, 1:4]

    D = 1
    orders = [[0,order_test,0], [1,order_test,0], [0,order_test,1], [1,order_test,1]]
    bestorder = 0
    aic = 1000000
    for i in range(len(orders)):
        model_fit = SARIMAX(y_train.astype(float), exog= xexog_train, trend='c', order=orders[i], seasonal_order=(1, D, 1, 52),
                        suppress_warnings=True, return_conf_int=True, error_action='ignore').fit()
        aicnew = model_fit.aic
        if aicnew < aic:
            aic = aicnew
            bestorder = orders[i]
            
    model_fit = SARIMAX(y_train.astype(float), exog= xexog_train,trend='c', order=bestorder, seasonal_order=(1, D, 1, 52),
                    suppress_warnings=True, return_conf_int=True, error_action='ignore').fit()

    if model_fit.param_names[0] == "intercept":
        pval = model_fit.pvalues[1:5]
    else:
        pval = model_fit.pvalues[0:4]
    dummies_sig = []
    for i in range(1, len(pval) + 1):
        if pval[i] < 0.05:
            dummies_sig.append(i)
        else:
            continue

    if len(dummies_sig) == 0:
        model_fit = SARIMAX(y_train.astype(float), trend='c', order=bestorder, seasonal_order=(1, D, 1, 52),
                            suppress_warnings=True, return_conf_int=True, error_action='ignore').fit()

        if np.array(model_fit.pvalues)[0] > 0.05:
            model_fit = SARIMAX(y_train.astype(float), order=bestorder, seasonal_order=(1, D, 1, 52),
                                suppress_warnings=True, return_conf_int=True, error_action='ignore').fit()

        yhat_train = model_fit.get_prediction()
        yhat_train = yhat_train._predicted_mean
        yhat_pred = model_fit.get_prediction(start=n_train, end=n_total - 1)
        yhat_pred = yhat_pred._predicted_mean

    else:
        x_regress = xexog[0:n_train].loc[:, dummies_sig]
        x_regress_pred = xexog[n_train:n_total].loc[:, dummies_sig]
        x_regress_total = xexog.loc[:, dummies_sig]
        model_fit = SARIMAX(y_train.astype(float), trend='c', exog=x_regress, order=bestorder, seasonal_order=(1, D, 1, 52),
                            error_action='ignore', suppress_warnings=True, return_conf_int=True).fit()
        if np.array(model_fit.pvalues)[0] > 0.05:
            model_fit = SARIMAX(y_train.astype(float), exog=x_regress, order=bestorder, seasonal_order=(1, D, 1, 52),
                                error_action='ignore', suppress_warnings=True, return_conf_int=True).fit()

        yhat_train = model_fit.get_prediction(exog=x_regress)
        yhat_train = yhat_train._predicted_mean
        yhat_pred = model_fit.get_prediction(start=n_train, end=n_total - 1, exog=x_regress_pred)
        yhat_pred = yhat_pred._predicted_mean

    nr_par = len(model_fit.params)
    AIC_base = model_fit.aic
    significant_dummies = dummy_significance(pval)
    seasonal_order =(1, D, 1, 52)

    return yhat_train, yhat_pred, AIC_base, seasonal_order, bestorder, nr_par, significant_dummies

##### ALGORITME VOOR MAKEN VAN DUIDELIJKE OUTPUT STREAKS & PEAKS #####
def give_streaks(predictions):

    # compute mean of predictions
    mean = predictions['predictions'].mean()
    # compute standard deviation
    st_dev = predictions['predictions'].std()

    counter_streak = 0 # houdt bij of de vorige observatie(s) boven- of ondergemiddeld waren om zo te weten of er een nieuwe streak moet beginnen
    streaks = np.zeros(len(predictions))
    peaks = np.zeros(len(predictions))


##### ALGORITME WELKE DE STREAKS & PEAKS MAAKT #####
    for i in range(len(predictions)): 
        if predictions.loc[i]['predictions'] >= mean + 2*st_dev:
            peaks[i] = 1
        if counter_streak == 0: # begin van het tellen + als je doorgaat binnen een streak
            if predictions.loc[i]['predictions'] > mean:
                streaks[i] = 1
            elif predictions.loc[i]['predictions'] <= mean:
                counter_streak = 1
            continue
        elif counter_streak == 1: # als de vorige ondergemiddeld was
            if predictions.loc[i]['predictions'] > mean:
                streaks[i] = 1
                counter_streak = 0
                if i > 1:
                    streaks[i-1] = 0.5
            elif predictions.loc[i]['predictions'] <= mean:
                counter_streak = 2
            continue
        elif counter_streak == 2: # als een streak beëindigd was/mogelijk begin van een nieuwe streak
            if predictions.loc[i]['predictions'] > mean:
                streaks[i] = 1
                counter_streak = 0
    

    # Checkt nog even of het einde en begin van de predictions (bij een jaar) een streak zijn --> geeft dan of week 51 of week 1 waarde 0.5, mochten ze zelf geen 1 zijn
    if len(predictions)%52 == 0:
        if (streaks[len(predictions)- 1] != 0) & (streaks[1] != 0) & (streaks[0] == 0):
            streaks[0] = 0.5
        elif (streaks[len(predictions) - 2] != 0) & (streaks[0] != 0) & (streaks[len(predictions) - 1] == 0):
            streaks[len(predictions) - 1] = 0.5

    name_keyword = pd.DataFrame(np.empty(len(streaks), dtype = np.str))
    length_streaks = pd.DataFrame(np.zeros(len(streaks))) # elk element is de duur van elke streak in het jaar
    indication_peak = pd.DataFrame(np.zeros(len(streaks))) 
    start_date = predictions['date']
    end_date = pd.DataFrame(pd.to_datetime(np.empty(len(streaks)))) # creates rows with value "1970-01-01" --> remove later
    insert_end_date = pd.DataFrame(make_dates(predictions.loc[0]['date'], len(predictions) + 105))
    length_streak = 0

    # Maak een 52x5 dataframe met de lengte van de streak en de bijbehorende datum (hierna alle 0'uit uit verwijderen --> overzichtelijke lijst)
    table = pd.DataFrame(pd.concat([name_keyword, length_streaks, indication_peak, start_date, end_date], axis = 1))
    table.columns = ['keyword','length', 'peak', 'start_date', 'end_date']

    # Berekent de duur van iedere streak
    for i in range(len(streaks)):
        if streaks[i] != 0:
            length_streak = length_streak + 1
        elif streaks[i] == 0:
            if i > 1:
                if streaks[i-1] != 0:
                    table.loc[i - length_streak,'length'] = length_streak # sla de lengte van de streak op in length_streaks bij de datum die hoort bij het begin van de streak
                    table.loc[i - length_streak, 'end_date'] = insert_end_date.loc[i-1]['date'] # zet de einddatum van de streak erbij
                    length_streak = 0
                else: 
                    length_streak = 0
        if (i == len(streaks) -1) & (length_streak != 0):
            table.loc[i - length_streak + 1,'length'] = length_streak # zet de lengte van de streak neer op de start van de streak (einde jaar)
            table.loc[i - length_streak + 1, 'end_date'] = insert_end_date.loc[i]['date']
            if length_streak%len(streaks) == 0: # extra safety measure voor als de streak 52 periodes is --> lenght_streak = 0
                length_streak = 0 

    # Nog een stukje code wat nog extra checkt of er een streak is die doorloopt van einde predictions tot begin predictions. Geeft dan de duur van de streak op de begindatum december/november
    if (len(predictions)%52 == 0) & (length_streak != 0) & (table.loc[0]['length'] != 0): # als de streak nog voortduurde einde jaar + als begin jaar ook een streak begon
        table.loc[len(streaks) - length_streak, 'length'] = table.loc[len(streaks) - length_streak]['length'] + table.loc[0]['length'] # zet op start streak datum einde jaar de totale lengte van de streak
        table.loc[len(streaks) - length_streak, 'end_date'] = insert_end_date.loc[len(streaks) - 1 + table.loc[0]['length']]['date'] # geef als einddatum de verwachte datum in het nieuwe jaar
        table.loc[0,'length'] = 0       


    # Stukje code wat test of de streak wel écht een streak is (met bijv meer dan 70% van de niet-0 punten > 0.5)
    for i in range(len(streaks)):
        if table.loc[i]['length'] > 0:
            max = table.loc[i]['length'] # length of streak
            sum = 0
            for j in range(int(table.loc[i]['length'])):
                if i + table.loc[i]['length'] > len(predictions) -1:
                    if i + j < len(predictions):
                        sum = sum + streaks[i+j]
                    if i + j >= len(predictions):
                        difference = i + j - len(predictions)
                        sum = sum + streaks[difference]
                else:
                    sum = sum + streaks[i+j] # sommeert waardes (aantal = lengte van de streak) in 'streaks' 
            if float(sum/max) < 0.7: # als minder dan 70% van alle niet-0 waardes 1'en zijn --> streak is ongeldig
                table.loc[i,'length'] = 0
    
    # zet pieken in de tabel
    for i in range(len(peaks)):
        if peaks[i] == 1:
            table.loc[i, 'peak'] = 1     

    # verwijdert alle rijen waar "length" < 4 en geen peak zit
    table = table[(table['length'] > 3) | (table['peak'] == 1)]   

    # reset index
    table = table.reset_index(drop = True)  

    # maakt de values integers
    table['length'] = table['length'].astype(np.int64)
    table['peak'] = table['peak'].astype(np.int64)


    # zet length < 4 met piek ook naar length = 0
    for i in range(len(table)):
        if table.loc[i]['length'] < 4:
            table.loc[i,'length'] = 0
            table.loc[i, 'end_date'] = table.loc[i]['start_date']

    # set end-date to begin-date for the peaks & remove time add
    for i in range(len(table)):
        if table.loc[i]['end_date'] == datetime(1970,1,1):
            table.loc[i,'end_date'] = table.loc[i]['start_date']
        else:
            table.loc[i,'end_date'] = pd.to_datetime(table.loc[i]['end_date']).date()

    return table

def give_streaks2(predictions, y_true):

    # if predictions is lager dan min(y_true), vervang die waarde dan door het minimum
    minimum = y_true['predictions'].min()

    # compute mean of predictions
    mean = predictions['predictions'].mean()
    # compute standard deviation
    st_dev = predictions['predictions'].std()

    for i in range(len(predictions)):
        if predictions.loc[i]['predictions'] < minimum:
            predictions.loc[i,'predictions'] = minimum

    # compute mean of predictions
    mean = predictions['predictions'].mean()
    # compute standard deviation
    st_dev = predictions['predictions'].std()

    counter_streak = 0 # houdt bij of de vorige observatie(s) boven- of ondergemiddeld waren om zo te weten of er een nieuwe streak moet beginnen
    streaks = np.zeros(len(predictions))
    peaks = np.zeros(len(predictions))


##### ALGORITME WELKE DE STREAKS & PEAKS MAAKT #####
    for i in range(len(predictions)): 
        if predictions.loc[i]['predictions'] >= mean + 2*st_dev:
            peaks[i] = 1
        if counter_streak == 0: # begin van het tellen + als je doorgaat binnen een streak
            if predictions.loc[i]['predictions'] > mean:
                streaks[i] = 1
            elif predictions.loc[i]['predictions'] <= mean:
                counter_streak = 1
            continue
        elif counter_streak == 1: # als de vorige ondergemiddeld was
            if predictions.loc[i]['predictions'] > mean:
                streaks[i] = 1
                counter_streak = 0
                if i > 1:
                    streaks[i-1] = 0.5
            elif predictions.loc[i]['predictions'] <= mean:
                counter_streak = 2
            continue
        elif counter_streak == 2: # als een streak beëindigd was/mogelijk begin van een nieuwe streak
            if predictions.loc[i]['predictions'] > mean:
                streaks[i] = 1
                counter_streak = 0
    

    # checkt nog even of het einde en begin van de predictions (bij een jaar) een streak zijn --> geeft dan of week 51 of week 1 waarde 0.5, mochten ze zelf geen 1 zijn
    if len(predictions)%52 == 0:
        if (streaks[len(predictions)- 1] != 0) & (streaks[1] != 0) & (streaks[0] == 0):
            streaks[0] = 0.5
        elif (streaks[len(predictions) - 2] != 0) & (streaks[0] != 0) & (streaks[len(predictions) - 1] == 0):
            streaks[len(predictions) - 1] = 0.5

    name_keyword = pd.DataFrame(np.empty(len(streaks), dtype = np.str))
    length_streaks = pd.DataFrame(np.zeros(len(streaks))) # elk element is de duur van elke streak in het jaar
    indication_peak = pd.DataFrame(np.zeros(len(streaks))) 
    start_date = predictions['date']
    end_date = pd.DataFrame(pd.to_datetime(np.empty(len(streaks)))) # creates rows with value "1970-01-01" --> remove later
    insert_end_date = pd.DataFrame(make_dates(predictions.loc[0]['date'], len(predictions) + 105))
    length_streak = 0

    # maak een 52x5 dataframe met de lengte van de streak en de bijbehorende datum (hierna alle 0'uit uit verwijderen --> overzichtelijke lijst)
    table = pd.DataFrame(pd.concat([name_keyword, length_streaks, indication_peak, start_date, end_date], axis = 1))
    table.columns = ['keyword','length', 'peak', 'start_date', 'end_date']

    # berekent de duur van iedere streak
    for i in range(len(streaks)):
        if streaks[i] != 0: 
            length_streak = length_streak + 1
        elif streaks[i] == 0:
            if i >= 1:
                if streaks[i-1] != 0:
                    table.loc[i - length_streak,'length'] = length_streak # sla de lengte van de streak op in length_streaks bij de datum die hoort bij het begin van de streak
                    table.loc[i - length_streak, 'end_date'] = insert_end_date.loc[i-1]['date'] # zet de einddatum van de streak erbij
                    length_streak = 0
                else: 
                    length_streak = 0
        if (i == len(streaks) -1) & (length_streak != 0):
            table.loc[i - length_streak + 1,'length'] = length_streak # zet de lengte van de streak neer op de start van de streak (einde jaar)
            table.loc[i - length_streak + 1, 'end_date'] = insert_end_date.loc[i]['date']
            if length_streak%len(streaks) == 0: # extra safety measure voor als de streak 52 periodes is --> lenght_streak = 0
                length_streak = 0 

    # nog een stukje code wat nog extra checkt of er een streak is die doorloopt van einde predictions tot begin predictions. Geeft dan de duur van de streak op de begindatum december/november
    if (len(predictions)%52 == 0) & (length_streak != 0) & (table.loc[0]['length'] != 0): # als de streak nog voortduurde einde jaar + als begin jaar ook een streak begon
        table.loc[len(streaks) - length_streak, 'length'] = table.loc[len(streaks) - length_streak]['length'] + table.loc[0]['length'] # zet op start streak datum einde jaar de totale lengte van de streak
        table.loc[len(streaks) - length_streak, 'end_date'] = insert_end_date.loc[len(streaks) - 1 + table.loc[0]['length']]['date'] # geef als einddatum de verwachte datum in het nieuwe jaar
        table.loc[0,'length'] = 0       


    # stukje code wat test of de streak wel écht een streak is (met bijv meer dan 70% van de niet-0 punten > 0.5)
    for i in range(len(streaks)):
        if table.loc[i]['length'] > 0:
            max = table.loc[i]['length'] # length of streak
            sum = 0
            for j in range(int(table.loc[i]['length'])):
                if i + table.loc[i]['length'] > len(predictions) -1:
                    if i + j < len(predictions):
                        sum = sum + streaks[i+j]
                    if i + j >= len(predictions):
                        difference = i + j - len(predictions)
                        sum = sum + streaks[difference]
                else:
                    sum = sum + streaks[i+j] # sommeert waardes (aantal = lengte van de streak) in 'streaks' 
            if float(sum/max) < 0.7: # als minder dan 70% van alle niet-0 waardes 1'en zijn --> streak is ongeldig
                table.loc[i,'length'] = 0
    
    # zet pieken in de tabel
    for i in range(len(peaks)):
        if peaks[i] == 1:
            table.loc[i, 'peak'] = 1     

    # verwijdert alle rijen waar "length" < 4 en geen peak zit
    table = table[(table['length'] > 3) | (table['peak'] == 1)]   

    # reset index
    table = table.reset_index(drop = True)  

    # maakt de values integers
    table['length'] = table['length'].astype(np.int64)
    table['peak'] = table['peak'].astype(np.int64)


    # zet length < 4 met piek ook naar length = 0
    for i in range(len(table)):
        if table.loc[i]['length'] < 4:
            table.loc[i,'length'] = 0
            table.loc[i, 'end_date'] = table.loc[i]['start_date']

    # set end-date to begin-date for the peaks & remove time add
    for i in range(len(table)):
        if table.loc[i]['end_date'] == datetime(1970,1,1):
            table.loc[i,'end_date'] = table.loc[i]['start_date']
        else:
            table.loc[i,'end_date'] = pd.to_datetime(table.loc[i]['end_date']).date()

    #print(table)
    return table

def give_prediction_score2(evaluation, evaluation_true):

    accuracy_score_peaks = 0
    accuracy_rate_peaks = 0
    number_of_peaks = 0
    accuracy_score_streaks = 0
    accuracy_rate_streaks = 0
    total_weeks_streaks = 0

    number_of_peaks_predicted = evaluation['peak'].sum()
    total_weeks_streaks_predicted = evaluation['length'].sum()

    tick_off_peaks = pd.DataFrame(np.zeros(len(evaluation)))
    tick_off_streaks = pd.DataFrame(np.zeros(len(evaluation)))

    tick_off_peaks_true = pd.DataFrame(np.zeros(len(evaluation_true)))
    tick_off_streaks_true = pd.DataFrame(np.zeros(len(evaluation_true)))
    
    evaluation = pd.DataFrame(pd.concat([evaluation, tick_off_peaks, tick_off_streaks], axis = 1))
    evaluation.columns = ['keyword','length', 'peak', 'start_date', 'end_date', 'tick_off_peaks', 'tick_off_streaks']

    evaluation_true = pd.DataFrame(pd.concat([evaluation_true, tick_off_peaks_true, tick_off_streaks_true], axis = 1))
    evaluation_true.columns = ['keyword','length', 'peak', 'start_date', 'end_date', 'tick_off_peaks', 'tick_off_streaks']

    for i in range(len(evaluation_true)):

        # peaks
        if evaluation_true.loc[i]['peak'] == 1: 
            number_of_peaks = number_of_peaks + 1
            date_peak = evaluation_true.loc[i]['start_date']
            score_current_peak = 0 # dit bekijkt de score die een true peak gekregen heeft
            score_current_peak_prev = 0
            for j in range(len(evaluation)):  # gaat hier alle predicted peaks af 
                if evaluation.loc[j]['peak'] == 1: # kijkt of een piek gepredict is en probeert deze te matchen met de true piek
                    date_peak_predicted = evaluation.loc[j]['start_date']
                    if date_peak == date_peak_predicted:  # als hij perfect gepredict is
                        score_current_peak_add = 1
                        evaluation.loc[j,'tick_off_peaks'] = 1 # geeft aan dat de predicted peak gekoppeld is aan een true peak (hoeft niet de uiteindelijke koppeling te zijn, indien beter, maar verdient geen penalty term)
                        if score_current_peak_add > score_current_peak:
                            score_current_peak = score_current_peak_add
                            accuracy_score_peaks = accuracy_score_peaks + score_current_peak - score_current_peak_prev
                            score_current_peak_prev = score_current_peak_add
                    else: # nog deelpunten krijgen als het 1-2 weken verschilt
                        if date_peak < date_peak_predicted: 
                            start = date_peak
                            end = date_peak_predicted
                        elif date_peak > date_peak_predicted:
                            start = date_peak_predicted
                            end = date_peak
                        weeks_between = rrule.rrule(rrule.WEEKLY, dtstart= start, until = end).count() - 1
                        if weeks_between == 1:
                            score_current_peak_add = 0.95 
                            evaluation.loc[j,'tick_off_peaks'] = 1 # geeft aan dat de predicted peak gekoppeld is aan een true peak (hoeft niet de uiteindelijke koppeling te zijn, indien beter, maar verdient geen penalty term)
                            if score_current_peak_add > score_current_peak:
                                score_current_peak = score_current_peak_add
                                accuracy_score_peaks = accuracy_score_peaks + score_current_peak  - score_current_peak_prev # als je er één week naast zit
                                score_current_peak_prev = score_current_peak_add
                        elif weeks_between == 2:
                            score_current_peak_add = 0.8
                            evaluation.loc[j,'tick_off_peaks'] = 1 # geeft aan dat de predicted peak gekoppeld is aan een true peak (hoeft niet de uiteindelijke koppeling te zijn, indien beter, maar verdient geen penalty term)
                            if score_current_peak_add > score_current_peak:
                                score_current_peak = score_current_peak_add 
                                accuracy_score_peaks = accuracy_score_peaks + score_current_peak - score_current_peak_prev # als je er twee weken naast zit
                                score_current_peak_prev = score_current_peak_add
                        elif weeks_between == 3:
                            score_current_peak_add = 0.4
                            evaluation.loc[j,'tick_off_peaks'] = 1 # geeft aan dat de predicted peak gekoppeld is aan een true peak (hoeft niet de uiteindelijke koppeling te zijn, indien beter, maar verdient geen penalty term)
                            if score_current_peak_add > score_current_peak:
                                score_current_peak = score_current_peak_add 
                                accuracy_score_peaks = accuracy_score_peaks + score_current_peak - score_current_peak_prev # als je er twee weken naast zit
                                score_current_peak_prev = score_current_peak_add                                 
            for j in range(len(evaluation)):  
                if score_current_peak == 0: # de piek is niet gepredict --> kijk of ie in een predicted streak valt. Gaat hier alle predicted streaks af
                    if evaluation.loc[j]['start_date'] <= date_peak <= evaluation.loc[j]['end_date']: # checkt of de piek (indien niet gepredict) binnen streak valt
                        accuracy_score_peaks = accuracy_score_peaks + 0.3
                        evaluation_true.loc[i,'tick_off_peaks'] = 1
                        break
                    try:
                        lower = evaluation.loc[j]['start_date'].replace(day = evaluation.loc[j]['start_date'].day - 2)
                    except ValueError as ve:
                        try:
                            lower = evaluation.loc[j]['start_date'].replace(month = evaluation.loc[j]['start_date'].month - 1, day = 28)
                        except ValueError as VE:
                            lower = evaluation.loc[j]['start_date'].replace(year = evaluation.loc[j]['start_date'].year - 1, month = 12, day = 28)
                    try:
                        higher = evaluation.loc[j]['end_date'].replace(day = evaluation.loc[j]['end_date'].day + 2)
                    except ValueError as ve:
                        try:
                            higher = evaluation.loc[j]['end_date'].replace(month = evaluation.loc[j]['end_date'].month + 1, day = 2)
                        except ValueError as VE:
                            higher = evaluation.loc[j]['end_date'].replace(year = evaluation.loc[j]['end_date'].year + 1, month = 1, day = 1)
                    if lower <= date_peak.replace(year = date_peak.year + 1) <= higher:
                        accuracy_score_peaks = accuracy_score_peaks + 0.3
                        evaluation_true.loc[i,'tick_off_peaks'] = 1
                        break
                    try:
                        lower = evaluation.loc[j]['start_date'].replace(day = evaluation.loc[j]['start_date'].day - 2)
                    except ValueError as ve:
                        try:
                            lower = evaluation.loc[j]['start_date'].replace(month = evaluation.loc[j]['start_date'].month - 1, day = 28)
                        except ValueError as VE:
                            lower = evaluation.loc[j]['start_date'].replace(year = evaluation.loc[j]['start_date'].year - 1, month = 12, day = 28)
                    try: 
                        higher = evaluation.loc[j]['end_date'].replace(day = evaluation.loc[j]['end_date'].day + 2)
                    except ValueError as ve:
                        try:
                            higher = evaluation.loc[j]['end_date'].replace(month = evaluation.loc[j]['end_date'].month + 1, day = 2)
                        except:
                            higher = evaluation.loc[j]['end_date'].replace(year = evaluation.loc[j]['end_date'].year + 1, month = 1, day = 2)
                    if lower <= date_peak.replace(year = date_peak.year - 1) <= higher:
                        accuracy_score_peaks = accuracy_score_peaks + 0.3
                        evaluation_true.loc[i,'tick_off_peaks'] = 1
                        break
        # streaks
        if evaluation_true.loc[i]['length'] != 0: # loopje wat alle streaks afgaat
            streak_dates = make_dates(evaluation_true.loc[i]['start_date'], evaluation_true.loc[i]['length'])
            total_weeks_streaks = total_weeks_streaks + evaluation_true.loc[i]['length']
            for k in range(len(evaluation)): # loopje wat alle predicted streaks afgaat (zie if 2 regels later)
                for j in range(len(streak_dates)): # loopje voor alle weken binnen een true streak
                    if evaluation.loc[k]['length'] != 0: # mogelijke predicted streak waar de data van de huidige true streak in zouden kunnen zitten
                        if evaluation.loc[k]['start_date'] <= streak_dates.loc[j]['date'] <= evaluation.loc[k]['end_date']: # als de week van de huidige true streak in een predicted streak zit
                            accuracy_score_streaks = accuracy_score_streaks + 1
                            evaluation.loc[k,'tick_off_streaks'] = 1 # checkt 1x dat de predicted streak gebruikt is
                            evaluation_true.loc[i, 'tick_off_streaks'] = 1 # checkt 1x dat de true streak gebruikt is
                            continue
                        try:
                            lower = evaluation.loc[k]['start_date'].replace(day = evaluation.loc[k]['start_date'].day - 2)
                        except ValueError as ve:
                            try:
                                lower = evaluation.loc[k]['start_date'].replace(month = evaluation.loc[k]['start_date'].month - 1, day = 28)
                            except ValueError as VE:
                                lower = evaluation.loc[k]['start_date'].replace(year = evaluation.loc[k]['start_date'].year - 1, month = 12, day = 28)                        
                        try:
                            higher = evaluation.loc[k]['end_date'].replace(day = evaluation.loc[k]['end_date'].day + 2)
                        except ValueError as ve:
                            try:
                                higher = evaluation.loc[k]['end_date'].replace(month = evaluation.loc[k]['end_date'].month + 1, day = 1)
                            except ValueError as VE:
                                higher = evaluation.loc[k]['end_date'].replace(year = evaluation.loc[k]['end_date'].year + 1, month = 1, day = 1)
                        if lower <= streak_dates.loc[j]['date'].replace(year = streak_dates.loc[j]['date'].year - 1) <= higher:
                            accuracy_score_streaks = accuracy_score_streaks + 1
                            evaluation.loc[k,'tick_off_streaks'] = 1 # checkt 1x dat de predicted streak gebruikt is
                            evaluation_true.loc[i, 'tick_off_streaks'] = 1 # checkt 1x dat de true streak gebruikt is
                            continue
                        try:
                            lower = evaluation.loc[k]['start_date'].replace(day = evaluation.loc[k]['start_date'].day - 2)
                        except ValueError as ve:
                            try:
                                lower = evaluation.loc[k]['start_date'].replace(month = evaluation.loc[k]['start_date'].month - 1, day = 28)
                            except ValueError as VE:
                                lower = evaluation.loc[k]['start_date'].replace(year = evaluation.loc[k]['start_date'].year - 1, month = 12, day = 28)
                        try: # gewoon 2 dagen verder
                            higher = evaluation.loc[k]['end_date'].replace(day = evaluation.loc[k]['end_date'].day + 2)
                        except ValueError as ve:
                            try: # als je dan net de maandsgrens over gaat
                                higher = evaluation.loc[k]['end_date'].replace(month = evaluation.loc[k]['end_date'].month + 1, day = 1)
                            except ValueError as VE: # als je dan net de jaargrens over gaat
                                higher = evaluation.loc[k]['end_date'].replace(year = evaluation.loc[k]['end_date'].year + 1, month = 1, day = 1)
                        if lower <= streak_dates.loc[j]['date'].replace(year = streak_dates.loc[j]['date'].year + 1) <= higher:
                            accuracy_score_streaks = accuracy_score_streaks + 1
                            evaluation.loc[k,'tick_off_streaks'] = 1 # checkt 1x dat de predicted streak gebruikt is
                            evaluation_true.loc[i, 'tick_off_streaks'] = 1 # checkt 1x dat de true streak gebruikt is
                            continue

    # penalty terms - false positives (minder erg)
    if evaluation['peak'].sum() - evaluation['tick_off_peaks'].sum() > 0: # true if there is at least one false positive peak --> penalty term
        for i in range(len(evaluation)):
            if (evaluation.loc[i]['tick_off_peaks'] == 0) & (evaluation.loc[i]['peak'] == 1): # deze peak is dus nog niet gebruikt (false positive) --> penalty term
                for j in range(len(evaluation_true)):    # checkt of de predicted streak wel nog in een true streak valt --> penalty vermindering
                    if evaluation_true.loc[j]['length'] != 0: # in een streak
                        if evaluation_true.loc[j]['start_date'] <= evaluation.loc[i]['start_date'] <= evaluation_true.loc[j]['end_date']: # predicted peak ligt in een true streak
                            accuracy_score_peaks = accuracy_score_peaks - 0.2
                            break
                    else: # predicted peak valt ook niet in een true streak
                        accuracy_score_peaks = accuracy_score_peaks - 0.4 
                        break
    if (evaluation['length']!= 0).sum() - evaluation['tick_off_streaks'].sum() > 0: # true if there are streaks (in totallity) that are predicted that don't exist
        for i in range(len(evaluation)):
            if (evaluation.loc[i]['tick_off_streaks'] == 0) & (evaluation.loc[i]['length'] != 0): # alleen penalty als er een hele streak gepredict is die niet bestaat (zelfs niet één week overlap)
                accuracy_score_streaks = accuracy_score_streaks - evaluation.loc[i]['length']*0.3


    # penalty terms - false negatives (erger)
    if evaluation_true['peak'].sum() - evaluation['tick_off_peaks'].sum() - evaluation_true['tick_off_peaks'].sum() > 0: # true if there is at least one false negative peak (true peak niet gekoppeld aan predicted streak of peak) --> penalty term
        accuracy_score_peaks = accuracy_score_peaks - 0.75*(evaluation_true['peak'].sum() - evaluation['tick_off_peaks'].sum() - evaluation_true['tick_off_peaks'].sum())

    if (evaluation_true['length']!= 0).sum() - evaluation_true['tick_off_streaks'].sum() > 0: # true if there are streaks (in totallity) that are not predicted (dus aantal true peaks - aantal gematchte - aantal true streaks in predicted peaks)
        for i in range(len(evaluation_true)):
            if (evaluation_true.loc[i]['tick_off_streaks'] == 0) & (evaluation_true.loc[i]['length'] != 0): # alleen penalty als er een hele streak is die niet gepredict is (zelfs niet één week overlap)
                accuracy_score_streaks = accuracy_score_streaks - evaluation_true.loc[i]['length']*0.5

    # maak accuracy scores (percentage)
    if number_of_peaks_predicted > 0:
        accuracy_rate_peaks = accuracy_score_peaks / float(number_of_peaks_predicted) * 100 # bereken de accuracy als percentage
    elif number_of_peaks_predicted == 0:
        accuracy_rate_peaks = -999

    if total_weeks_streaks_predicted > 0:
        accuracy_rate_streaks = accuracy_score_streaks / float(total_weeks_streaks_predicted) * 100 # bereken de accuracy als percentage
    elif total_weeks_streaks_predicted ==0: # in het geval er enkel false positive streaks zijn, geen true streaks
        accuracy_rate_streaks = -999

    # mits zowel pieken als streaks bestaan --> bereken gemiddelde accuracy
    if (number_of_peaks_predicted > 0) & (total_weeks_streaks_predicted > 0):
        average_accuracy = accuracy_rate_peaks/3 + 2/3*accuracy_rate_streaks
    elif (accuracy_rate_peaks == -999) & (accuracy_rate_streaks != -999):
        average_accuracy = accuracy_rate_streaks
    elif (accuracy_rate_streaks == -999) & (accuracy_rate_peaks != -999):
        average_accuracy = accuracy_rate_peaks
    else:
        average_accuracy = -999
    
    return accuracy_rate_peaks, accuracy_rate_streaks, average_accuracy

def make_dates(starting_date, number_of_weeks):
    dates = pd.DataFrame(pd.date_range(starting_date, periods=number_of_weeks, freq='W'), columns = ['date'])

    return dates    

def get_unique_keywords(all_data, country):
    data_nl = all_data[all_data['countryCode'] == country] #Filter data on a country (NL)
    unique_keywords = np.unique(data_nl['keyword'])

    return list(unique_keywords)

def evaluate_zeros(keyword_data):
    check_keyword_data = np.array(keyword_data['interest'])
    
    if (np.count_nonzero(check_keyword_data==0)/len(check_keyword_data)) > 0.9:
        return "more than 90% zeros"
    else:
        return "Less than 90% zeros"
    
def add_monthly_dummies(data_with_holidays):
    jan = pd.DataFrame(np.zeros(len(data_with_holidays)))
    feb = pd.DataFrame(np.zeros(len(data_with_holidays)))
    mar = pd.DataFrame(np.zeros(len(data_with_holidays)))
    apr = pd.DataFrame(np.zeros(len(data_with_holidays)))
    may = pd.DataFrame(np.zeros(len(data_with_holidays)))
    jun = pd.DataFrame(np.zeros(len(data_with_holidays)))
    jul = pd.DataFrame(np.zeros(len(data_with_holidays)))
    aug = pd.DataFrame(np.zeros(len(data_with_holidays)))
    sep = pd.DataFrame(np.zeros(len(data_with_holidays)))
    oct = pd.DataFrame(np.zeros(len(data_with_holidays)))
    nov = pd.DataFrame(np.zeros(len(data_with_holidays)))
    dec = pd.DataFrame(np.zeros(len(data_with_holidays)))

    for i in range(len(data_with_holidays)):
        if data_with_holidays.loc[i]['date'].month == 1:
            jan.loc[i,0] = 1
        if data_with_holidays.loc[i]['date'].month == 2:
            feb.loc[i,0] = 1
        if data_with_holidays.loc[i]['date'].month == 3:
            mar.loc[i,0] = 1
        if data_with_holidays.loc[i]['date'].month == 4:
            apr.loc[i,0] = 1        
        if data_with_holidays.loc[i]['date'].month == 5:
            may.loc[i,0] = 1
        if data_with_holidays.loc[i]['date'].month == 6:
            jun.loc[i,0] = 1
        if data_with_holidays.loc[i]['date'].month == 7:
            jul.loc[i,0] = 1
        if data_with_holidays.loc[i]['date'].month == 8:
            aug.loc[i,0] = 1
        if data_with_holidays.loc[i]['date'].month == 9:
            sep.loc[i,0] = 1    
        if data_with_holidays.loc[i]['date'].month == 10:
            oct.loc[i,0] = 1
        if data_with_holidays.loc[i]['date'].month == 11:
            nov.loc[i,0] = 1
        if data_with_holidays.loc[i]['date'].month == 12:
            dec.loc[i,0] = 1

    data_with_holidays_and_monthly_dummies = pd.DataFrame(pd.concat([data_with_holidays, jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec], axis = 1))
    data_with_holidays_and_monthly_dummies.columns = ['date', 'interest', 'pasen', 'hemelvaart', 'pinksteren', 'kerst', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug','sep', 'oct', 'nov', 'dec']
    return data_with_holidays_and_monthly_dummies

def seasonality_check(data_keyword):

    data_with_holidays = add_holidays(data_keyword)
    data_with_holidays_and_monthly_dummies = add_monthly_dummies(data_with_holidays)

    #### TEST VOOR MONTHLY SEASONALITY ####
    data_with_holidays_and_monthly_dummies.drop(data_with_holidays_and_monthly_dummies.index[208:259],0, inplace = True)
    data_with_holidays_and_monthly_dummies.drop(data_with_holidays_and_monthly_dummies.index[208:259],0, inplace = True)

    y = data_with_holidays_and_monthly_dummies['interest']
    data_with_holidays_and_monthly_dummies.drop('interest', inplace = True, axis = 1)
    data_with_holidays_and_monthly_dummies.drop('pasen', inplace = True, axis = 1)
    data_with_holidays_and_monthly_dummies.drop('hemelvaart', inplace = True, axis = 1)
    data_with_holidays_and_monthly_dummies.drop('pinksteren', inplace = True, axis = 1)
    data_with_holidays_and_monthly_dummies.drop('kerst', inplace = True, axis = 1)
    data_with_holidays_and_monthly_dummies.drop('date', inplace = True, axis = 1)
    data_with_holidays_and_monthly_dummies.drop('dec', inplace = True, axis = 1)

    data_with_holidays_and_monthly_dummies = sm.add_constant(data_with_holidays_and_monthly_dummies, prepend = True)

    seasonality= api.OLS(y, data_with_holidays_and_monthly_dummies).fit()

    x_week = [[1],
        [1],
        [1],
        [1]]

    X_dummy_week = pd.DataFrame(np.kron(x_week,np.identity(52)))
    X_dummy_week.drop(51,inplace = True, axis = 1)
    X_dummy_week = sm.add_constant(X_dummy_week, prepend = True)

    seas_model = api.OLS(y, X_dummy_week).fit() 

    # test for seasonality
    if (seasonality.f_pvalue <= 0.05) & (seas_model.f_pvalue <= 0.05):
        season = 'weekly and monthly'

    elif seasonality.f_pvalue <= 0.05:
        season = 'monthly'

    elif seas_model.f_pvalue <= 0.05:
        season = 'weekly'

    else: #(seasonality.f_pvalue > 0.05) & (seas_model.f_pvalue > 0.05):
        season = 'not seasonal'

    return season

def check_less_than_100(keyword_data):
    check_keyword_data = np.array(keyword_data['interest'])

    if ((check_keyword_data < 100).sum()/len(check_keyword_data)) > 0.9:
        status = "More than 90% of interest < 100"
    else:
        status = "Less than 90% of interest > 100"
    
    return status

def retrieve_1yearfreq(keyword_data, filtered_imag, filtered_freq, y_train):
    N = len(filtered_freq)
    t = np.arange(0, len(keyword_data))
    OLS_input_features = np.empty((N, len(t))) # n rows = Amount of selected frequencies, cols = Train + Predictions
    for i in range(N):
    
        phase = np.angle(filtered_imag[i])
        OLS_input_features[i] = np.cos(2*np.pi * filtered_freq[i] * t + phase)
    
    yearly_inphase = np.zeros(N)
    indexes_to_drop = []
    for i in range(N):
        if round(OLS_input_features[i][0],5) == round(OLS_input_features[i][52], 5):
            yearly_inphase[i] = 1
        else:
            indexes_to_drop.append(i)
    
    interesting_freq = []
    interesting_imag = []
    for i in range(N):
        if yearly_inphase[i]==1:
            interesting_freq.append(filtered_freq[i])
            interesting_imag.append(filtered_imag[i])
    
    OLS_input_features = pd.DataFrame(OLS_input_features)
    OLS_input_features.drop(index = indexes_to_drop, inplace = True)
    
    matrix = pd.concat([np.transpose(pd.DataFrame(np.array(data_with_holidays['pasen'][0:OLS_input_features.shape[1]]))), 
                        np.transpose(pd.DataFrame(np.array(data_with_holidays['hemelvaart'][0:OLS_input_features.shape[1]]))), 
                        np.transpose(pd.DataFrame(np.array(data_with_holidays['pinksteren'][0:OLS_input_features.shape[1]]))), 
                        np.transpose(pd.DataFrame(np.array(data_with_holidays['kerst'][0:OLS_input_features.shape[1]]))), 
                        pd.DataFrame(OLS_input_features)], axis=0)
    
    OLS_x_matrix = matrix.reset_index().fillna(0).drop(['index'], axis=1).T
    
    return interesting_freq, OLS_x_matrix, interesting_imag

def determine_AIC_2(locs, OLS_x_matrix, n_train, N, y_train):
    AIC_values = []
    LR_values = []
    locations = locs
    for i in range(N):
        if i in locations:
            AIC_values.append(100000)
            continue
        
        locations = locations+[i]
        model = api.OLS(y_train.astype(float), np.array(OLS_x_matrix[0:n_train].loc[:, locations].astype(float))).fit() #Fit model based on first 156 observations in order to make a model
        AIC_values.append(model.aic + 2*(len(locations)-5))
        LR_values.append(model.llf)
        locations.remove(i)

    lowestAIC = min(AIC_values)
    loclowest = AIC_values.index(min(AIC_values))
    return lowestAIC, loclowest, AIC_values

def best_N_selection_2(dummies, AIC_criteria, OLS_x_matrix, n_train, length, y_train):
    locs2 = dummies
    N = length
    lowest_AIC = 1000000
    totalAIC = []
    for i in range(N-len(locs2)):
        AIC, location, AIC_values_all = determine_AIC_2(locs2, OLS_x_matrix, n_train, length, y_train)
        totalAIC.append(AIC)
        locs2.append(location)
        lowest_AIC = AIC
    return locs2, lowest_AIC, totalAIC

def dummy_significance(pval):
    dummies = []

    if pval[1] < 0.05:
        dummies = dummies+["pasen"]
    if pval[2] < 0.05:
        dummies = dummies+["hemelvaart"]
    if pval[3] < 0.05:
        dummies = dummies+["pinksteren"]
    if pval[4] < 0.05:
        dummies = dummies + ["kerst"]
    return dummies

def y_predict(y_train, OLS_x_matrix, n_train, n_total, locs):
    final_model = api.OLS(y_train.astype(float), OLS_x_matrix[0:n_train].loc[:,locs].astype(float)).fit() #Fit model based on first 156 observations in order to make a model
    significance_dummies = dummy_significance(final_model.pvalues) 
    indexed_to_drop = []
    if len(significance_dummies) == 0:
        indexed_to_drop = [1,2,3,4]
    else:
        for i in range(len(significance_dummies)):
            if "pasen" not in significance_dummies:
                indexed_to_drop.append(1)
            if "hemelvaart" not in significance_dummies:
                indexed_to_drop.append(2)
            if "pinksteren" not in significance_dummies:
                indexed_to_drop.append(3)
            if "kerst" not in significance_dummies:
                indexed_to_drop.append(4)
    if final_model.pvalues[0] > 0.05:
        indexed_to_drop.append(0)
        
    locs = [elem for elem in locs if elem not in indexed_to_drop]
    final_model = api.OLS(y_train.astype(float), OLS_x_matrix[0:n_train].loc[:,locs].astype(float)).fit() #Fit model based on first 156 observations in order to make a model
    final_AIC = final_model.aic
    
    y_est_train = final_model.predict(OLS_x_matrix[0:n_train].loc[:,locs]) 
    y_est_pred = final_model.predict(OLS_x_matrix[n_train:n_total].loc[:,locs])#Predict the dependent variable for 100 observations of x
    return y_est_train, y_est_pred, final_AIC, significance_dummies, locs

def logl_aic(n_obs, max_lag, y_true, y_pred, nr_par):
    sigma = (1/(n_obs-max_lag-2))*np.sum((np.array(y_true[max_lag:n_obs]) - np.array(y_pred[max_lag:n_obs]))**2)
    logl = -((n_obs-max_lag)/2)*np.log(2*np.pi) - ((n_obs-max_lag)/2)*np.log(sigma)-(1/(2* sigma))*np.sum((np.array(y_true[max_lag:n_obs])-np.array(y_pred[max_lag:n_obs]))**2)
    aic = -2*logl + 2*nr_par
    return sigma, logl, aic

def logl_aic_fourier(n_obs, max_lag, y_true, y_pred, nr_par, sign_dummies):
    
    sigma = (1/(n_obs-max_lag-2))*np.sum((np.array(y_true[max_lag:n_obs]) - np.array(y_pred[max_lag:n_obs]))**2)
    logl = -((n_obs-max_lag)/2)*np.log(2*np.pi) - ((n_obs-max_lag)/2)*np.log(sigma)-(1/(2* sigma))*np.sum((np.array(y_true[max_lag:n_obs])-np.array(y_pred[max_lag:n_obs]))**2)
    aic = -2*logl + 4*(nr_par-sign_dummies) + 2*sign_dummies
    return sigma, logl, aic

def logl_aic_fourier_SARIMAF(n_obs, max_lag, y_true, y_pred, nr_par, sign_dummies, max_order_y_true, n_obs_true):
    
    sigma = (1/(n_obs-max_lag-2))*np.sum((np.array(y_true[max_order_y_true:n_obs_true]) - np.array(y_pred[max_lag:n_obs]))**2)
    logl = -((n_obs-max_lag)/2)*np.log(2*np.pi) - ((n_obs-max_lag)/2)*np.log(sigma)-(1/(2* sigma))*np.sum((np.array(y_true[max_order_y_true:n_obs_true])-np.array(y_pred[max_lag:n_obs]))**2)
    aic = -2*logl + 4*(nr_par-sign_dummies) + 2*sign_dummies
    return sigma, logl, aic


def diff_order(y_train):
    adftest = ndiffs(y_train, test='kpss')
    return adftest

def Lasso_AIC(OLS_x_matrix, y_train, key_word, n_train, par):
    locs1 = []
    locs2 = []
    aicvalues = []
    alphavalues = []
    for alpha in np.arange(0,40,0.5):
        model2 = Lasso(alpha= alpha)
        model2.fit(np.array(OLS_x_matrix[0:len(y_train)]), np.array(y_train))
        locs = np.nonzero(model2.coef_)[0].tolist()
        if locs == []:
            break
        model = api.OLS(y_train.astype(float), np.array(OLS_x_matrix[0:n_train].loc[:, locs].astype(float))).fit() 
        
        if par == 1:
            penalty = copy.deepcopy(locs)
            for j in range(0,4):
                if j in penalty:
                    penalty.remove(j)

            AIC_value = model.aic + 2*len(penalty)
            aicvalues.append(AIC_value)
        else:
            AIC_value = model.aic + 2*len(locs)
            aicvalues.append(AIC_value)

        alphavalues.append(alpha)
        locs1.append(locs)
        locs2.append(len(locs))
    
    key_word_list = list([key_word]) * len(locs2) 
    results = pd.concat([pd.Series(key_word_list, name='key_word'), pd.Series(locs1, name='locs'), pd.Series(locs2, name='length_locs'), pd.Series(alphavalues, name='alpha_value'), pd.Series(aicvalues, name='aic_values')], axis=1)
    best_result = results.iloc[results['aic_values'].idxmin(), :]
   
    return results, best_result

def y_predict_L(y_train, OLS_x_matrix, n_train, n_total, locs):
    final_model = api.OLS(y_train.astype(float), OLS_x_matrix[0:n_train].loc[:,locs].astype(float)).fit() #Fit model based on first 156 observations in order to make a model
    y_est_train = final_model.predict(OLS_x_matrix[0:n_train].loc[:,locs]) 
    y_est_pred = final_model.predict(OLS_x_matrix[n_train:n_total].loc[:,locs])#Predict the dependent variable for 100 observations of x
    return y_est_train, y_est_pred


# =============================================================================
# 3. Running code
# =============================================================================

all_data = pd.read_csv('data seasonality usecase.csv', sep=";")
pdf = matplotlib.backends.backend_pdf.PdfPages("outputfile.pdf")

keywords_monthly = list()
keywords_weekly = list()
keywords_both = list()
keywords_no = list()
keywords_nointerest_seasonal = list()
keywords_nointerest_not_seasonal = list()

accuracy_scores = pd.DataFrame(columns = ['keyword',
                    "FOURIER",
                    "FOURIER lasso",
                    "FARIMA",
                    "FARIMA lasso",
                    "SARIMA",
                    "SARIMAF",
                    "SARIMAF lasso"])

N_included = pd.DataFrame(columns = ['keyword',
                  "FOURIER",
                  "FOURIER lasso",
                  "FARIMA",
                  "FARIMA lasso",
                  "SARIMA",
                  "SARIMAF",
                  "SARIMAF lasso"])  
  
AIC_scores = pd.DataFrame(columns = ['keyword',
                                     "FOURIER",
                                     "FOURIER lasso",
                                     "FARIMA",
                                     "FARIMA lasso",
                                     "SARIMA",
                                     "SARIMAF",
                                     "SARIMAF lasso"])

streaks_and_peaks = pd.DataFrame(columns = ['keyword',
                                     "FOURIER",
                                     "FOURIER lasso",
                                     "FARIMA",
                                     "FARIMA lasso",
                                     "SARIMA",
                                     "SARIMAF",
                                     "SARIMAF lasso"])


j = 1
k = 0
p = 2

cycles_per_period = []
cycles_per_period_lasso = []

resultaten_aic = []
resultaten_lasso = []

total_predictions = []

for i in range(0,2):
    
    keyword_data, key_word = data_keyword_index(all_data, i, "NL")
    ### If one wants to model only for a single keyword (dish), you can use the single line below.
    # keyword_data = data_keyword_str(all_data, "hazenbout", "NL") 
    key_word = key_word[i]
    keyword_data = keyword_data.drop(['keyword'], axis=1)
    
    ### Seasonality test
    if seasonality_check(keyword_data) == "not seasonal":
        keywords_no.append(key_word)
        if check_less_than_100(keyword_data) == "More than 90% of interest < 100":
            keywords_nointerest_not_seasonal.append(key_word)      
    else:
        if seasonality_check(keyword_data) == "weekly":
            keywords_weekly.append(key_word)
        elif seasonality_check(keyword_data) == "monthly":
            keywords_monthly.append(key_word)
        else:
            keywords_both.append(key_word)

        data_with_holidays = add_holidays(keyword_data)

        keyword_data, pval, beta_trend = detrending2(keyword_data)
        
        ### Define the length of the train set
        perc_train = 0.75
        shift_div = len(keyword_data)*perc_train
        shift = int(shift_div%52)

        train, test = train_test_split(keyword_data, perc_train)
        y_train = train['interest']
        y_test = test['interest']
        
        ### Get the frequencies and amplitudes from the fast fourier transform
        ordered_freq, ordered_ampl, ordered_imag, imaginary_ampl, unordered_freq = ordered_freq_apl(y_train)
        filtered_freq, filtered_imag = filter_freq_apl(ordered_freq, ordered_imag, f_thres = 1, ampl_thres = 0)
        interesting_freq, OLS_x_matrix, interesting_imag = retrieve_1yearfreq(keyword_data, filtered_imag, filtered_freq, y_train)
        OLS_x_matrix = sm.add_constant(OLS_x_matrix, prepend = True)
        OLS_x_matrix.columns = range(OLS_x_matrix.shape[1])
        
        stationary_order = diff_order(y_train)
        exog_regress = OLS_x_matrix
        
        length = len(interesting_freq)+5
        n_train = len(y_train)
        n_total = len(keyword_data)
        dummies = [0,1,2,3,4]
        criteria_AIC = 5
        extended_pred = 0
        t = np.arange(len(keyword_data)+ extended_pred)
        
        ### Sparse data test
        if check_less_than_100(keyword_data) == "More than 90% of interest < 100":
            keywords_nointerest_seasonal.append(key_word)
            y_SARIMA_train, y_SARIMA_pred, AIC_SARIMA, seas_order, normal_order, nr_par_SARIMA, sign_dummies_sarima = baseline_model(
                y_train, shift, n_train, n_total, stationary_order, exog_regress)

            fig4 = plt.figure(k, figsize=(11.69, 7))
            txt4 = str(key_word) + " (This dish has 90% of the interest < 100, hence only SARIMA is fitted on this dish) " + str(
                " Significant dummy: ") + str(sign_dummies_sarima) + str("\n N SARIMA: ") + str(
                nr_par_SARIMA) + " AIC SARIMA: " + str(round(AIC_SARIMA))
            plt.figure(k)
            plt.plot(np.arange(0, n_train), y_train, label="TRUE")
            plt.plot(np.arange(n_train, n_total), y_test, u'#1f77b4')
            plt.plot(np.arange(np.max(seas_order), n_train), y_SARIMA_train[np.max(seas_order):], label="SARIMA train")
            plt.plot(np.arange(n_total - np.max(seas_order), n_total), y_SARIMA_pred, label="SARIMA pred")
            plt.xlabel("Weeks")
            plt.ylabel("Interest")
            plt.text(0.05, 0.90, txt4, transform=fig4.transFigure, size=10)
            plt.legend()
            plt.close()
            pdf.savefig(fig4)
            k = k + 3
            j = j + 3

            continue
        
        # =============================================================================
        # 3.1 Compute FOURIER with AIC forward selection
        # =============================================================================

        dummies = [0,1,2,3,4]
        locs2, bestAIC, allAIC = best_N_selection_2(dummies, criteria_AIC, OLS_x_matrix, n_train, length, y_train)
        indexlow = allAIC.index(min(allAIC))
        locs = locs2[:indexlow+6]
        
        interesting_period = []
        for q in np.arange(5,len(locs)):
            interesting_period.append(round(interesting_freq[locs[q]-5]*52))
        cycles_per_period.append(interesting_period)
        
        y_est_train, y_est_pred, final_AIC, significance_dummies, locs_final = y_predict(y_train, OLS_x_matrix, n_train, n_total,locs)
        nr_par_FOURIER = len(locs_final)
               
        results = [key_word, locs_final, nr_par_FOURIER, round(final_AIC, 2)]
        resultaten_aic.append(results)
        
        # =============================================================================
        # 3.2 Compute FOURIER with Lasso
        # =============================================================================

        allAIC_lasso, minAIC_lasso = Lasso_AIC(OLS_x_matrix, y_train, key_word, n_train, 1)
        locs_lasso = minAIC_lasso['locs']
        dummies = [0,1,2,3,4]
        sign_dummies_lasso = []
        for o in range(len(dummies)):
            if dummies[o] in locs_lasso:
                sign_dummies_lasso.append(dummies[o])
        
        interesting_period_lasso = []
        for d in np.arange(0,len(locs_lasso)):
            if locs_lasso[d]-5 >= 0:
                interesting_period_lasso.append(round(interesting_freq[locs_lasso[d]-5]*52))
        
        cycles_per_period_lasso.append(interesting_period_lasso)
        
        nr_par_LASSO = len(locs_lasso)
        results2 = [key_word, locs_lasso, nr_par_LASSO, round(minAIC_lasso['aic_values'], 2)]
        resultaten_lasso.append(results2)

        y_train_lasso, y_pred_lasso = y_predict_L(y_train, OLS_x_matrix, n_train, n_total, locs_lasso)
        
        # =============================================================================
        # 3.3 Compute SARIMA 
        # =============================================================================
    
        y_SARIMA_train, y_SARIMA_pred, AIC_SARIMA, seas_order, normal_order, nr_par_SARIMA, sign_dummies_sarima  = baseline_model(y_train, shift, n_train, n_total, stationary_order, exog_regress)        

        # =============================================================================
        # 3.4 Compute FARIMA with Lasso and AIC forward selection
        # =============================================================================
    
        n_obs = len(y_train)
        train_fourier = OLS_x_matrix[0:n_train].loc[:,locs_final]
        test_fourier = OLS_x_matrix[n_train:n_total].loc[:,locs_final]
        train_fourier_lasso = OLS_x_matrix[0:n_train].loc[:,locs_lasso]
        test_fourier_lasso = OLS_x_matrix[n_train:n_total].loc[:,locs_lasso]
        
        y_residuals = y_train - y_est_train
        y_residuals_lasso = y_train - y_train_lasso
        
        resArima_model = auto_arima(y=y_residuals, seasonal=True, suppress_warnings = True, fit_args = True,stepwise=True,trace=True)
        resArima_model_lasso = auto_arima(y=y_residuals_lasso, seasonal=True, suppress_warnings = True, fit_args = True)

        order = resArima_model.order
        order_lasso = resArima_model_lasso.order

        seasonal_order = resArima_model.seasonal_order
        seasonal_order_lasso = resArima_model_lasso.seasonal_order

        FARIMA = SARIMAX(y_train.astype(float), exog = train_fourier, order = order, seasonal_order = seasonal_order, suppress_warnings = True,return_conf_int=True)
        FARIMA_fit = FARIMA.fit()

        FARIMA_lasso = SARIMAX(y_train.astype(float), exog = train_fourier_lasso, order = order_lasso, seasonal_order = seasonal_order_lasso, suppress_warnings = True,return_conf_int=True)
        FARIMA_fit_lasso = FARIMA_lasso.fit()

        nr_par_FARIMA = len(FARIMA_fit.params)
        nr_par_FARIMA_lasso = len(FARIMA_fit_lasso.params)
        
        y_FARIMA_train = FARIMA_fit.get_prediction(exog = train_fourier)
        y_FARIMA_train = y_FARIMA_train._predicted_mean
        y_FARIMA_pred = FARIMA_fit.get_forecast(steps = (n_total-n_train), exog = test_fourier)
        y_FARIMA_pred = y_FARIMA_pred._predicted_mean 
        
        y_FARIMA_train_lasso = FARIMA_fit_lasso.get_prediction(exog = train_fourier_lasso)
        y_FARIMA_train_lasso = y_FARIMA_train_lasso._predicted_mean
        y_FARIMA_pred_lasso = FARIMA_fit_lasso.get_forecast(steps = (n_total-n_train), exog = test_fourier_lasso)
        y_FARIMA_pred_lasso = y_FARIMA_pred_lasso._predicted_mean 

       # =============================================================================
       # 3.5 Compute SARIMAF with Lasso and AIC forward selection
       # =============================================================================
          
        max_order = max(max(order), max(normal_order), max(seas_order), max(seasonal_order))
        
        coef_SARIMA = nr_par_SARIMA 
        train_resid_SARIMA, res_SARIMAF_train, res_SARIMAF_pred, train_plus_test_resid_SARIMA, nr_par_SARIMAF, nr_par_SARIMAF_lasso, train_SARIMAF_lasso, pred_SARIMAF_lasso = fourier_residSARIMA(key_word, y_SARIMA_train, y_SARIMA_pred, y_train, y_test, max_order, coef_SARIMA, 0)
        y_SARIMAF_train = np.array(y_SARIMA_train[max_order:]) + res_SARIMAF_train
        y_SARIMAF_pred = y_SARIMA_pred + res_SARIMAF_pred
        y_SARIMAF_train_lasso = np.array(y_SARIMA_train[max_order:]) + train_SARIMAF_lasso
        y_SARIMAF_pred_lasso = y_SARIMA_pred + pred_SARIMAF_lasso

       # =============================================================================
       # 3.6 Compute in-sample adjusted AIC for each final model for corrected training set size, based on the maximum lag of SARIMAX model 
       # =============================================================================
       
        sigma2_FOURIER, logL_FOURIER, aic_fourier = logl_aic_fourier(n_obs, max_order, y_train, y_est_train, nr_par_FOURIER, len(significance_dummies))
        sigma2_FOURIER_lasso, logL_FOURIER_lasso, aic_fourier_lasso = logl_aic_fourier(n_obs, max_order, y_train, y_train_lasso, nr_par_LASSO, len(sign_dummies_lasso))        
        sigma2_SARIMA, logL_SARIMA, aic_sarima = logl_aic(n_obs, max_order, y_train, y_SARIMA_train, nr_par_SARIMA)
        sigma2_FARIMA, logL_FARIMA, aic_farima = logl_aic_fourier(n_obs, max_order, y_train, y_FARIMA_train, nr_par_FARIMA, len(significance_dummies) + nr_par_FARIMA - nr_par_FOURIER)
        sigma2_FARIMA_lasso, logL_FARIMA_lasso, aic_farima_lasso = logl_aic_fourier(n_obs, max_order, y_train, y_FARIMA_train_lasso, nr_par_FARIMA_lasso, len(sign_dummies_lasso) + nr_par_FARIMA_lasso - nr_par_LASSO)
        sigma2_SARIMAF, logL_SARIMAF, aic_sarimaf = logl_aic_fourier_SARIMAF(len(y_SARIMAF_train), 0, y_train, y_SARIMAF_train, nr_par_SARIMAF, coef_SARIMA, max_order, n_obs)
        sigma2_SARIMAF_lasso, logL_SARIMAF_lasso, aic_sarimaf_lasso = logl_aic_fourier_SARIMAF(len(y_SARIMAF_train_lasso), 0, y_train, y_SARIMAF_train_lasso, nr_par_SARIMAF_lasso, coef_SARIMA, max_order, n_obs)

        # =============================================================================
        # 3.7 Streak and peak predictions and evaluation score of all models
        # =============================================================================
        
        ## Adding timestamps to the predictions (include the dates)
        dates = make_dates("2018-01-01", n_total)
        predict_fourier = predictions(n_train, shift, y_est_pred, dates)
        predict_fourier_lasso = predictions(n_train, shift, y_pred_lasso, dates)
        predict_sarima = predictions(n_train, shift, y_SARIMA_pred, dates)
        predict_farima =  predictions(n_train, shift, y_FARIMA_pred, dates)
        predict_farima_lasso = predictions(n_train, shift, y_FARIMA_pred_lasso, dates)
        predict_sarimaf =  predictions(n_train, shift, y_SARIMAF_pred, dates)
        predict_sarimaf_lasso = predictions(n_train, shift, y_SARIMAF_pred_lasso, dates)
        ## Make a "predictions" dataframe of the real y
        true_y_dataframe = predictions(n_train, shift, y_test, dates)

        ## Compute streaks and peaks of given predictions
        evaluation_fourier = give_streaks2(predict_fourier, true_y_dataframe)
        evaluation_fourier_lasso = give_streaks2(predict_fourier_lasso, true_y_dataframe)
        evaluation_sarima = give_streaks2(predict_sarima, true_y_dataframe)
        evaluation_farima = give_streaks2(predict_farima, true_y_dataframe)
        evaluation_farima_lasso = give_streaks2(predict_farima_lasso, true_y_dataframe)
        evaluation_sarimaf = give_streaks2(predict_sarimaf, true_y_dataframe)
        evaluation_sarimaf_lasso = give_streaks2(predict_sarimaf_lasso, true_y_dataframe)
        ## Make a "evauation" of the real y
        evaluation_true = give_streaks2(true_y_dataframe, true_y_dataframe)

        acc_peaks_fourier, acc_streaks_fourier, avg_acc_fourier = give_prediction_score2(evaluation_fourier,evaluation_true)
        acc_peaks_fourier_lasso, acc_streaks_fourier_lasso, avg_acc_fourier_lasso = give_prediction_score2(evaluation_fourier_lasso, evaluation_true)
        acc_peaks_sarima, acc_streaks_sarima, avg_acc_sarima = give_prediction_score2(evaluation_sarima,evaluation_true)
        acc_peaks_farima, acc_streaks_farima, avg_acc_farima = give_prediction_score2(evaluation_farima,evaluation_true)
        acc_peaks_farima_lasso, acc_streaks_farima_lasso, avg_acc_farima_lasso = give_prediction_score2(evaluation_farima_lasso,evaluation_true)
        acc_peaks_sarimaf, acc_streaks_sarimaf, avg_acc_sarimaf = give_prediction_score2(evaluation_sarimaf,evaluation_true)
        acc_peaks_sarimaf_lasso, acc_streaks_sarimaf_lasso, avg_acc_sarimaf_lasso = give_prediction_score2(evaluation_sarimaf_lasso,evaluation_true)
        
        results3 = [key_word, list(y_est_pred),
        list(y_pred_lasso),
        list(y_FARIMA_pred),
        list(y_FARIMA_pred_lasso),
        list( y_SARIMA_pred),
        list(y_SARIMAF_pred),
        list(y_SARIMAF_pred_lasso)]
        total_predictions.append(results3)
        
        
        accuracy_scores.loc[i, ['keyword',
                            "FOURIER",
                            "FOURIER lasso",
                            "FARIMA",
                            "FARIMA lasso",
                            "SARIMA",
                            "SARIMAF",
                            "SARIMAF lasso"]] = [key_word,
                                                round(avg_acc_fourier), 
                                                round(avg_acc_fourier_lasso), 
                                                round(avg_acc_farima),
                                                round(avg_acc_farima_lasso),
                                                round(avg_acc_sarima),
                                                round(avg_acc_sarimaf),
                                                round(avg_acc_sarimaf_lasso)]
                                                
        
        AIC_scores.loc[i, ['keyword',
                           "FOURIER",
                           "FOURIER lasso",
                           "FARIMA",
                           "FARIMA lasso",
                           "SARIMA",
                           "SARIMAF",
                           "SARIMAF lasso"]]= [key_word,
                                               round(aic_fourier),
                                               round(aic_fourier_lasso),
                                               round(aic_farima),
                                               round(aic_farima_lasso),
                                               round(aic_sarima),
                                               round(aic_sarimaf),
                                               round(aic_sarimaf_lasso)]
        
        N_included.loc[i, ['keyword',
                           "FOURIER",
                           "FOURIER lasso",
                           "FARIMA",
                           "FARIMA lasso",
                           "SARIMA",
                           "SARIMAF",
                           "SARIMAF lasso"]] = [key_word,
                                               nr_par_FOURIER,
                                               nr_par_LASSO,
                                               nr_par_FARIMA,
                                               nr_par_FARIMA_lasso,
                                               nr_par_SARIMA,
                                               nr_par_SARIMAF,
                                               nr_par_SARIMAF_lasso]
                                                
        streaks_and_peaks.loc[i, ['keyword',
                                   "FOURIER",
                                   "FOURIER lasso",
                                   "FARIMA",
                                   "FARIMA lasso",
                                   "SARIMA",
                                   "SARIMAF",
                                   "SARIMAF lasso"]] = [key_word,
                                                       evaluation_fourier,
                                                       evaluation_fourier_lasso,
                                                       evaluation_farima,
                                                       evaluation_farima_lasso,
                                                       evaluation_sarima,
                                                       evaluation_sarimaf,
                                                       evaluation_sarimaf_lasso]

        # =============================================================================
        # 3.8 Create output figures                                              
        # =============================================================================

        ### Figure to plot true data against a certain model predictions
        plt.figure(10)
        plt.figure(dpi=1200) 
        plt.plot(dates[0:n_train], y_train[0:n_train], label="TRUE train")
        plt.plot(dates[n_train:n_total], y_test, u'#1f77b4', label = "TRUE test")
        plt.plot(dates[max_order:n_train], y_train_lasso[max_order:], label="FOURIER (Lasso) train")
        plt.plot(dates[n_train:n_total], y_pred_lasso, label="FOURIER (Lasso) predict")
        plt.xticks(rotation=70) 
        plt.xlabel("Weeks")
        plt.ylabel("Interest")
        plt.legend()
        
        ### Create figures for in the pdf file                                               

        fig = plt.figure(k, figsize=(11.69, 7))
        txt1 = str(key_word) + str(" Significant dummy: ") + str(sign_dummies_sarima) + str("\n N SARIMA: ") + str(
            nr_par_SARIMA) + ", AIC SARIMA: " + str(round(aic_sarima)) + str("\n N SARIMAF (AIC): ") + str(
            nr_par_SARIMAF) + ", AIC SARIMAF (AIC): " + str(round(aic_sarimaf))
        plt.figure(k)
        plt.plot(dates[0:n_train], y_train, label="TRUE")
        plt.plot(dates[n_train:n_total], y_test, u'#1f77b4')
        plt.plot(dates[max_order:n_train], y_SARIMA_train[max_order:], label="SARIMA train")
        plt.plot(dates[n_train:n_total], y_SARIMA_pred, label="SARIMA pred")
        plt.plot(dates[max_order:n_train], y_SARIMAF_train, label="SARIMAF (AIC) train")
        plt.plot(dates[n_train:n_total], y_SARIMAF_pred, label="SARIMAF (AIC) pred")
        plt.xticks(rotation=70) 
        plt.xlabel("Weeks")
        plt.ylabel("Interest")
        plt.text(0.05, 0.90, txt1, transform=fig.transFigure, size=10)
        plt.legend()
        plt.close()
        pdf.savefig(fig)
        k = k + 3

        fig2 = plt.figure(j, figsize=(11.69, 7))
        txt2 = str(key_word) + str(" Significant dummy:") + str(significance_dummies) + str("\n N FOURIER: ") + str(
            nr_par_FOURIER) + ", AIC FOURIER: " + str(round(aic_fourier)) + str("\n N FARIMA (AIC): ") + str(
            nr_par_FARIMA) + ", AIC FARIMA (AIC): " + str(round(aic_farima))
        plt.figure(j)
        plt.plot(dates[0:n_train], y_train, label="TRUE")
        plt.plot(dates[n_train:n_total], y_test, u'#1f77b4')
        plt.plot(dates[max_order:n_train], y_FARIMA_train[max_order:], label="FARIMA (AIC) train")
        plt.plot(dates[n_train:n_total], y_FARIMA_pred, label="FARIMA (AIC) pred")
        plt.plot(dates[max_order:n_train], y_est_train[max_order:], label="FOURIER train")
        plt.plot(dates[n_train:n_total], y_est_pred, label="FOURIER pred")
        plt.text(0.05, 0.90, txt2, transform=fig2.transFigure, size=10)
        plt.xticks(rotation=70) 
        plt.xlabel("Weeks")
        plt.ylabel("Interest")
        plt.legend()
        plt.close()
        pdf.savefig(fig2)
        j = j + 3
        
        fig3 = plt.figure(p, figsize=(11.69, 7))
        txt3 = str(key_word)  + str("\n N FOURIER (AIC): ") + str(
            nr_par_FOURIER) + ", AIC FOURIER (AIC): " + str(round(aic_fourier)) + str("\n N FOURIER (Lasso): ") + str(
            nr_par_LASSO) + ", AIC FOURIER (Lasso): " + str(round(aic_fourier_lasso))
        plt.figure(p)
        plt.plot(np.arange(0, n_train), y_train, label="TRUE")
        plt.plot(np.arange(n_train, n_total), y_test, u'#1f77b4')
        plt.plot(np.arange(max_order, n_train), y_est_train, label="Fourier (AIC) train")
        plt.plot(np.arange(n_total - max_order, n_total), y_est_pred, label="Fourier (AIC) pred")
        plt.plot(np.arange(max_order, n_train), y_train_lasso, label="Fourier (Lasso) train")
        plt.plot(np.arange(n_train, n_total), y_pred_lasso, label="Fourier (Lasso) pred")
        plt.xticks(rotation=70) 
        plt.xlabel("Weeks")
        plt.ylabel("Interest")
        plt.legend()
        plt.close()
        pdf.savefig(fig3)
        p = p + 3   
     
pdf.close()

# =============================================================================
# 4. Code to convert all output from the analysis to CSV files (run after the loop is finished)
# =============================================================================
 
pd.DataFrame(resultaten_lasso, columns = ['key word', 'locs', 'length locs', 'min AIC value']).to_csv('compare_lasso_all.csv', sep=';')
pd.DataFrame(resultaten_aic, columns = ['key word', 'locs', 'length locs', 'min AIC value']).to_csv('compare_aic_all.csv', sep=';')
pd.DataFrame(total_predictions, columns = ['key_word', 'y_est_pred', 'y_pred_lasso', 'y_FARIMA_pred', 'y_FARIMA_pred_lasso', 'y_SARIMA_pred', 'y_SARIMAF_pred', 'y_SARIMAF_pred_lasso']).to_csv('compare_pred_all.csv', sep=';')

pd.DataFrame(AIC_scores).to_csv('aic_all_keys.csv', sep=';')
pd.DataFrame(accuracy_scores).to_csv('accuracy_all_keys.csv', sep=';')
pd.DataFrame(N_included).to_csv('N_all_keys.csv', sep=';')
pd.DataFrame(streaks_and_peaks).to_csv('Streaks&Peaks_part_all.csv', sep=';')
