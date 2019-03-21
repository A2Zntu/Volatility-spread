# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 22:08:45 2018

@author: Evan
"""

import pandas as pd
import numpy as np
from datetime import date
from math import exp, sqrt, log, fabs
import matplotlib.pyplot as plt
from itertools import repeat
from scipy.stats import shapiro, norm, kurtosis, skew
from statsmodels.graphics.gofplots import qqplot
import statsmodels.formula.api as sm
from statsmodels.tsa.stattools import adfuller
import seaborn as sns; sns.set()
import os
from tqdm import tqdm

#%%
work_dir = os.getcwd()
work_dir = os.path.join(work_dir, 'Documents', 'GitHub', 'Volatility-spread')
Path_default_readcsv = os.path.join(work_dir,'Read_csv')
#%%
time_period =  pd.read_csv(Path_default_readcsv + "\period_Quote.csv", header = None)
time_period.columns = ['Head', 'Tail', 'Middle']

x_axis_hour = [] #generate x axis
for s in time_period['Middle']:
    if s < 100000: #Avoid the disorder in ploting the x-axis
        if not int((s % 10000)/100) == 0:
            x_axis_hour.append('0' + str(int(s/10000)) +':' + str(int((s % 10000)/100)))
        else:
            x_axis_hour.append('0' + str(int(s/10000)) +':00')
    else:
        if not int((s % 10000)/100) == 0:
            x_axis_hour.append(str(int(s/10000)) +':' + str(int((s % 10000)/100)))
        else:
            x_axis_hour.append(str(int(s/10000)) +':00')
x_axis_hour = np.array(x_axis_hour)

list_year_and_month = []
list_year = []
start_year = 2007
end_year = 2017
start_month = 1
end_month = 12

for i in range(start_year, end_year + 1):
    list_year.append(str(i))
    for j in range(start_month, end_month + 1):
        if j < 10:
            list_year_and_month.append(str(i) + '0' + str(j))
        else:
            list_year_and_month.append(str(i) +str(j))

#%% large sample size convert t-test to Z-test 
def twosampleZ(data1, data2):
    '''large sample size convert t-test to Z-test '''
    X1_bar = np.nanmean(data1)
    X2_bar = np.nanmean(data2)
    std1 = np.nanstd(data1)
    std2 = np.nanstd(data2)
    n1 = len(data1) - sum(np.isnan(data1)) #real value
    n2 = len(data1) - sum(np.isnan(data2))
    pooledSE = sqrt(std1**2/n1 + std2**2/n2)
    z = ((X1_bar - X2_bar))/pooledSE
    pval = 2*(1 - norm.cdf(abs(z))) #two-sided z test
    
    return round(z, 4), round(pval, 4)    

#%% plot            
def plot_des_info(df_year_vs, yearname):
    plt.style.use('ggplot')
    plt.figure(figsize=(15,6))
    plt.plot(x_axis_hour, df_year_vs)
    plt.title(yearname)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Spread Volatility', fontsize=14)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


#%% turn results into dataframe

def results_summary_to_df(results):
    '''take the result of an statsmodel results table and transforms it into a dataframe'''
    pvals = results.pvalues
    coeff = results.params
    tvals = results.tvalues
    rsquare = results.rsquared


    results_df = pd.DataFrame({"pvals":pvals,
                               "coeff":coeff,
                               "tvals":tvals,
                               "R^2":rsquare
                                })

    #Reordering...
    results_df = results_df[["coeff","tvals","pvals", "R^2"]]
    return results_df
    
#%% test the population mean of IVS between 2007 to 2017 (11 X 14)
Path_output_csv = os.path.join(work_dir, 'Output_Result')    
df_IVS = pd.read_csv(os.path.join(Path_output_csv, 'df_overall_vs.csv'), index_col = 0)
# Assume all the distributions are not normal. 
list_stat = []
list_p = []
list_corrcoef = []
for i, hour in enumerate(x_axis_hour):
    list_stat.append([])
    list_p.append([])
    list_corrcoef.append([])
    for other_hour in x_axis_hour: 
        stat, p = twosampleZ(df_IVS[hour], df_IVS[other_hour])
        list_stat[i].append(stat)
        list_p[i].append(p)
        list_corrcoef[i].append(np.corrcoef(df_IVS[hour], df_IVS[other_hour])[0][1])

df_p = pd.DataFrame(list_p, columns = x_axis_hour)
df_p.index = x_axis_hour
df_stat = pd.DataFrame(list_stat, columns = x_axis_hour)
df_stat.index = x_axis_hour
df_corrcoef = pd.DataFrame(list_corrcoef, columns = x_axis_hour)
df_corrcoef.index = x_axis_hour

#%% discriptive statistic
# find how many nan value in each year
whole_days = list(df_IVS.index)

start_year = 2007
yit = [0,]
for i, day in enumerate(whole_days):
    if str(day)[:4] == str(start_year):
        pass
    else: 
        start_year = start_year + 1
        yit.append(i) 
        
yit.append(len(whole_days)) #include the last digit

# total length = 15; first = 1
df_aggre_nan = pd.DataFrame()
for i in range(1, 12):
    init = yit[i-1]
    end = yit[i]
    df_year_vs = df_IVS[init:end]
    yearname = str(df_year_vs.index[0])[:4]
    df_mean = df_year_vs.mean()
    df_std = df_year_vs.std()
    df_nan = df_year_vs.isna().sum()
    df_aggre_nan = df_aggre_nan.append(pd.DataFrame(df_nan).T)
df_aggre_nan.index = list_year

#%% read Huge IVS data with timediff, Moneyness, Maturity, Dummy 
# Quote Data Num: 1692542
def clip_the_info_data(chunk_num = 0, chunksize = 500000, Info_path = os.path.join(work_dir, 'Output_Result',  'df_aggre_vs_info.csv')):
    '''Since the Info VS is too big, we cut it in small pieces'''
    traintypes = {
    'IVS': 'float', 
    'timediff':'float',
    'Moneyness':'float', 
    'Maturity':'float', 
    'Dummy':'float'}
    
    cols = list(traintypes.keys())
    
    df_list = [] 
    for df_chunk in tqdm(pd.read_csv(Info_path, usecols = cols, dtype = traintypes, chunksize=chunksize)):
        df_list.append(df_chunk) 
        
    return df_list[chunk_num]

#%% timediff, Moneyness, Maturity, Dummy 
def print_correlation(df_info, item1, item2, item1abs = True, item2abs = True):
    if type(item1) == str and type(item2) == str:
        if item1abs == True:
            df_info[item1] = abs(df_info[item1])
        if item2abs == True:
            df_info[item2] = abs(df_info[item2])
            
        info_corr = np.corrcoef(df_info[item1], df_info[item2])
        print('Correlation of %s and %s: '%(item1, item2))
        print(info_corr)
        print("==================================")
    else:
        print("Input Type is Wrong!!")

df_info = clip_the_info_data(chunk_num =3)  

#%% print correlation and do regression 
# The t-statistics are Newy-west adjusted. 
print_correlation(df_info, 'IVS', 'Maturity')

df_info['IVS'] = abs(df_info['IVS'])
df_info['Moneyness'] = abs(df_info['Moneyness'])

results = sm.ols(formula = 'IVS ~ timediff + Moneyness + Maturity',
                 data = df_info, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags':1})
results_summary = results.summary()
print(results.summary()) 
df_res = results_summary_to_df(results)


#%% read data
Path_SP_data = os.path.join(Path_default_readcsv, 'SPX_PUTCALL.csv')
df_SP = pd.read_csv(Path_SP_data, index_col = 0)
list_SPXprice = [price for price in df_SP['SPX Price']]
list_SPXreturn = [np.nan,]
# Return = log(Pt+1/Pt)
for i in range(1, len(list_SPXprice)):
    list_SPXreturn.append(np.log(list_SPXprice[i]/list_SPXprice[i-1]))
    
# Paste the SPX information to df_IVS
df_IVS['SPX price'] = df_IVS.index.map(df_SP['SPX Price'])
df_SP['SPX_return'] = list_SPXreturn
df_IVS['SPX return'] =  df_IVS.index.map(df_SP['SPX_return'])
df_IVS['lag return'] = df_IVS['SPX return'].shift(1)

#%%
Path_DEF_data = os.path.join(work_dir, 'BondYield', 'DDEF.csv')
Path_TERM_data = os.path.join(work_dir, 'BondYield', 'DTERM.csv')
df_DEF = pd.read_csv(Path_DEF_data, index_col = 0)
df_TERM = pd.read_csv(Path_TERM_data, index_col = 0)

df_IVS['DDEF'] =  df_IVS.index.map(df_DEF['DDEF'])
df_IVS['DTERM'] =  df_IVS.index.map(df_TERM['DTERM(10year-1month)'])

df_copy = df_IVS.copy()
columns_name = ['t0830', 't0900', 't0930', 't1000', 
                't1030', 't1100', 't1130', 't1200', 
                't1230', 't1300', 't1330', 't1400', 
                't1430', 't1500', 'SPX_price', 'SPX_return',
                'lag_return', 'DDEF', 'DTERM']

df_copy.columns = columns_name
df_copy.to_csv(os.path.join(Path_output_csv, 'df_IVS_SPX.csv'))

#%% regression on return
def test_SPX_return(time_input = 't0800', read_path = os.path.join(Path_output_csv, 'df_IVS_SPX.csv')):
    '''Input the time period and check the test statisitics
        e.g. time_input = 't1200', it means we'd like to test 12:00; 
        read_path = the path of csv with SPX_return and DDEF, DTERM appended
    '''
    
    df_change = pd.read_csv(read_path, index_col = 0)
    results = sm.ols(formula = 'SPX_return ~ {} + DDEF + DTERM + lag_return'.format(time_input), data =  df_change, missing='drop').fit(cov_type='HAC',cov_kwds={'maxlags':1})
    print(results.summary())
    df_res = results_summary_to_df(results)
    return df_res

#%%
df_res = test_SPX_return(time_input='t0800')