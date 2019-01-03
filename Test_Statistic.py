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
#%%
time_period =  pd.read_csv("E:/Spyder/period_trade_1.csv", header = None)
time_period.columns = ['Head', 'Tail', 'Fake_middle', 'Middle']

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
df_IVS = pd.read_csv("E:/Spyder/IVS/vs_10days_mkdweighted_2007to2017.csv", index_col = 0)
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

#%% read 192391 IVS data with TimeDiff, Moneyness, Maturity, Dummy 

df_info = pd.read_csv("E:/Spyder/IVS/info_vs_aggre_2007to2017.csv", skiprows = 0, index_col = 0)


df_info['Moneyness'] = abs(df_info['Moneyness'])
df_info['TimeDiff'] = df_info['TimeDiff']/300
df_info['IVS'] = abs(df_info['IVS'])
df_info['SpriceDiff'] = abs(df_info['SpriceDiff'])
#results = adfuller(df_info['IVS'], autolag='AIC')
#print('ADF Statistic: %f' % results[0])
#print('p-value: %f' % results[1])
print("Corr:")
info_corr = np.corrcoef(df_info['IVS'], df_info['SpriceDiff'])
print(info_corr)
print("Corr:")
info_corr = np.corrcoef(df_info['TimeDiff'], df_info['SpriceDiff'])
print(info_corr)
print("Corr:")
info_corr = np.corrcoef(df_info['TimeDiff'], df_info['IVS'])
print(info_corr)
#print("kurtosis:%f"%kurtosis(df_info['IVS']))
#print("skewness:%f"%skew(df_info['IVS']))
#autocorr = df_info['IVS'].autocorr(lag=1)
#print(autocorr)

#%% regression on IVS
# The t-statistics are Newy-west adjusted. 

results = sm.ols(formula = 'IVS ~ TimeDiff + Moneyness+ Maturity + C(Dummy) + SpriceDiff ', data =  df_info).fit(cov_type='HAC',cov_kwds={'maxlags':1})
results_summary = results.summary()
print(results.summary())
df_res = results_summary_to_df(results)

#%% plot the IVS
#plt.figure(figsize=(15,6))
#meanIVS = np.average(df_info['IVS'])
#stdIVS = np.std(df_info['IVS'])
#upp = meanIVS + 2*stdIVS
#low = meanIVS - 2*stdIVS
#plt.hist(x = df_info['IVS'], bins = 1000, range = [low, upp],  color='#0504aa')
#plt.grid(axis='y', alpha=0.75)
#plt.xlabel('Value')
#plt.ylabel('Frequency')
#plt.title('IVS')

#%% read data
df_SP = pd.read_csv("E:/Spyder/BMG/SPX_PUTCALL.csv", index_col = 0)
list_SPXprice = [price for price in df_SP['SPX Price']]
list_SPXreturn = [np.nan,]
# Return = log(Pt+1/Pt)
for i in range(1, len(list_SPXprice)):
    list_SPXreturn.append(np.log(list_SPXprice[i]/list_SPXprice[i-1]))

df_IVS['SPX price'] = df_IVS.index.map(df_SP['SPX Price'])
df_SP['SPX_return'] = list_SPXreturn
df_IVS['SPX return'] =  df_IVS.index.map(df_SP['SPX_return'])
df_IVS['lag return'] = df_IVS['SPX return'].shift(1)

#%%
df_DEF = pd.read_csv("E:/Spyder/BondYield/DDEF.csv", index_col = 0)
df_TERM = pd.read_csv("E:/Spyder/BondYield/DTERM.csv", index_col = 0)
df_IVS['DDEF'] =  df_IVS.index.map(df_DEF['DDEF'])
df_IVS['DTERM'] =  df_IVS.index.map(df_TERM['DTERM(10year-1month)'])

df_copy = df_IVS.copy()
columns_name = ['t0830', 't0900', 't0930', 't1000', 
                't1030', 't1100', 't1130', 't1200', 
                't1230', 't1300', 't1330', 't1400', 
                't1430', 't1500', 'SPX_price', 'SPX_return',
                'DDEF', 'DTERM', 'lag_return']
df_copy.columns = columns_name
df_copy.to_csv("E:/Spyder/IVS/copy.csv")

#%% regression on return
df_change = pd.read_csv("E:/Spyder/IVS/copy.csv", index_col = 0)
results = sm.ols(formula = 'SPX_return ~ t1500 + DDEF + DTERM + lag_return', data =  df_copy, missing='drop').fit(cov_type='HAC',cov_kwds={'maxlags':1})
results_summary = results.summary()
print(results.summary())
df_res = results_summary_to_df(results)

