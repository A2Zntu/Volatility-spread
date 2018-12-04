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
from scipy.stats import shapiro, norm
from statsmodels.graphics.gofplots import qqplot
import statsmodels.formula.api as sm
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
start_year = 2004
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
    
#%%
df_IVS = pd.read_csv("E:/Spyder/IVS/vs_10days_withinfo.csv", index_col = 0)
# Assume all the distributions are not normal. 
list_stat = []
list_p = []
for i, hour in enumerate(x_axis_hour):
    list_stat.append([])
    list_p.append([])
    for other_hour in x_axis_hour: 
        stat, p = twosampleZ(df_IVS[hour], df_IVS[other_hour])
        list_stat[i].append(stat)
        list_p[i].append(p)

df_p = pd.DataFrame(list_p, columns = x_axis_hour)
df_p.index = x_axis_hour
df_stat = pd.DataFrame(list_stat, columns = x_axis_hour)
df_stat.index = x_axis_hour


#%% discriptive statistic

whole_days = list(df_IVS.index)

start_year = 2004
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
for i in range(1, 15):
    init = yit[i-1]
    end = yit[i]
    df_year_vs = df_IVS[init:end]
    yearname = str(df_year_vs.index[0])[:4]
    df_mean = df_year_vs.mean()
    df_std = df_year_vs.std()
    df_nan = df_year_vs.isna().sum()
    df_aggre_nan = df_aggre_nan.append(pd.DataFrame(df_nan).T)
#    plot_des_info(df_nan, yearname)
df_aggre_nan.index = list_year

#%% regression 
    
df_SP = pd.read_csv("E:/Spyder/BMG/SPX_PUTCALL.csv")
df_info = pd.read_csv("E:/Spyder/IVS/aggre_vs_info.csv", skiprows = 0, index_col = 0)

#results_as_html = results_summary.tables[1].as_html()
#pd.read_html(results_as_html, header=0, index_col=0)
    

#%%

def results_summary_to_dataframe(results):
    '''take the result of an statsmodel results table and transforms it into a dataframe'''
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    results_df = pd.DataFrame({"pvals":pvals,
                               "coeff":coeff,
                               "conf_lower":conf_lower,
                               "conf_higher":conf_higher
                                })

    #Reordering...
    results_df = results_df[["coeff","pvals","conf_lower","conf_higher"]]
    return results_df
#%%
df_info = pd.read_csv("E:/Spyder/IVS/aggre_vs_info.csv", skiprows = 0, index_col = 0)
df_info['Moneyness'] = df_info['Moneyness']
df_info['TimeDiff'] = df_info['TimeDiff']/300
df_info['IVS'] = abs(df_info['IVS'])

results = sm.ols(formula = 'IVS ~ TimeDiff ', data =  df_info).fit()
results_summary = results.summary()
print(results.summary())

#%%
plt.figure(figsize=(15,6))
meanIVS = np.average(df_info['IVS'])
stdIVS = np.std(df_info['IVS'])
upp = meanIVS + 2*stdIVS
low = meanIVS - 2*stdIVS
plt.hist(x = df_info['IVS'], bins = 1000, range = [low, upp],  color='#0504aa')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('IVS')
#maxfreq = n.max()
## Set a clean upper y-axis limit.
#plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)




