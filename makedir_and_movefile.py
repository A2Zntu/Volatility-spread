# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 14:42:11 2018

@author: user_2
"""

import os, shutil, glob

path = 'D:/SPX Tick Data/'


list_year_and_month = []
list_year = []
start_year = 2010
end_year = 2018
start_month = 1
end_month = 12


for i in range(start_year, end_year + 1):
    list_year.append(str(i))
    for j in range(start_month, end_month + 1):
        if j < 10:
            list_year_and_month.append(str(i) + '0' + str(j))
        else:
            list_year_and_month.append(str(i) +str(j))


for i, year in enumerate(range(start_year,end_year)):
    for j in range(i*12, (i+1)*12):
        if not os.path.exists(path + list_year[i] + '/' + list_year_and_month[j]):
            os.makedirs(path + list_year[i] + '/' + list_year_and_month[j])


list_number = []            
for i in range(1):
    list_hi = []
    for j in range(i*12,(i+1)*12):
        list_hi.append(j)
    list_number.append(list_hi)


    
#%%
def right_dir(year, res):
    loc = list_year.index(year)
    for j in range(loc*12, (loc+1)*12):
        a = res.find(list_year_and_month[j])
        if a != -1:
            break
    return list_year_and_month[j]
    


extension = 'gz'

for i, year in enumerate(range(start_year, end_year)):
    os.chdir(path + str(year))
    result = [i for i in glob.glob('*.{}'.format(extension))]
    for j in range(len(result)):
        subpath = right_dir(list_year[i], result[j])
        print(subpath)
        shutil.move(path + list_year[i] + '/' + result[j], path + list_year[i] + '/' +str(subpath) + '/' + result[j])


