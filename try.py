# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 10:55:52 2018

@author: user_2
"""


def dummy_hour(the_time):
    dummy = 0
    for i in range(len(time_period)):
        if the_time >= time_period['Head'][i] and the_time <= time_period['Tail'][i]:
            dummy = i
    return dummy

the_time = df['TRADE_TIME'][65]
dummy = dummy_hour(the_time)
print(dummy)