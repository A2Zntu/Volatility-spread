# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:14:50 2018

@author: user_2
"""

S = 0.0
K = 1600.0
t = 25
def MK_disc(S, K, t): # Maturity and strike discount 
    try:
        m = float(K/S - 1) # moneyness
        M = max(1, t/30.0) #days to month; at least one-month
        w = exp(-(m**2)/2 - (M - 1)**2)
    except ZeroDivisionError:
        M = max(1, t/30.0)
        w = exp(- (M - 1)**2)
    return w
w = MK_disc(S, K, t)