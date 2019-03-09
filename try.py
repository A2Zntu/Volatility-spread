# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 10:38:50 2019

@author: Evan
"""
import mysql
from mysql.connector import *
from mysql.connector import errorcode


config = {
  'host':'127.0.0.1',
  'user':'root',
  'password':'Ntu830531'
}

# Construct connection string
try:
   conn = mysql.connector.connect(**config)
   print("Connection established")
except mysql.connector.Error as err:
  if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
    print("Something is wrong with the user name or password")
  elif err.errno == errorcode.ER_BAD_DB_ERROR:
    print("Database does not exist")
  else:
    print(err)
else:
  cursor = conn.cursor()