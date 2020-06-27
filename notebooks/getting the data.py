# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:07:25 2020

@author: chigr
"""

import pandas as pd
import numpy as np
import datetime as dt

save=True
path = r'D:\Serega\Education\WQU\Capstone'
token = '4c4b8dc5a8dde465f3a22f79c92b3b603ad379fb'

try:
    import fxcmpy

    con = fxcmpy.fxcmpy(access_token=token, log_level='error')
    
    available_tickers = con.get_instruments()
    
    
    tickers = ['EUR/USD', 'USD/JPY', 'GBP/USD', 'AUD/USD','AUD/JPY',\
               'NZD/USD', 'USD/CAD', 'AUD/CAD', 'USD/CHF', 'USD/NOK', 'USD/SEK']
        
    na_tickers = set(tickers) - set(available_tickers)
    print(f'The following tickers are not available: {na_tickers}')
    
    yr = 6
    time = dt.datetime.utcnow()
    dates = {dt.datetime(year =time.year-i, month=time.month, day=time.day):dt.datetime(year =time.year-i+1, month=time.month, day=time.day)  for i in range(yr,0,-1)}
    
    for ticker in tickers:
        pair = pd.concat([con.get_candles(ticker, period='H2', start=k, end=v) for k,v in dates.items()])
        pair = pair.loc[~pair.index.duplicated(keep='first')]       
        pair.to_csv(path + '//input//' +  ''.join(ticker.split('/')) + '.csv')

    # minutes: m1, m5, m15 and m30,
    # hours: H1, H2, H3, H4, H6 and H8,
    # one day: D1,
    # one week: W1,
    # one month: M1.
except:
    print("Reading the data offline")
    

    
    