"""
This is a pipeline 'data_pull'
generated using Kedro 0.19.3

It is used to pull data from the Yahoo Finance API 

NOTE: This is often depricated and may need to be adjusted when there is a change to the underlying API but not the specific python package
"""

import pandas as pd
import yfinance as yf
import datetime as dt
import time as tm
import logging
import random
import os
from utils import fix_columns

logger = logging.getLogger(__name__)


# functions: 

def pull_stock_data(data_pull_parameters: dict) -> pd.DataFrame:
    
    start = tm.time()
    
    if data_pull_parameters['single_dataframe'] == True:
        pulls = 0

        for stock in data_pull_parameters['equities']:
            print(f'retrieving: {stock.strip()}')

            if pulls == 0:
                df = yf.download(stock.strip(), start = data_pull_parameters['start_date'], end = data_pull_parameters['end_date'])
                pulls+=1

                df = df.reset_index()
                df['ticker'] = stock
                
                sleep_time = random.randint(2, 10)
                print(f"sleeping for: {sleep_time} seconds")
                # sleep between pulls so to not arouse suspicion:
                tm.sleep(sleep_time)

            else:
                temp = yf.download(stock.strip(), start = data_pull_parameters['start_date'], end = data_pull_parameters['end_date'])
                pulls+=1

                temp = temp.reset_index()
                temp['ticker'] = stock
    
                # union-all into the main dataframe:
                df = pd.concat([df, temp], ignore_index= True)

                del temp
                sleep_time = random.randint(2, 10)

                print(f"sleeping for: {sleep_time} seconds")
                # sleep between pulls so to not arouse suspicion:
                tm.sleep(sleep_time)

        del pulls, sleep_time

        df.columns = fix_columns(df.columns)
        return df

    else:

        for stock in data_pull_parameters['equities']:

            df = yf.download(stock.strip(), start = data_pull_parameters['start_date'], end = data_pull_parameters['end_date'])
            df['ticker'] = stock
            df = df.reset_index()
            df.to_csv(os.path.join('../data/01_raw/separate_stock_pulls/', stock.strip() +'.csv'), index = False)
            sleep_time = random.randint(2, 10)
            tm.sleep(sleep_time)
            print(f"sleeping for: {sleep_time} seconds")
            print('saving: ', stock, ' data to: ', os.path.join('../data/01_raw/separate_stock_pulls/', stock.strip() +'.csv'))
        
        return df # only returns the last df to the catalogue ***


