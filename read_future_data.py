#%%
import pandas as pd
import numpy as np
import os
import datetime
year = 2019

trading_list = pd.read_csv('E:/future/fut_map_2018_2022.csv')

future = trading_list[trading_list['ts_code']=='AU.SHF']
future_year = future[future['year']==year]

future_year['contract'] = future_year['mapping_ts_code'].str.extract('(.{0,6})')
future_year['contract'] = future_year['contract'].str.lower()
future_year['trade_date'] = future_year['trade_date'].astype(str)

date = []
g = os.walk('F:\\zhongtai_%s'%(year))

for path, dir_list, file_list in g:
    for dir_name in dir_list:
        date.append(dir_name)

        # print(dir_name)

all_data = pd.DataFrame()
for i in date:
    for j in future_year['trade_date']:
        if i == j:
            contract = future_year[future_year['trade_date'] == i]['contract'].values[0]
            dir_file = 'F:\\zhongtai_%s\\%s\\%s.csv'%(year, i, contract)
            # print(dir_file)
            file = pd.read_csv(dir_file)
            cols = ['datetime', 'datetime_nano', 'last_price', 'highest', 'lowest',
       'volume', 'amount', 'open_interest', 'bid_price1', 'bid_volume1',
       'ask_price1', 'ask_volume1', 'bid_price2', 'bid_volume2', 'ask_price2',
       'ask_volume2', 'bid_price3', 'bid_volume3', 'ask_price3', 'ask_volume3',
       'bid_price4', 'bid_volume4', 'ask_price4', 'ask_volume4', 'bid_price5',
       'bid_volume5', 'ask_price5', 'ask_volume5' ]
            file.columns = cols
            all_data = all_data.append(file)

all_data.to_csv('F:\\zhongtai_au\\au_%s.csv'%(year))
