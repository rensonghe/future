import pandas as pd
import numpy as np
import itertools
import tqdm
from tqdm import tqdm
import pyarrow
from pyarrow import fs
import pyarrow as pa
import pyarrow.parquet as pq
#%%
minio = fs.S3FileSystem(endpoint_override="192.168.34.57:9000", access_key="zVGhI7gEzJtcY5ph",
                        secret_key="9n8VeSiudgnvzoGXxDoLTA6Y39Yg2mQx", scheme="http")
schema = pa.schema([
    ('open', pa.float64()),
    ('high', pa.float64()),
    ('low', pa.float64()),
    ('close', pa.float64()),
    ('open_interest', pa.float64()),
    ('closse_interest', pa.float64()),
    ('open_volume', pa.float64()),
    ('closse_volume', pa.float64()),
    ('open_timestamp', pa.int64()),
    ('close_timestamp', pa.int64()),
    ('order_book_id', pa.string()),
    ('year', pa.int64()),
    ('month', pa.int64())
])
#%%
main_future_list = ['A888', 'AG888', 'AL888', 'AP888', 'AU888', 'B888', 'BC888', 'BU888', 'C888', 'CF888', 'CJ888', 'CS888', 'CU888', 'CY888', 'EB888', 'EG888',
                    'FB888', 'FG888', 'FU888', 'HC888', 'I888', 'J888', 'JD888', 'JM888', 'L888', 'LH888', 'LU888', 'M888', 'MA888', 'NI888', 'NR888', 'OI888',
                    'P888', 'PB888', 'PF888', 'PG888', 'PK888', 'PP888', 'RB888', 'RM888', 'RU888', 'SA888', 'SC888', 'SF888', 'SM888', 'SN888', 'SP888', 'SR888',
                    'SS888', 'TA888', 'UR888', 'V888', 'Y888', 'ZC888', 'ZN888']
year = [2020, 2021, 2022]
month = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
#%%
all_data = pd.DataFrame()
for order_book_id in main_future_list:
    for i in year:
        for j in month:
            filters = [('order_book_id', '=', order_book_id), ('year', '=', i), ('month', '=', j)]
            data = pq.ParquetDataset('futures/main_eight_kline', filters=filters, filesystem=minio, schema=schema)
            data_kline = data.read_pandas().to_pandas()
            data_kline['datetime'] = pd.to_datetime(data_kline['close_timestamp']+28800, unit='s')
            all_data = all_data.append(data_kline)
            print(all_data)

#%%
# data_time = pd.read_csv('/run/media/ps/data/songhe/future/all_data_future.csv')
#%%
data_time = all_data.loc[:,['datetime','close_timestamp','open', 'high', 'low', 'close','closse_volume','order_book_id']]
data_time = data_time.rename({'closse_volume': 'volume','order_book_id':'future'}, axis='columns')
#%%
import ta
from ta.volume import ForceIndexIndicator, EaseOfMovementIndicator
from ta.volatility import BollingerBands, KeltnerChannel, DonchianChannel
from ta.trend import MACD, macd_diff, macd_signal, SMAIndicator
from ta.momentum import stochrsi, stochrsi_k, stochrsi_d
# data_time = all_data.copy()
#%%
import talib
from talib import stream
#%%
forceindex_30 = ForceIndexIndicator(close=data_time['close'], volume=data_time['volume'], window=30)
data_time['forceindex_30'] = forceindex_30.force_index()
easyofmove_30 = EaseOfMovementIndicator(high=data_time['high'], low=data_time['low'], volume=data_time['volume'], window=30)
data_time['easyofmove_30'] = easyofmove_30.ease_of_movement()
easyofmove_60 = EaseOfMovementIndicator(high=data_time['high'], low=data_time['low'], volume=data_time['volume'], window=60)
data_time['easyofmove_60'] = easyofmove_60.ease_of_movement()
bollingband_30 = BollingerBands(close=data_time['close'], window=30, window_dev=30)
data_time['bollingerhband_30'] = bollingband_30.bollinger_hband()
data_time['bollingerlband_30'] = bollingband_30.bollinger_lband()
data_time['bollingermband_30'] = bollingband_30.bollinger_mavg()
data_time['bollingerpband_30'] = bollingband_30.bollinger_pband()
data_time['bollingerwband_30'] = bollingband_30.bollinger_wband()
bollingband_60 = BollingerBands(close=data_time['close'], window=60, window_dev=60)
data_time['bollingerhband_60'] = bollingband_60.bollinger_hband()
data_time['bollingerlband_60'] = bollingband_60.bollinger_lband()
data_time['bollingermband_60'] = bollingband_60.bollinger_mavg()
data_time['bollingerpband_60'] = bollingband_60.bollinger_pband()
data_time['bollingerwband_60'] = bollingband_60.bollinger_wband()
keltnerchannel_30 = KeltnerChannel(high=data_time['high'], low=data_time['low'], close=data_time['close'], window=30)
data_time['keltnerhband_30'] = keltnerchannel_30.keltner_channel_hband()
data_time['keltnerlband_30'] = keltnerchannel_30.keltner_channel_lband()
data_time['keltnerwband_30'] = keltnerchannel_30.keltner_channel_wband()
data_time['keltnerpband_30'] = keltnerchannel_30.keltner_channel_pband()
# keltnerchannel_60 = KeltnerChannel(high=data_time['high'], low=data_time['low'], close=data_time['close'], window=60)
# data_time['keltnerhband_60'] = keltnerchannel_60.keltner_channel_hband()
# data_time['keltnerlband_60'] = keltnerchannel_60.keltner_channel_lband()
# data_time['keltnerwband_60'] = keltnerchannel_60.keltner_channel_wband()
# data_time['keltnerpband_60'] = keltnerchannel_60.keltner_channel_pband()
donchichannel_30 = DonchianChannel(high=data_time['high'], low=data_time['low'], close=data_time['close'],window=30)
data_time['donchimband_30'] = donchichannel_30.donchian_channel_mband()
data_time['donchilband_30'] = donchichannel_30.donchian_channel_lband()
data_time['donchipband_30'] = donchichannel_30.donchian_channel_pband()
data_time['donchiwband_30'] = donchichannel_30.donchian_channel_wband()
# donchichannel_60 = DonchianChannel(high=data_time['high'], low=data_time['low'], close=data_time['close'],window=60)
# data_time['donchimband_60'] = donchichannel_60.donchian_channel_mband()
# data_time['donchilband_60'] = donchichannel_60.donchian_channel_lband()
# data_time['donchipband_60'] = donchichannel_60.donchian_channel_pband()
# data_time['donchiwband_60'] = donchichannel_60.donchian_channel_wband()
macd_30 = MACD(close=data_time['close'],window_fast=30, window_slow=60)
data_time['macd_30'] = macd_30.macd()
# macd_60 = MACD(close=data_time['close'],window_fast=60, window_slow=120)
# data_time['macd_60'] = macd_60.macd()
data_time['macdsignal_30'] = macd_signal(close=data_time['close'],window_fast=30,window_slow=60)
# data_time['macdsignal_60'] = macd_signal(close=data_time['close'],window_fast=60,window_slow=120)
data_time['macddiff_30'] = macd_diff(close=data_time['close'],window_fast=30, window_slow=60)
# data_time['macddiff_60'] = macd_diff(close=data_time['close'],window_fast=60, window_slow=120)
smafast_30 = SMAIndicator(close=data_time['close'],window=30)
data_time['smafast_30'] = smafast_30.sma_indicator()
# smafast_60 = SMAIndicator(close=data_time['close'],window=60)
# data_time['smafast_60'] = smafast_60.sma_indicator()
# smaslow_120 = SMAIndicator(close=data_time['close'],window=120)
# data_time['smaslow_120'] = smaslow_120.sma_indicator()
data_time['stochrsi_30'] = stochrsi(close=data_time['close'],window=30, smooth1=30, smooth2=15)
data_time['stochrsi_k_30'] = stochrsi_k(close=data_time['close'],window=30, smooth1=30, smooth2=15)
data_time['stochrsi_d_30'] = stochrsi_d(close=data_time['close'],window=30, smooth1=30, smooth2=15)
# data_time['stochrsi_60'] = stochrsi(close=data_time['close'],window=60, smooth1=30, smooth2=15)
# data_time['stochrsi_k_60'] = stochrsi_k(close=data_time['close'],window=60, smooth1=30, smooth2=15)
# data_time['stochrsi_d_60'] = stochrsi_d(close=data_time['close'],window=60, smooth1=30, smooth2=15)
#%%
data_time['datetime'] = pd.to_datetime(data_time['datetime'])
data_time['time'] = data_time['datetime'].dt.strftime('%H:%M:%S')
data_time_9_00 = data_time[data_time['time'].isin(['09:00:00'])]
# data_time_9_15 = data_time[data_time['time'].isin(['09:15:00'])]   #开仓15分钟
data_time_9_30 = data_time[data_time['time'].isin(['09:30:00'])]
# data_time_10_00 = data_time[data_time['time'].isin(['10:00:00'])]  #开仓60分钟
data_time_11_30 = data_time[data_time['time'].isin(['11:29:00'])]
# data_time_13_30 = data_time[data_time['time'].isin(['13:30:00'])]
# data_time_15_00 = data_time[data_time['time'].isin(['14:59:00'])]
data_time_21_30 = data_time[data_time['time'].isin(['21:30:00'])]
# data_time_21_15 = data_time[data_time['time'].isin(['21:15:00'])]   #开仓15分钟
# data_time_22_00 = data_time[data_time['time'].isin(['22:00:00'])]   #开仓60分钟
# data_time_9_00 = data_time[data_time['time'].isin(['09:00:00'])]
# data_time_2_00 = data_time[data_time['time'].isin(['01:59:00'])]
#%%
data_time_9_00['date'] = data_time_9_00['datetime'].dt.strftime('%Y-%m-%d')
# data_time_9_15['date'] = data_time_9_15['datetime'].dt.strftime('%Y-%m-%d')
# data_time_10_00['date'] = data_time_10_00['datetime'].dt.strftime('%Y-%m-%d')
data_time_9_30['date'] = data_time_9_30['datetime'].dt.strftime('%Y-%m-%d')
data_time_11_30['date'] = data_time_11_30['datetime'].dt.strftime('%Y-%m-%d')
# data_time_13_30['date'] = data_time_13_30['datetime'].dt.strftime('%Y-%m-%d')
# data_time_15_00['date'] = data_time_15_00['datetime'].dt.strftime('%Y-%m-%d')
data_time_21_30['date'] = data_time_21_30['datetime'].dt.strftime('%Y-%m-%d')
# data_time_21_15['date'] = data_time_21_15['datetime'].dt.strftime('%Y-%m-%d')
# data_time_22_00['date'] = data_time_22_00['datetime'].dt.strftime('%Y-%m-%d')
# data_time_9_00['date'] = data_time_9_00['datetime'].dt.strftime('%Y-%m-%d')
# data_time_2_00['date'] = data_time_2_00['datetime'].dt.strftime('%Y-%m-%d')
#%%
# data_time_9_15 = data_time_9_15.reset_index(drop=True)
# data_time_10_00 = data_time_10_00.reset_index(drop=True)
data_time_9_30 = data_time_9_30.reset_index(drop=True)
data_time_11_30 = data_time_11_30.reset_index(drop=True)
# data_time_13_30 = data_time_13_30.reset_index(drop=True)
# data_time_15_00 = data_time_15_00.reset_index(drop=True)
data_time_21_30 = data_time_21_30.reset_index(drop=True)
# data_time_21_15 = data_time_21_15.reset_index(drop=True)
# data_time_22_00 = data_time_22_00.reset_index(drop=True)
#%%
data_time_9_00 = data_time_9_00.reset_index(drop=True)
# data_time_2_00 = data_time_2_00.reset_index(drop=True)
#%%
final_data_9_00 = data_time_9_00[['date','close','future']]
final_data_9_30 = data_time_9_30[['date','close','future']]
final_data_11_30 = data_time_11_30[['date','close','future']]
# final_data_13_30 = data_time_13_30[['date','close','future']]
# final_data_15_00 = data_time_15_00[['date','close','future']]
# final_data_2_00 = data_time_2_00[['date','close','future']]
final_data_21_30 = data_time_21_30[['date','close','future']]
#%%
a = pd.merge(data_time_9_30, data_time_21_30, on=['date', 'future'], how='left')
# a = pd.merge(data_time_9_15, data_time_21_15, on=['date', 'future'], how='left')
# a = pd.merge(data_time_10_00, data_time_22_00, on=['date', 'future'], how='left')
#%%
# a['future_y'] = a['future']
#%%
a = a.set_index(['date', 'future'])
#%%
df1 = a[a.isna().any(axis=1)] #have nan
df2 = a[~a.isna().any(axis=1)] # do not have nan
#%%
col_9_30 = df1.iloc[:,0:11]
col_21_30 = df2.iloc[:,11:]
col = col_21_30.columns
col_9_30.columns = col
#%%
# col_21_15 = col_21_15.reset_index()
# c_21_15_11_30 = pd.merge(col_21_15, final_data_11_30, on=['date','future'], how='left')
# c_21_15_11_30['close_11_30'] = c_21_15_11_30['close'].shift(-1)
# c_21_15_11_30 = c_21_15_11_30.dropna(axis=0)
# c_21_15_11_30 = c_21_15_11_30.drop(['close'],axis=1)
#
# col_9_15 = col_9_15.reset_index()
# c_9_15_11_30 = pd.merge(col_9_15, final_data_11_30,on=['date','future'], how='left')
# c_9_15_11_30['close_11_30'] = c_9_15_11_30['close']
# c_9_15_11_30 = c_9_15_11_30.drop(['close'],axis=1)
#
# c_final_data = pd.concat([c_21_15_11_30,c_9_15_11_30])
# c_final_data = c_final_data.set_index(['date','future']).sort_index()
# c_final_data['target'] = np.log(c_final_data['close_11_30']/c_final_data['close_y'])*100
# c_final_data = c_final_data.reset_index()
# c_final_data = c_final_data[~c_final_data['date'].isin(['2022-05-13'])]
#%%
# col_22_00 = col_22_00.reset_index()
# c_22_00_11_30 = pd.merge(col_22_00, final_data_11_30, on=['date','future'], how='left')
# c_22_00_11_30['close_11_30'] = c_22_00_11_30['close'].shift(-1)
# c_22_00_11_30 = c_22_00_11_30.dropna(axis=0)
# c_22_00_11_30 = c_22_00_11_30.drop(['close'],axis=1)
#
# col_10_00 = col_10_00.reset_index()
# c_10_00_11_30 = pd.merge(col_10_00, final_data_11_30,on=['date','future'], how='left')
# c_10_00_11_30['close_11_30'] = c_10_00_11_30['close']
# c_10_00_11_30 =c_10_00_11_30.drop(['close'],axis=1)
#
# c_final_data = pd.concat([c_22_00_11_30,c_10_00_11_30])
# c_final_data = c_final_data.set_index(['date','future']).sort_index()
# c_final_data['target'] = np.log(c_final_data['close_11_30']/c_final_data['close_y'])*100
# c_final_data = c_final_data.reset_index()
# c_final_data = c_final_data[~c_final_data['date'].isin(['2022-05-13'])]
#%% 夜盘品种夜盘收盘平仓 白天盘品种转天9点平仓
# col_21_15 = col_21_15.reset_index()
# c_21_15_2_00 = pd.merge(col_21_15, final_data_2_00, on=['date','future'], how='left')
# c_21_15_2_00['close_x'] = c_21_15_2_00['close'].shift(-1)
# c_21_15_2_00 = c_21_15_2_00.dropna(axis=0)
# c_21_15_2_00 = c_21_15_2_00.drop(['close'],axis=1)

# col_9_15 = col_9_15.reset_index()
# c_9_15_9_00 = pd.merge(col_9_15, final_data_9_00,on=['date','future'], how='left')
# c_9_15_9_00['close_x'] = c_9_15_9_00['close'].shift(-2)
# c_9_15_9_00 = c_9_15_9_00.dropna(axis=0)
# c_9_15_9_00 = c_9_15_9_00.drop(['close'],axis=1)

# c_final_data = pd.concat([c_21_15_2_00,c_9_15_9_00])
# c_final_data = c_final_data.set_index(['date','future']).sort_index()
# c_final_data['target'] = np.log(c_final_data['close_x']/c_final_data['close_y'])*100
# c_final_data = c_final_data.reset_index()
# c_final_data = c_final_data[~c_final_data['date'].isin(['2022-05-13'])]
#%%
col_21_30 = col_21_30.reset_index()
c_21_30_11_30 = pd.merge(col_21_30, final_data_11_30, on=['date','future'], how='left')
c_21_30_11_30['close_11_30'] = c_21_30_11_30['close'].shift(-1)
c_21_30_11_30 = c_21_30_11_30.dropna(axis=0)
c_21_30_11_30 = c_21_30_11_30.drop(['close'],axis=1)

col_9_30 = col_9_30.reset_index()
c_9_30_11_30 = pd.merge(col_9_30, final_data_11_30,on=['date','future'], how='left')
c_9_30_11_30['close_11_30'] = c_9_30_11_30['close']
c_9_30_11_30 = c_9_30_11_30.drop(['close'],axis=1)

c_final_data = pd.concat([c_21_30_11_30,c_9_30_11_30])
c_final_data = c_final_data.set_index(['date','future']).sort_index()
c_final_data['target'] = np.log(c_final_data['close_11_30']/c_final_data['close_y'])*100
c_final_data = c_final_data.reset_index()
c_final_data = c_final_data[~c_final_data['date'].isin(['2022-06-01'])]
#%%
# c_21_30_13_30 = pd.merge(col_21_30, final_data_13_30, on=['date','future'], how='left')
# c_21_30_13_30['close_13_30'] = c_21_30_13_30['close'].shift(-1)
# c_21_30_13_30 = c_21_30_13_30.dropna(axis=0)
# c_21_30_13_30 = c_21_30_13_30.drop(['close'],axis=1)
#
# c_9_30_13_30 = pd.merge(col_9_30, final_data_13_30,on=['date','future'], how='left')
# c_9_30_13_30['close_13_30'] = c_9_30_13_30['close']
# c_9_30_13_30 = c_9_30_13_30.drop(['close'],axis=1)
#
# c_final_data_2 = pd.concat([c_21_30_13_30,c_9_30_13_30])
# c_final_data_2 = c_final_data_2.set_index(['date','future']).sort_index()
# c_final_data_2['target'] = np.log(c_final_data_2['close_13_30']/c_final_data_2['close_y'])*100
#
# c_final_data_2 = c_final_data_2.reset_index()
# c_final_data_2 = c_final_data_2[~c_final_data_2['date'].isin(['2022-05-13'])]
#%%
# col_21_15 = col_21_15.reset_index()
# c_21_15_15_00 = pd.merge(col_21_15, final_data_15_00, on=['date','future'], how='left')
# c_21_15_15_00['close_15_00'] = c_21_15_15_00['close'].shift(-1)
# c_21_15_15_00 = c_21_15_15_00.dropna(axis=0)
# c_21_15_15_00 = c_21_15_15_00.drop(['close'],axis=1)
#
# # col_9_15 = col_9_15.reset_index()
# c_9_15_15_00 = pd.merge(col_9_15, final_data_15_00,on=['date','future'], how='left')
# c_9_15_15_00['close_15_00'] = c_9_15_15_00['close']
# c_9_15_15_00 = c_9_15_15_00.drop(['close'],axis=1)
#
# c_final_data_2 = pd.concat([c_21_15_15_00,c_9_15_15_00])
# c_final_data_2 = c_final_data_2.set_index(['date','future']).sort_index()
# c_final_data_2['target'] = np.log(c_final_data_2['close_15_00']/c_final_data_2['close_y'])*100
# c_final_data_2 = c_final_data_2.reset_index()
# c_final_data_2 = c_final_data_2[~c_final_data_2['date'].isin(['2022-05-13'])]
#%%
# col_21_15 = col_21_15.reset_index()
# c_22_00_15_00 = pd.merge(col_22_00, final_data_15_00, on=['date','future'], how='left')
# c_22_00_15_00['close_15_00'] = c_22_00_15_00['close'].shift(-1)
# c_22_00_15_00 = c_22_00_15_00.dropna(axis=0)
# c_22_00_15_00 = c_22_00_15_00.drop(['close'],axis=1)
#
# # col_9_15 = col_9_15.reset_index()
# c_10_00_15_00 = pd.merge(col_10_00, final_data_15_00,on=['date','future'], how='left')
# c_10_00_15_00['close_15_00'] = c_10_00_15_00['close']
# c_10_00_15_00 = c_10_00_15_00.drop(['close'],axis=1)
#
# c_final_data_2 = pd.concat([c_22_00_15_00,c_10_00_15_00])
# c_final_data_2 = c_final_data_2.set_index(['date','future']).sort_index()
# c_final_data_2['target'] = np.log(c_final_data_2['close_15_00']/c_final_data_2['close_y'])*100
# c_final_data_2 = c_final_data_2.reset_index()
# c_final_data_2 = c_final_data_2[~c_final_data_2['date'].isin(['2022-05-13'])]
#%% 晚上9点半开仓，转天21点半平仓
col_21_30 = col_21_30.reset_index()
c_21_30_21_30 = pd.merge(col_21_30, final_data_21_30, on=['date','future'], how='left')
c_21_30_21_30['close_21_30'] = c_21_30_21_30['close'].shift(-1)
c_21_30_21_30 = c_21_30_21_30.dropna(axis=0)
c_21_30_21_30 = c_21_30_21_30.drop(['close'],axis=1)
c_21_30_21_30['target'] = np.log(c_21_30_21_30['close_21_30']/c_21_30_21_30['close_y'])
#%%
col_9_30 = col_9_30.reset_index()
c_9_30_9_30 = pd.merge(col_9_30, final_data_9_30,on=['date','future'], how='left')
c_9_30_9_30['close_9_30'] = c_9_30_9_30['close']
c_9_30_9_30 = c_9_30_9_30.drop(['close'],axis=1)
c_9_30_9_30['target'] = np.log(c_9_30_9_30['close_9_30']/c_9_30_9_30['close_y'])
#%%
c_final_data_3 = pd.concat([c_21_30_21_30,c_9_30_9_30])
c_final_data_3 = c_final_data_3.set_index(['date','future']).sort_index()
#%%
c_final_data_3 = c_final_data_3.drop(['close_9_30','close_21_30'], axis=1)
# c_final_data_3['target_21_30'] = np.log(c_final_data_3['close_21_30']/c_final_data_3['close_y'])*100
c_final_data_3 = c_final_data_3.reset_index()
#%%
c_final_data_3 = c_final_data_3[~c_final_data_3['date'].isin(['2022-06-22'])]
#%%0.498982
c_final_data.to_csv('all_contract_min_data_22_00_10_00_11_30_close_60var.csv')
#%%
c_final_data_2.to_csv('all_contract_min_data_22_00_10_00_15_00_close_60var.csv')
#%%
c_final_data_3.to_csv('all_data_all_future_anotherday.csv')
#%%
c_final_data.to_csv('test_51var.csv')