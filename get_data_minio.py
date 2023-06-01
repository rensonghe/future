import time
import pyarrow
from pyarrow import fs
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numba as nb
import numpy as np
schema = pa.schema([
    ('datetime',pa.string()),
    ('trading_date', pa.string()),
    ('open', pa.float64()),
    ('last', pa.float64()),
    ('high', pa.float64()),
    ('low', pa.float64()),
    ('prev_settlement', pa.float64()),
    ('prev_close', pa.float64()),
    ('open_interest', pa.float64()),
    ('volume', pa.float64()),
    ('total_turnover', pa.int64()),
    ('limit_up', pa.int64()),
    ('limit_down', pa.int64()),
    ('ask_price1', pa.float64()),
    ('ask_price2', pa.float64()),
    ('ask_price3', pa.float64()),
    ('ask_price4', pa.float64()),
    ('ask_price5', pa.float64()),
    ('bid_price1', pa.float64()),
    ('bid_price2', pa.float64()),
    ('bid_price3', pa.float64()),
    ('bid_price4', pa.float64()),
    ('bid_price5', pa.float64()),
    ('ask_size1', pa.float64()),
    ('ask_size2', pa.float64()),
    ('ask_size3', pa.float64()),
    ('ask_size4', pa.float64()),
    ('ask_size5', pa.float64()),
    ('bid_size1', pa.float64()),
    ('bid_size2', pa.float64()),
    ('bid_size3', pa.float64()),
    ('bid_size4', pa.float64()),
    ('bid_size5', pa.float64()),
    ('change_rate', pa.float64()),
    ('timestamp', pa.float64()),
    ('order_book_id',pa.string()),
    ('year', pa.int64()),
    ('month', pa.int64())
])
# 从minio 中拿数据
minio = fs.S3FileSystem(endpoint_override="192.168.34.57:9000", access_key="zVGhI7gEzJtcY5ph", secret_key="9n8VeSiudgnvzoGXxDoLTA6Y39Yg2mQx", scheme="http")
#%%
all_data = pd.DataFrame()
order_book_id = 'RB888'
platform = 'gate_swap_u'
year = 2022
# month = 2
for month in range(1, 2):
    # 拿orderbook的数据
    data_type = 'main_eight'
    filters = [('order_book_id', '=', order_book_id) , ('year', '=', year) , ('month','=',month)]
    data = pq.ParquetDataset('futures/{}'.format(data_type), filters=filters, filesystem=minio, schema=schema)
    data = data.read_pandas().to_pandas()
    # depth['datetime'] = pd.to_datetime(depth['close_timestamp']+28800, unit='s')
    data = data.sort_values(by='timestamp',ascending=True)
    data = data.iloc[:,:-3]
    # data = reduce_mem_usage(data)[0]
    data = data.reset_index(drop=True)
    all_data = all_data.append(data)
    print(all_data)
    # data.to_csv('D://test//{}//{}_feat_{}_{}.csv'.format(order_book_id, order_book_id, year, month), index=False)
#%%
data_2021 = all_data
#%%
data_2022 = all_data
#%%
# all_data = pd.concat([data_2021, data_2022], axis=0)
all_data = all_data.sort_values(by='timestamp', ascending=True)
all_data = all_data.rename({'timestamp': 'closetime', 'volume':'size', 'last':'price'}, axis='columns')
#%%
del data_2022, data_2021, data
#%%
trade = all_data.loc[:, ['closetime', 'price', 'size', 'total_turnover']]
depth = all_data.loc[:,
        ['closetime', 'ask_price1', 'ask_size1', 'bid_price1', 'bid_size1', 'ask_price2', 'ask_size2', 'bid_price2',
         'bid_size2', 'ask_price3', 'ask_size3', 'bid_price3', 'bid_size3', 'ask_price4', 'ask_size4', 'bid_price4',
         'bid_size4', 'ask_price5', 'ask_size5', 'bid_price5', 'bid_size5','size','open_interest','price']]
#%%  计算这一行基于bid和ask的wap
def calc_wap1(df):
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (
            df['bid_size1'] + df['ask_size1'])
    return wap


# Function to calculate second WAP
def calc_wap2(df):
    wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (
            df['bid_size2'] + df['ask_size2'])
    return wap


def calc_wap3(df):
    wap = (df['bid_price1'] * df['bid_size1'] + df['ask_price1'] * df['ask_size1']) / (
            df['bid_size1'] + df['ask_size1'])
    return wap


def calc_wap4(df):
    wap = (df['bid_price2'] * df['bid_size2'] + df['ask_price2'] * df['ask_size2']) / (
            df['bid_size2'] + df['ask_size2'])
    return wap


def calc_wap12(df):
    var1 = df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']
    var2 = df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']
    den = df['bid_size1'] + df['ask_size1'] + df['bid_size2'] + df['ask_size2']
    return (var1 + var2) / den


def calc_wap34(df):
    var1 = df['bid_price1'] * df['bid_size1'] + df['ask_price1'] * df['ask_size1']
    var2 = df['bid_price2'] * df['bid_size2'] + df['ask_price2'] * df['ask_size2']
    den = df['bid_size1'] + df['ask_size1'] + df['bid_size2'] + df['ask_size2']
    return (var1 + var2) / den


def calc_swap1(df):
    return df['wap1'] - df['wap3']


def calc_swap12(df):
    return df['wap12'] - df['wap34']


def calc_tswap1(df):
    return -df['swap1'].diff()


def calc_tswap12(df):
    return -df['swap12'].diff()


def calc_wss12(df):
    ask = (df['ask_price1'] * df['ask_size1'] + df['ask_price2'] * df['ask_size2']) / (
            df['ask_size1'] + df['ask_size2'])
    bid = (df['bid_price1'] * df['bid_size1'] + df['bid_price2'] * df['bid_size2']) / (
            df['bid_size1'] + df['bid_size2'])
    mid = (df['ask_price1'] + df['bid_price1']) / 2
    return (ask - bid) / mid


def calc_tt1(df):
    p1 = df['ask_price1'] * df['ask_size1'] + df['bid_price1'] * df['bid_size1']
    p2 = df['ask_price2'] * df['ask_size2'] + df['bid_price2'] * df['bid_size2']
    return p2 - p1


def calc_price_impact(df):
    ask = (df['ask_price1'] * df['ask_size1'] + df['ask_price2'] * df['ask_size2']) / (
            df['ask_size1'] + df['ask_size2'])
    bid = (df['bid_price1'] * df['bid_size1'] + df['bid_price2'] * df['bid_size2']) / (
            df['bid_size1'] + df['bid_size2'])
    return (df['ask_price1'] - ask) / df['ask_price1'], (df['bid_price1'] - bid) / df['bid_price1']


# Calculate order book slope
def calc_slope(df):
    v0 = (df['bid_size1'] + df['ask_size1']) / 2
    p0 = (df['bid_price1'] + df['ask_price1']) / 2
    slope_bid = ((df['bid_size1'] / v0) - 1) / abs((df['bid_price1'] / p0) - 1) + (
            (df['bid_size2'] / df['bid_size1']) - 1) / abs((df['bid_price2'] / df['bid_price1']) - 1)
    slope_ask = ((df['ask_size1'] / v0) - 1) / abs((df['ask_price1'] / p0) - 1) + (
            (df['ask_size2'] / df['ask_size1']) - 1) / abs((df['ask_price2'] / df['ask_price1']) - 1)
    return (slope_bid + slope_ask) / 2, abs(slope_bid - slope_ask)


# Calculate order book dispersion
def calc_dispersion(df):
    bspread = df['bid_price1'] - df['bid_price2']
    aspread = df['ask_price2'] - df['ask_price1']
    bmid = (df['bid_price1'] + df['ask_price1']) / 2 - df['bid_price1']
    bmid2 = (df['bid_price1'] + df['ask_price1']) / 2 - df['bid_price2']
    amid = df['ask_price1'] - (df['bid_price1'] + df['ask_price1']) / 2
    amid2 = df['ask_price2'] - (df['bid_price1'] + df['ask_price1']) / 2
    bdisp = (df['bid_size1'] * bmid + df['bid_size2'] * bspread) / (df['bid_size1'] + df['bid_size2'])
    bdisp2 = (df['bid_size1'] * bmid + df['bid_size2'] * bmid2) / (df['bid_size1'] + df['bid_size2'])
    adisp = (df['ask_size1'] * amid + df['ask_size2'] * aspread) / (df['ask_size1'] + df['ask_size2'])
    adisp2 = (df['ask_size1'] * amid + df['ask_size2'] * amid2) / (df['ask_size1'] + df['ask_size2'])
    return bspread, aspread, bmid, amid, bdisp, adisp, (bdisp + adisp) / 2, (bdisp2 + adisp2) / 2


# Calculate order book depth
def calc_depth(df):
    depth = df['bid_price1'] * df['bid_size1'] + df['ask_price1'] * df['ask_size1'] + df['bid_price2'] * df[
        'bid_size2'] + df['ask_price2'] * df['ask_size2']
    return depth


#  order flow imbalance
def calc_ofi(df):
    a = df['bid_size1'] * np.where(df['bid_price1'].diff() >= 0, 1, 0)
    b = df['bid_size1'].shift() * np.where(df['bid_price1'].diff() <= 0, 1, 0)
    c = df['ask_size1'] * np.where(df['ask_price1'].diff() <= 0, 1, 0)
    d = df['ask_size1'].shift() * np.where(df['ask_price1'].diff() >= 0, 1, 0)
    return (a - b - c + d).fillna(0)


# Function to calculate the log of the return
# Remember that logb(x / y) = logb(x) - logb(y)
def log_return(series):
    return np.log(series).diff()


# Calculate the realized volatility
def realized_volatility(series):
    return np.sqrt(np.sum(series ** 2))


def realized_quarticity(series):
    # return (np.sum(series**4)*series.shape[0]/3)
    return (series.count() / 3) * np.sum(series ** 4)


def reciprocal_transformation(series):
    return np.sqrt(1 / series) * 100000


def square_root_translation(series):
    return series ** (1 / 2)


# Calculate the realized absolute variation
def realized_absvar(series):
    return np.sqrt(np.pi / (2 * series.count())) * np.sum(np.abs(series))


# Calculate the realized skew
def realized_skew(series):
    return np.sqrt(series.count()) * np.sum(series ** 3) / (realized_volatility(series) ** 3)


# Calculate the realized kurtosis
def realized_kurtosis(series):
    return series.count() * np.sum(series ** 4) / (realized_volatility(series) ** 4)

@nb.jit
def get_age(prices):
    last_value = prices[-1]
    age = 0
    for i in range(2, len(prices)):
        if prices[-i] != last_value:
            return age
        age += 1
    return age


def ask_bid_age(depth, rolling=10):
    bp1 = depth['bid_price1']
    bp1_changes = bp1.rolling(rolling).apply(get_age, engine='numba', raw=True).fillna(0)
    return bp1_changes


def inf_ratio(depth=None, trade=None, rolling=100):
    quasi = trade.price.diff().abs().rolling(rolling).sum().fillna(10)
    dif = trade.price.diff(rolling).abs().fillna(10)
    return quasi / (dif + quasi)


def depth_price_range(depth=None, trade=None, rolling=100):
    return (depth.ask_price1.rolling(rolling).max() / depth.ask_price1.rolling(rolling).min() - 1).fillna(0)


def arrive_rate(depth, trade, rolling=300):
    res = trade['closetime'].diff(rolling).fillna(0) / rolling
    return res


def bp_rank(depth, trade, rolling=100):
    return ((depth.bid_price1.rolling(rolling).rank()) / rolling * 2 - 1).fillna(0)


def ap_rank(depth, trade, rolling=100):
    return ((depth.ask_price1.rolling(rolling).rank()) / rolling * 2 - 1).fillna(0)


def price_impact(depth, trade, level=5):
    ask, bid, ask_v, bid_v = 0, 0, 0, 0
    for i in range(1, level + 1):
        ask += depth[f'ask_price{i}'] * depth[f'ask_size{i}']
        bid += depth[f'bid_price{i}'] * depth[f'bid_size{i}']
        ask_v += depth[f'ask_size{i}']
        bid_v += depth[f'bid_size{i}']
    ask /= ask_v
    bid /= bid_v
    return pd.Series(
        -(depth['ask_price1'] - ask) / depth['ask_price1'] - (depth['bid_price1'] - bid) / depth['bid_price1'],
        name="price_impact")


def depth_price_skew(depth, trade):
    prices = ["bid_price5", "bid_price4", "bid_price3", "bid_price2", "bid_price1", "ask_price1", "ask_price2",
              "ask_price3", "ask_price4", "ask_price5"]
    return depth[prices].skew(axis=1)


def depth_price_kurt(depth, trade):
    prices = ["bid_price5", "bid_price4", "bid_price3", "bid_price2", "bid_price1", "ask_price1", "ask_price2",
              "ask_price3", "ask_price4", "ask_price5"]
    return depth[prices].kurt(axis=1)


def rolling_return(depth, trade, rolling=100):
    mp = ((depth.ask_price1 + depth.bid_price1) / 2)
    return (mp.diff(rolling) / mp).fillna(0)


def buy_increasing(depth, trade, rolling=100):
    v = trade['size'].copy()
    v[v < 0] = 0
    return np.log1p(((v.rolling(2 * rolling).sum() + 1) / (v.rolling(rolling).sum() + 1)).fillna(1))


def sell_increasing(depth, trade, rolling=100):
    v = trade['size'].copy()
    v[v > 0] = 0
    return np.log1p(((v.rolling(2 * rolling).sum() - 1) / (v.rolling(rolling).sum() - 1)).fillna(1))

@nb.jit
def first_location_of_maximum(x):
    max_value = max(x)  # 一个for 循环
    for loc in range(len(x)):
        if x[loc] == max_value:
            return loc + 1


def price_idxmax(depth, trade, rolling=20):
    return depth['ask_price1'].rolling(rolling).apply(first_location_of_maximum, engine='numba', raw=True).fillna(0)

@nb.jit
def mean_second_derivative_centra(x):
    sum_value = 0
    for i in range(len(x) - 5):
        sum_value += (x[i + 5] - 2 * x[i + 3] + x[i]) / 2
    return sum_value / (2 * (len(x) - 5))


def center_deri_two(depth, trade, rolling=20):
    return depth['ask_price1'].rolling(rolling).apply(mean_second_derivative_centra, engine='numba', raw=True).fillna(0)


def quasi(depth, trade, rolling=100):
    return depth.ask_price1.diff(1).abs().rolling(rolling).sum().fillna(0)


def last_range(depth, trade, rolling=100):
    return trade.price.diff(1).abs().rolling(rolling).sum().fillna(0)


# def arrive_rate(depth, trade, rolling=100):
#     return (trade.ts.shift(rolling) - trade.ts).fillna(0)

def avg_trade_volume(depth, trade, rolling=100):
    return (trade['size'][::-1].abs().rolling(rolling).sum().shift(-rolling + 1)).fillna(0)[::-1]


def avg_spread(depth, trade, rolling=200):
    return (depth.ask_price1 - depth.bid_price1).rolling(rolling).mean().fillna(0)


def avg_turnover(depth, trade, rolling=500):
    return depth[
        ['ask_size1', 'ask_size2', 'ask_size3', 'ask_size4', "ask_price5", 'bid_size1', 'bid_size2', 'bid_size3',
         'bid_size4', "bid_price5"]].sum(axis=1)


def abs_volume_kurt(depth, trade, rolling=500):
    return trade['size'].abs().rolling(rolling).kurt().fillna(0)


def abs_volume_skew(depth, trade, rolling=500):
    return trade['size'].abs().rolling(rolling).skew().fillna(0)


def volume_kurt(depth, trade, rolling=500):
    return trade['size'].rolling(rolling).kurt().fillna(0)


def volume_skew(depth, trade, rolling=500):
    return trade['size'].rolling(rolling).skew().fillna(0)


def price_kurt(depth, trade, rolling=500):
    return trade.price.rolling(rolling).kurt().fillna(0)


def price_skew(depth, trade, rolling=500):
    return trade.price.rolling(rolling).skew().abs().fillna(0)


def bv_divide_tn(depth, trade, rolling=10):
    bvs = depth.bid_size1 + depth.bid_size2 + depth.bid_size3 + depth.bid_size4 + depth.bid_size5

    def volume(depth, trade, rolling):
        return trade['size']

    v = volume(depth=depth, trade=trade, rolling=rolling)
    v[v > 0] = 0
    return (v.rolling(rolling).sum() / bvs).fillna(0)


def av_divide_tn(depth, trade, rolling=10):
    avs = depth.ask_size1 + depth.ask_size2 + depth.ask_size3 + depth.ask_size4 + depth.bid_size5

    def volume(depth, trade, n):
        return trade['size']

    v = volume(depth=depth, trade=trade, n=rolling)
    v[v < 0] = 0
    return (v.rolling(rolling).sum() / avs).fillna(0)


def weighted_price_to_mid(depth, trade, levels=5, alpha=1):
    def get_columns(name, levels):
        return [name + str(i) for i in range(1, levels + 1)]

    avs = depth[get_columns("ask_size", levels)]
    bvs = depth[get_columns("bid_size", levels)]
    aps = depth[get_columns("ask_price", levels)]
    bps = depth[get_columns("bid_price", levels)]
    mp = (depth['ask_price1'] + depth['bid_price1']) / 2
    return (avs.values * aps.values + bvs.values * bps.values).sum(axis=1) / (avs.values + bvs.values).sum(axis=1) - mp

@nb.njit
def _ask_withdraws_volume(l, n, levels=5):
    withdraws = 0
    for price_index in range(2, 2 + 4 * levels, 4):
        now_p = n[price_index]
        for price_last_index in range(2, 2 + 4 * levels, 4):
            if l[price_last_index] == now_p:
                withdraws -= min(n[price_index + 1] - l[price_last_index + 1], 0)

    return withdraws

@nb.njit
def _bid_withdraws_volume(l, n, levels=5):
    withdraws = 0
    for price_index in range(0, 4 * levels, 4):
        now_p = n[price_index]
        for price_last_index in range(0, 4 * levels, 4):
            if l[price_last_index] == now_p:
                withdraws -= min(n[price_index + 1] - l[price_last_index + 1], 0)

    return withdraws


def ask_withdraws(depth, trade):
    ob_values = depth.values
    flows = np.zeros(len(ob_values))
    for i in range(1, len(ob_values)):
        flows[i] = _ask_withdraws_volume(ob_values[i - 1], ob_values[i])
    return pd.Series(flows)


def bid_withdraws(depth, trade):
    ob_values = depth.values
    flows = np.zeros(len(ob_values))
    for i in range(1, len(ob_values)):
        flows[i] = _bid_withdraws_volume(ob_values[i - 1], ob_values[i])
    return pd.Series(flows)

# %%
lags = [300, 600, 900, 1800]
#%%
def depth_factor_process(data, rolling=60):
    df = data.loc[:, ['closetime', 'ask_price1', 'ask_size1', 'bid_price1', 'bid_size1',
                      'ask_price2', 'ask_size2', 'bid_price2', 'bid_size2', 'ask_price3',
                      'ask_size3', 'bid_price3', 'bid_size3', 'ask_price4', 'ask_size4',
                      'bid_price4', 'bid_size4','size','open_interest','price']]

    df['wap1'] = calc_wap1(df)
    df['wap2'] = calc_wap2(df)
    df['wap3'] = calc_wap3(df)
    df['wap4'] = calc_wap4(df)

    df['wap_balance1'] = abs(df['wap1'] - df['wap2'])
    df['wap_balance2'] = abs(df['wap1'] - df['wap3'])
    df['wap_balance3'] = abs(df['wap2'] - df['wap3'])
    df['wap_balance4'] = abs(df['wap3'] - df['wap4'])

    df['wap12'] = calc_wap12(df)
    df['wap34'] = calc_wap34(df)

    df['swap1'] = calc_swap1(df)
    df['swap12'] = calc_swap12(df)

    df['depth_1s_swap1_shift_1_diff'] = calc_tswap1(df)
    df['depth_1s_swap12_shift_1_diff'] = calc_tswap12(df)

    df['wss12'] = calc_wss12(df)
    df['tt1'] = calc_tt1(df)

    df['price_impact1'], df['price_impact2'] = calc_price_impact(df)

    df['slope1'], df['slope2'] = calc_slope(df)

    df['bspread'] = df['bid_price1'] - df['bid_price2']
    df['aspread'] = df['ask_price2'] - df['ask_price1']
    df['bmid'] = (df['bid_price1'] + df['ask_price1']) / 2 - df['bid_price1']
    df['bmid2'] = (df['bid_price1'] + df['ask_price1']) / 2 - df['bid_price2']
    df['amid'] = df['ask_price1'] - (df['bid_price1'] + df['ask_price1']) / 2
    df['amid2'] = df['ask_price2'] - (df['bid_price1'] + df['ask_price1']) / 2
    df['bdisp'] = (df['bid_size1'] * df['bmid'] + df['bid_size2'] * df['bspread']) / (df['bid_size1'] + df['bid_size2'])
    df['bdisp2'] = (df['bid_size1'] * df['bmid'] + df['bid_size2'] * df['bmid2']) / (df['bid_size1'] + df['bid_size2'])
    df['adisp'] = (df['ask_size1'] * df['amid'] + df['ask_size2'] * df['aspread']) / (df['ask_size1'] + df['ask_size2'])
    df['adisp2'] = (df['ask_size1'] * df['amid'] + df['ask_size2'] * df['amid2']) / (df['ask_size1'] + df['ask_size2'])

    df['depth'] = calc_depth(df)

    df['ofi'] = calc_ofi(df)

    df['bspread'], df['aspread'], df['bmid'], df['amid'], df['bdisp'], df['adisp'], df['bdisp_adisp'], df[
        'bdisp2_adisp2'] = calc_dispersion(df)

    df['HR1'] = ((df['bid_price1'] - df['bid_price1'].shift(1)) - (df['ask_price1'] - df['ask_price1'].shift(1))) / (
            (df['bid_price1'] - df['bid_price1'].shift(1)) + (df['ask_price1'] - df['ask_price1'].shift(1)))

    df['pre_vtA'] = np.where(df['ask_price1'] == df['ask_price1'].shift(1), df['ask_size1'] - df['ask_size1'].shift(1),
                             0)
    df['vtA'] = np.where(df['ask_price1'] > df['ask_price1'].shift(1), df['ask_size1'], df['pre_vtA'])
    df['pre_vtB'] = np.where(df['bid_price1'] == df['bid_price1'].shift(1), df['bid_size1'] - df['bid_size1'].shift(1),
                             0)
    df['vtB'] = np.where(df['bid_price1'] > df['bid_price1'].shift(1), df['bid_size1'], df['pre_vtB'])

    df['mid_price1'] = (df['ask_price1'] + df['bid_price1']) / 2
    df['mid_price2'] = (df['ask_price2'] + df['bid_price2']) / 2

    df['price_spread1'] = (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1']) / 2)
    df['price_spread2'] = (df['ask_price2'] - df['bid_price2']) / ((df['ask_price2'] + df['bid_price2']) / 2)
    df['price_spread3'] = (df['ask_price3'] - df['bid_price3']) / ((df['ask_price3'] + df['bid_price3']) / 2)
    df['price_spread4'] = (df['ask_price4'] - df['bid_price4']) / ((df['ask_price4'] + df['bid_price4']) / 2)

    df['bid_ask_size1_minus'] = df['bid_size1'] - df['ask_size1']
    df['bid_ask_size1_plus'] = df['bid_size1'] + df['ask_size1']
    df['bid_ask_size2_minus'] = df['bid_size2'] - df['ask_size2']
    df['bid_ask_size2_plus'] = df['bid_size2'] + df['ask_size2']
    df['bid_ask_size3_minus'] = df['bid_size3'] - df['ask_size3']
    df['bid_ask_size3_plus'] = df['bid_size3'] + df['ask_size3']
    df['bid_ask_size4_minus'] = df['bid_size4'] - df['ask_size4']
    df['bid_ask_size4_plus'] = df['bid_size4'] + df['ask_size4']

    # df['depth_1s_bid_size1_shift_1_diff'] = df['bid_size1'] - df['bid_size1'].shift()
    # df['depth_1s_ask_size1_shift_1_diff'] = df['ask_size1'] - df['ask_size1'].shift()
    # df['depth_1s_bid_size2_shift_1_diff'] = df['bid_size2'] - df['bid_size2'].shift()
    # df['depth_1s_ask_size2_shift_1_diff'] = df['ask_size2'] - df['ask_size2'].shift()
    # df['depth_1s_bid_size3_shift_1_diff'] = df['bid_size3'] - df['bid_size3'].shift()
    # df['depth_1s_ask_size3_shift_1_diff'] = df['ask_size3'] - df['ask_size3'].shift()

    df['bid_ask_size1_spread'] = df['bid_ask_size1_minus'] / df['bid_ask_size1_plus']
    df['bid_ask_size2_spread'] = df['bid_ask_size2_minus'] / df['bid_ask_size2_plus']
    df['bid_ask_size3_spread'] = df['bid_ask_size3_minus'] / df['bid_ask_size3_plus']
    df['bid_ask_size4_spread'] = df['bid_ask_size4_minus'] / df['bid_ask_size4_plus']

    df['HR2'] = ((df['bid_price2'] - df['bid_price2'].shift(1)) - (df['ask_price2'] - df['ask_price2'].shift(1))) / (
            (df['bid_price2'] - df['bid_price2'].shift(1)) + (df['ask_price2'] - df['ask_price2'].shift(1)))

    df['QR1'] = (df['bid_size1'] - df['ask_size1']) / (df['bid_size1'] + df['ask_size1'])
    df['QR2'] = (df['bid_size2'] - df['ask_size2']) / (df['bid_size2'] + df['ask_size2'])

    df['openInterestChg'] = df['open_interest'] - df['open_interest'].shift(1)
    df['sizeChg'] = df['size'] - df['size'].shift(1)
    df['openContract'] = (df['open_interest']+df['size'])/2
    df['closeContract'] = df['open_interest'] - df['size']



    for rolling in lags:


        # type1 type2 type3 type4
        df[f'type_1_{rolling}'] = np.where((df['open_interest']>df['open_interest'].shift(rolling))&(df['price']>df['price'].shift(rolling)),1,0)
        df[f'type_2_{rolling}'] = np.where((df['open_interest']>df['open_interest'].shift(rolling))&(df['price']<df['price'].shift(rolling)),2,0)
        df[f'type_3_{rolling}'] = np.where((df['open_interest']<df['open_interest'].shift(rolling))&(df['price']>df['price'].shift(rolling)),3,0)
        df[f'type_4_{rolling}'] = np.where((df['open_interest']<df['open_interest'].shift(rolling))&(df['price']<df['price'].shift(rolling)),4,0)
        df[f'type_1_pct_{rolling}'] = df[f'type_1_{rolling}'].apply(lambda x: x == 1).rolling(
            rolling).sum() / df[f'type_1_{rolling}'].apply(lambda x: x == x).rolling(
                                         rolling).sum()


        # # wap1 genetic functions
        # df[f'depth_1s_wap1_shift_{rolling}_log_return'] = np.log(df['wap1'].shift(1) / df['wap1'].shift(rolling))
        # df[f'depth_1s_wap1_rolling_{rolling}_realized_volatility'] = df['wap1'].rolling(rolling).apply(realized_volatility)
        # df[f'depth_1s_wap1_rolling_{rolling}_realized_absvar'] = df['wap1'].rolling(rolling).apply(realized_absvar)
        # df[f'depth_1s_wap1_rolling_{rolling}_realized_skew'] = df['wap1'].rolling(rolling).skew()
        # df[f'depth_1s_wap1_rolling_{rolling}_realized_kurtosis'] = df['wap1'].rolling(rolling).kurt()
        #
        # df[f'depth_1s_wap1_rolling_{rolling}_mean'] = df['wap1'].rolling(rolling).mean()
        # df[f'depth_1s_wap1_rolling_{rolling}_std'] = df['wap1'].rolling(rolling).std()
        # df[f'depth_1s_wap1_rolling_{rolling}_min'] = df['wap1'].rolling(rolling).min()
        # df[f'depth_1s_wap1_rolling_{rolling}_max'] = df['wap1'].rolling(rolling).max()
        #
        # df[f'depth_1s_wap1_rolling_{rolling}_mean/std'] = df[f'depth_1s_wap1_rolling_{rolling}_mean'] / df[
        #     f'depth_1s_wap1_rolling_{rolling}_std']
        #
        # df[f'depth_1s_wap1_rolling_{rolling}_quantile_25'] = df['wap1'].rolling(rolling).quantile(.25)
        # df[f'depth_1s_wap1_rolling_{rolling}_quantile_75'] = df['wap1'].rolling(rolling).quantile(.75)
        #
        # # wap2
        # df[f'depth_1s_wap2_shift1_{rolling}_log_return'] = np.log(df['wap2'].shift(1) / df['wap2'].shift(rolling))
        # df[f'depth_1s_wap2_rolling_{rolling}_realized_volatility'] = df['wap2'].rolling(rolling).apply(
        #     realized_volatility)
        # df[f'depth_1s_wap2_rolling_{rolling}_realized_absvar'] = df['wap2'].rolling(rolling).apply(realized_absvar)
        # df[f'depth_1s_wap2_rolling_{rolling}_realized_skew'] = df['wap2'].rolling(rolling).skew()
        # df[f'depth_1s_wap2_rolling_{rolling}_realized_kurtosis'] = df['wap2'].rolling(rolling).kurt()
        #
        # df[f'depth_1s_wap2_rolling_{rolling}_mean'] = df['wap2'].rolling(rolling).mean()
        # df[f'depth_1s_wap2_rolling_{rolling}_std'] = df['wap2'].rolling(rolling).std()
        # df[f'depth_1s_wap2_rolling_{rolling}_min'] = df['wap2'].rolling(rolling).min()
        # df[f'depth_1s_wap2_rolling_{rolling}_max'] = df['wap2'].rolling(rolling).max()
        #
        # df[f'depth_1s_wap2_rolling_{rolling}_mean/std'] = df[f'depth_1s_wap2_rolling_{rolling}_mean'] / df[
        #     f'depth_1s_wap2_rolling_{rolling}_std']
        #
        # df[f'depth_1s_wap2_rolling_{rolling}_quantile_25'] = df['wap2'].rolling(rolling).quantile(.25)
        # df[f'depth_1s_wap2_rolling_{rolling}_quantile_75'] = df['wap2'].rolling(rolling).quantile(.75)
        #
        # df[f'depth_1s_wap3_shift1_{rolling}_log_return'] = np.log(df['wap3'].shift(1) / df['wap3'].shift(rolling))
        # df[f'depth_1s_wap3_rolling_{rolling}_realized_volatility'] = df['wap3'].rolling(rolling).apply(
        #     realized_volatility)
        # df[f'depth_1s_wap3_rolling_{rolling}_realized_absvar'] = df['wap3'].rolling(rolling).apply(realized_absvar)
        # df[f'depth_1s_wap3_rolling_{rolling}_realized_skew'] = df['wap3'].rolling(rolling).skew()
        # df[f'depth_1s_wap3_rolling_{rolling}_realized_kurtosis'] = df['wap3'].rolling(rolling).kurt()
        #
        # df[f'depth_1s_wap3_rolling_{rolling}_mean'] = df['wap3'].rolling(rolling).mean()
        # df[f'depth_1s_wap3_rolling_{rolling}_std'] = df['wap3'].rolling(rolling).std()
        # df[f'depth_1s_wap3_rolling_{rolling}_min'] = df['wap3'].rolling(rolling).min()
        # df[f'depth_1s_wap3_rolling_{rolling}_max'] = df['wap3'].rolling(rolling).max()
        #
        # df[f'depth_1s_wap3_rolling_{rolling}_mean/std'] = df[f'depth_1s_wap3_rolling_{rolling}_mean'] / df[
        #     f'depth_1s_wap3_rolling_{rolling}_std']
        #
        # df[f'depth_1s_wap3_rolling_{rolling}_quantile_25'] = df['wap3'].rolling(rolling).quantile(.25)
        # df[f'depth_1s_wap3_rolling_{rolling}_quantile_75'] = df['wap3'].rolling(rolling).quantile(.75)
        #
        # # wap4 genetic functions
        # df[f'depth_1s_wap4_shift_{rolling}_log_return'] = np.log(df['wap4'].shift(1) / df['wap4'].shift(rolling))
        # df[f'depth_1s_wap4_rolling_{rolling}_realized_volatility'] = df['wap4'].rolling(rolling).apply(
        #     realized_volatility)
        # df[f'depth_1s_wap4_rolling_{rolling}_realized_absvar'] = df['wap4'].rolling(rolling).apply(realized_absvar)
        # df[f'depth_1s_wap4_rolling_{rolling}_realized_skew'] = df['wap4'].rolling(rolling).skew()
        # df[f'depth_1s_wap4_rolling_{rolling}_realized_kurtosis'] = df['wap4'].rolling(rolling).kurt()
        #
        # df[f'depth_1s_wap4_rolling_{rolling}_mean'] = df['wap4'].rolling(rolling).mean()
        # df[f'depth_1s_wap4_rolling_{rolling}_std'] = df['wap4'].rolling(rolling).std()
        # df[f'depth_1s_wap4_rolling_{rolling}_min'] = df['wap4'].rolling(rolling).min()
        # df[f'depth_1s_wap4_rolling_{rolling}_max'] = df['wap4'].rolling(rolling).max()
        #
        # df[f'depth_1s_wap4_rolling_{rolling}_mean/std'] = df[f'depth_1s_wap4_rolling_{rolling}_mean'] / df[
        #     f'depth_1s_wap4_rolling_{rolling}_std']
        #
        # df[f'depth_1s_wap4_rolling_{rolling}_quantile_25'] = df['wap4'].rolling(rolling).quantile(.25)
        # df[f'depth_1s_wap4_rolling_{rolling}_quantile_75'] = df['wap4'].rolling(rolling).quantile(.75)
        #
        # df[f'depth_1s_HR1_rolling_{rolling}_mean'] = df['HR1'].rolling(rolling).mean()
        # df[f'depth_1s_HR1_rolling_{rolling}_std'] = df['HR1'].rolling(rolling).std()
        # df[f'depth_1s_HR1_rolling_{rolling}_mean/std'] = df[f'depth_1s_HR1_rolling_{rolling}_mean'] / df[
        #     f'depth_1s_HR1_rolling_{rolling}_std']
        #
        # df[f'depth_1s_vtA_rolling_{rolling}_mean'] = df['vtA'].rolling(rolling).mean()
        # df[f'depth_1s_vtA_rolling_{rolling}_std'] = df['vtA'].rolling(rolling).std()
        # df[f'depth_1s_vtA_rolling_{rolling}_mean/std'] = df[f'depth_1s_vtA_rolling_{rolling}_mean'] / df[
        #     f'depth_1s_vtA_rolling_{rolling}_std']
        #
        # df[f'depth_1s_vtB_rolling_{rolling}_mean'] = df['vtB'].rolling(rolling).mean()
        # df[f'depth_1s_vtB_rolling_{rolling}_std'] = df['vtB'].rolling(rolling).std()
        # df[f'depth_1s_vtB_rolling_{rolling}_mean/std'] = df[f'depth_1s_vtB_rolling_{rolling}_mean'] / df[
        #     f'depth_1s_vtB_rolling_{rolling}_std']
        #
        # df['Oiab'] = df['vtB'] - df['vtA']
        # df[f'Oiab_{rolling}'] = df[f'depth_1s_vtB_rolling_{rolling}_mean'] - df[f'depth_1s_vtA_rolling_{rolling}_mean']
        # df[f'Oiab_{rolling}'] = df[f'depth_1s_vtB_rolling_{rolling}_mean'] - df[f'depth_1s_vtA_rolling_{rolling}_mean']
        # df[f'Oiab_{rolling}'] = df[f'depth_1s_vtB_rolling_{rolling}_mean'] - df[f'depth_1s_vtA_rolling_{rolling}_mean']
        #
        # df[f'depth_1s_bid_ask_size1_minus_rolling_{rolling}_mean'] = df[f'bid_ask_size1_minus'].rolling(rolling).mean()
        # df[f'depth_1s_bid_ask_size1_minus_rolling_{rolling}_std'] = df[f'bid_ask_size1_minus'].rolling(rolling).std()
        # df[f'depth_1s_bid_ask_size1_minus_rolling_{rolling}_mean/std'] = df[
        #                                                                      f'depth_1s_bid_ask_size1_minus_rolling_{rolling}_mean'] / \
        #                                                                  df[
        #                                                                      f'depth_1s_bid_ask_size1_minus_rolling_{rolling}_std']
        # df[f'depth_1s_bid_ask_size2_minus_rolling_{rolling}_mean'] = df['bid_ask_size2_minus'].rolling(rolling).mean()
        # df[f'depth_1s_bid_ask_size2_minus_rolling_{rolling}_std'] = df['bid_ask_size2_minus'].rolling(rolling).std()
        # df[f'depth_1s_bid_ask_size2_minus_rolling_{rolling}_mean/std'] = df[
        #                                                                      f'depth_1s_bid_ask_size2_minus_rolling_{rolling}_mean'] / \
        #                                                                  df[
        #                                                                      f'depth_1s_bid_ask_size2_minus_rolling_{rolling}_std']
        # df[f'depth_1s_bid_ask_size3_minus_rolling_{rolling}_mean'] = df['bid_ask_size3_minus'].rolling(rolling).mean()
        # df[f'depth_1s_bid_ask_size3_minus_rolling_{rolling}_std'] = df['bid_ask_size3_minus'].rolling(rolling).std()
        # df[f'depth_1s_bid_ask_size3_minus_rolling_{rolling}_mean/std'] = df[
        #                                                                      f'depth_1s_bid_ask_size3_minus_rolling_{rolling}_mean'] / \
        #                                                                  df[
        #                                                                      f'depth_1s_bid_ask_size3_minus_rolling_{rolling}_std']
        #
        # df[f'depth_1s_bid_ask_size1_spread_rolling_{rolling}_mean'] = df['bid_ask_size1_spread'].rolling(rolling).mean()
        # df[f'depth_1s_bid_ask_size1_spread_rolling_{rolling}_std'] = df['bid_ask_size1_spread'].rolling(rolling).std()
        # df[f'depth_1s_bid_ask_size1_spread_rolling_{rolling}_mean/std'] = df[
        #                                                                       f'depth_1s_bid_ask_size1_spread_rolling_{rolling}_mean'] / \
        #                                                                   df[
        #                                                                       f'depth_1s_bid_ask_size1_spread_rolling_{rolling}_std']
        # df[f'depth_1s_bid_ask_size2_spread_rolling_{rolling}_mean'] = df['bid_ask_size2_spread'].rolling(rolling).mean()
        # df[f'depth_1s_bid_ask_size2_spread_rolling_{rolling}_std'] = df['bid_ask_size2_spread'].rolling(rolling).std()
        # df[f'depth_1s_bid_ask_size2_spread_rolling_{rolling}_mean/std'] = df[
        #                                                                       f'depth_1s_bid_ask_size2_spread_rolling_{rolling}_mean'] / \
        #                                                                   df[
        #                                                                       f'depth_1s_bid_ask_size2_spread_rolling_{rolling}_std']
        #
        # df[f'bidprice1_press_rolling_{rolling}'] = (df['mid_price1'] / (df['bid_price1'] - df['mid_price1'])) / (
        #         df['mid_price1'] / (df['bid_price1'] - df['mid_price1'])).rolling(rolling).sum()
        # df[f'askprice1_press_rolling_{rolling}'] = (df['mid_price1'] / (df['ask_price1'] - df['mid_price1'])) / (
        #         df['mid_price1'] / (df['ask_price1'] - df['mid_price1'])).rolling(rolling).sum()
        # df[f'bidprice2_press_rolling_{rolling}'] = (df['mid_price2'] / (df['bid_price2'] - df['mid_price2'])) / (
        #         df['mid_price2'] / (df['bid_price2'] - df['mid_price2'])).rolling(rolling).sum()
        # df[f'askprice2_press_rolling_{rolling}'] = (df['mid_price2'] / (df['ask_price2'] - df['mid_price2'])) / (
        #         df['mid_price2'] / (df['ask_price2'] - df['mid_price2'])).rolling(rolling).sum()
        #
        # df[f'bidask1_press_rolling_{rolling}'] = np.log(
        #     (df[f'bidprice1_press_rolling_{rolling}'] * df['bid_size1'].rolling(rolling).sum()) / (
        #         df[f'askprice1_press_rolling_{rolling}']) * df[
        #         'ask_size1'].rolling(rolling).sum())
        # df[f'bidask2_press_rolling_{rolling}'] = np.log(
        #     (df[f'bidprice2_press_rolling_{rolling}'] * df['bid_size2'].rolling(rolling).sum()) / (
        #         df[f'askprice2_press_rolling_{rolling}']) * df[
        #         'ask_size2'].rolling(rolling).sum())

    # df = df.fillna(method='ffill')
    # df = df.fillna(method='bfill')
    # df = df.replace(np.inf, 1)
    # df = df.replace(-np.inf, -1)

    return df

def trade_factor_process(data, rolling=10):
    df = data.loc[:, ['closetime', 'price', 'size']]
    df['BS'] = np.where(df['size'] > 0, 'B', (np.where(df['size'] < 0, 'S', 0)))
    df['active_buy'] = np.where(df['BS'] == 'B', df['price'], 0)
    df['active_sell'] = np.where(df['BS'] == 'S', df['price'], 0)
    df = df.drop(['BS'], axis=1)
    for rolling in lags:
        df[f'buy_ratio_rolling_{rolling}'] = (df['active_buy'] * df['size']).rolling(rolling).mean() / (
                df['active_buy'] * df['size']).rolling(rolling).std()
        df[f'sell_ratio_rolling_{rolling}'] = (df['active_sell'] * abs(df['size'])).rolling(rolling).mean() / (
                df['active_buy'] * abs(df['size'])).rolling(rolling).std()

        df[f'depth_1s_last_price_shift_{rolling}_60_log_return'] = np.log(
            df['price'].shift(1) / df['price'].shift(rolling))
        # realized volatility
        df[f'depth_1s_log_return_rolling_{rolling}_realized_volatility'] = df[
            f'depth_1s_last_price_shift_{rolling}_60_log_return'].rolling(rolling).apply(realized_volatility)
        # realized absvar
        df[f'depth_1s_log_return_rolling_{rolling}_realized_absvar'] = df[
            f'depth_1s_last_price_shift_{rolling}_60_log_return'].rolling(rolling).apply(realized_absvar)
        # realized skew
        df[f'depth_1s_log_return_rolling_{rolling}_realized_skew'] = df[
            f'depth_1s_last_price_shift_{rolling}_60_log_return'].rolling(rolling).skew()
        # realized kurt
        df[f'depth_1s_log_return_rolling_{rolling}_realized_skew'] = df[
            f'depth_1s_last_price_shift_{rolling}_60_log_return'].rolling(rolling).kurt()

        df[f'depth_1s_log_rolling_{rolling}_quantile_25'] = df[
            f'depth_1s_last_price_shift_{rolling}_60_log_return'].rolling(rolling).quantile(.25)
        df[f'depth_1s_log_rolling_{rolling}_quantile_75'] = df[
            f'depth_1s_last_price_shift_{rolling}_60_log_return'].rolling(rolling).quantile(.75)

        df[f'depth_1s_log_percentile_rolling_{rolling}'] = df[f'depth_1s_log_rolling_{rolling}_quantile_75'] - df[
            f'depth_1s_log_rolling_{rolling}_quantile_25']

        df[f'depth_1s_size_rolling_{rolling}_realized_absvar'] = df['size'].rolling(rolling).apply(realized_absvar)

        df[f'depth_1s_size_rolling_{rolling}_quantile_25'] = df['size'].rolling(rolling).quantile(.25)
        df[f'depth_1s_size_rolling_{rolling}_quantile_75'] = df['size'].rolling(rolling).quantile(.75)
        df[f'depth_1s_size_percentile_rolling_{rolling}'] = df[f'depth_1s_size_rolling_{rolling}_quantile_75'] - df[
            f'depth_1s_size_rolling_{rolling}_quantile_25']

        # amount genetic functions
        df['amount'] = df['price'] * df['size']

        df['trade_mid_price'] = np.where(df['size'] > 0, (df['amount'] - df['amount'].shift(1)) / df['size'],
                                         df['price'])
        df[f'depth_1s_mid_price_rolling_{rolling}_mean'] = df['trade_mid_price'].rolling(rolling).mean()
        df[f'depth_1s_mid_price_rolling_{rolling}_std'] = df['trade_mid_price'].rolling(rolling).std()
        df[f'depth_1s_mid_price_rolling_{rolling}_mean/std'] = df[f'depth_1s_mid_price_rolling_{rolling}_mean'] / df[
            f'depth_1s_mid_price_rolling_{rolling}_std']

        df[f'depth_1s_amount_rolling_{rolling}_mean'] = df['amount'].rolling(rolling).mean()
        df[f'depth_1s_amount_rolling_{rolling}_std'] = df['amount'].rolling(rolling).std()
        df[f'depth_1s_amount_rolling_{rolling}_mean/std'] = df[f'depth_1s_amount_rolling_{rolling}_mean'] / df[
            f'depth_1s_amount_rolling_{rolling}_std']
        df[f'depth_1s_amount_rolling_{rolling}_quantile_25'] = df['amount'].rolling(rolling).quantile(.25)
        df[f'depth_1s_amount_rolling_{rolling}_quantile_75'] = df['amount'].rolling(rolling).quantile(.75)

    # df = df.fillna(0)
    # df = df.replace(np.inf, 1)
    # df = df.replace(-np.inf, -1)

    return df
#%%
def add_factor_process(depth, trade):

    # df = pd.DataFrame()
    df = depth.loc[:,
        ['closetime', 'ask_price1', 'ask_size1', 'bid_price1', 'bid_size1', 'ask_price2', 'ask_size2', 'bid_price2',
         'bid_size2', 'ask_price3', 'ask_size3', 'bid_price3', 'bid_size3', 'ask_price4', 'ask_size4', 'bid_price4',
         'bid_size4', 'ask_price5', 'ask_size5', 'bid_price5', 'bid_size5']]
    # df['closetime'] = depth['closetime']
    df['price'] = trade.loc[:,['price']]
    df['size'] = trade.loc[:, ['size']]
    df['total_turnover'] = trade.loc[:, ['total_turnover']]
    df['ask_bid_age'] = ask_bid_age(depth=depth, rolling=10)
    df['inf_ratio'] = inf_ratio(depth=None, trade=trade, rolling=100)
    df['arrive_rate'] = arrive_rate(depth=None, trade=trade, rolling=300)
    df['depth_price_range'] = depth_price_range(depth=depth, trade=None)
    df['bp_rank'] = bp_rank(depth=depth, trade=None, rolling=100)
    df['ap_rank'] = ap_rank(depth=depth, trade=None, rolling=100)
    df['price_impact'] = price_impact(depth=depth, trade=None, level=5)
    df['depth_price_skew'] = depth_price_skew(depth=depth, trade=None)
    df['depth_price_kurt'] = depth_price_kurt(depth=depth, trade=None)
    df['rolling_return'] = rolling_return(depth=depth, trade=None, rolling=100)
    df['buy_increasing'] = buy_increasing(depth=None, trade=trade, rolling=100)
    df['sell_increasing'] = sell_increasing(depth=None, trade=trade, rolling=100)
    df['price_idxmax'] = price_idxmax(depth=depth, trade=None, rolling=20)
    df['center_deri_two'] = center_deri_two(depth=depth, trade=None, rolling=20)
    df['quasi'] = quasi(depth=depth, trade=None, rolling=100)
    df['last_range'] = last_range(depth=None, trade=trade, rolling=100)
    df['avg_trade_volume'] = avg_trade_volume(depth=depth, trade=trade, rolling=100)
    df['avg_spread'] = avg_spread(depth=depth, trade=None, rolling=200)
    df['avg_turnover'] = avg_turnover(depth=depth, trade=trade, rolling=500)
    df['abs_volume_kurt'] = abs_volume_kurt(depth=None, trade=trade, rolling=500)
    df['abs_volume_skew'] = abs_volume_skew(depth=None, trade=trade, rolling=500)
    df['volume_kurt'] = volume_kurt(depth=None, trade=trade, rolling=500)
    df['volume_skew'] = volume_skew(depth=None, trade=trade, rolling=500)
    df['price_kurt'] = price_kurt(depth=None, trade=trade, rolling=500)
    df['price_skew'] = price_skew(depth=None, trade=trade, rolling=500)
    df['bv_divide_tn'] = bv_divide_tn(depth=depth, trade=trade, rolling=10)
    df['av_divide_tn'] = av_divide_tn(depth=depth, trade=trade, rolling=10)
    df['weighted_price_to_mid'] = weighted_price_to_mid(depth=depth, trade=None, levels=5, alpha=1)
    df['ask_withdraws'] = ask_withdraws(depth=depth, trade=None)
    df['bid_withdraws'] = bid_withdraws(depth=depth, trade=None)

    return df
#%%
def order_aggressiveness(data, rolling=10):

    df = data.loc[:, ['closetime', 'ask_price1', 'ask_size1', 'bid_price1', 'bid_size1',
                      'ask_price2', 'ask_size2', 'bid_price2', 'bid_size2', 'ask_price3',
                      'ask_size3', 'bid_price3', 'bid_size3', 'ask_price4', 'ask_size4',
                      'bid_price4', 'bid_size4']]
    for rolling in lags:

        df[f'buy_order_aggressive_1_{rolling}'] = np.where(
            (df['ask_price1'] < df['bid_price1'].shift(rolling)) & (df['ask_size1'] < df['bid_size1'].shift(rolling)), 1, 0)
        df[f'buy_order_aggressive_2_{rolling}'] = np.where(
            (df['ask_price1'] == df['bid_price1'].shift(rolling)) & (df['ask_size1'] < df['bid_size1'].shift(rolling)), 2, 0)
        df[f'sell_order_aggressive_1_{rolling}'] = np.where(
            (df['bid_price1'] > df['ask_price1'].shift(rolling)) & (df['bid_size1'] < df['ask_size1'].shift(rolling)), 1, 0)
        df[f'sell_order_aggressive_2_{rolling}'] = np.where(
            (df['bid_price1'] == df['ask_price1'].shift(rolling)) & (df['bid_size1'] < df['ask_size1'].shift(rolling)), 2, 0)

        df[f'buy_order_aggressive_3_{rolling}'] = np.where(
            (df['ask_price1'] < df['bid_price1'].shift(rolling)) & (df['ask_size1'] >= df['bid_size1'].shift(rolling)), 3, 0)
        df[f'sell_order_aggressive_3_{rolling}'] = np.where(
            (df['bid_price1'] > df['ask_price1'].shift(rolling)) & (df['bid_size1'] >= df['ask_size1'].shift(rolling)), 3, 0)

        df[f'buy_order_aggressive_4_{rolling}'] = np.where(
            (df['bid_price1'] < df['bid_price1'].shift(rolling)) & (df['ask_price1'] > df['bid_price1'].shift(rolling)), 4, 0)
        df[f'sell_order_aggressive_4_{rolling}'] = np.where(
            (df['bid_price1'] < df['ask_price1'].shift(rolling)) & (df['ask_price1'] > df['ask_price1'].shift(rolling)), 4, 0)

        df[f'buy_order_aggressive_5_{rolling}'] = np.where(
            df['bid_price1'] == df['bid_price1'].shift(rolling), 5, 0)
        df[f'sell_order_aggressive_5_{rolling}'] = np.where(
            df['ask_price1'] == df['ask_price1'].shift(rolling), 5, 0)

        df[f'buy_order_aggressive_6_{rolling}'] = np.where(
            df['bid_price1'] > df['bid_price1'].shift(rolling), 6, 0)
        df[f'sell_order_aggressive_6_{rolling}'] = np.where(
            df['ask_price1'] < df['ask_price1'].shift(rolling), 6, 0)

        df[f'buy_pct_1_{rolling}'] = df[f'buy_order_aggressive_1_{rolling}'].apply(lambda x: x == 1).rolling(rolling).sum() / \
                              df[f'buy_order_aggressive_1_{rolling}'].apply(lambda x: x == x).rolling(rolling).sum()
        df[f'buy_pct_2_{rolling}'] = df[f'buy_order_aggressive_2_{rolling}'].apply(lambda x: x == 2).rolling(rolling).sum() / \
                                df[f'buy_order_aggressive_2_{rolling}'].apply(lambda x: x == x).rolling(rolling).sum()
        df[f'buy_pct_3_{rolling}'] = df[f'buy_order_aggressive_3_{rolling}'].apply(lambda x: x == 3).rolling(rolling).sum() / \
                                    df[f'buy_order_aggressive_3_{rolling}'].apply(lambda x: x == x).rolling(rolling).sum()
        df[f'buy_pct_4_{rolling}'] = df[f'buy_order_aggressive_4_{rolling}'].apply(lambda x: x == 4).rolling(rolling).sum() / \
                                    df[f'buy_order_aggressive_4_{rolling}'].apply(lambda x: x == x).rolling(rolling).sum()
        df[f'buy_pct_5_{rolling}'] = df[f'buy_order_aggressive_5_{rolling}'].apply(lambda x: x == 6).rolling(rolling).sum() / \
                                    df[f'buy_order_aggressive_5_{rolling}'].apply(lambda x: x == x).rolling(rolling).sum()
        df[f'buy_pct_6_{rolling}'] = df[f'buy_order_aggressive_6_{rolling}'].apply(lambda x: x == 6).rolling(rolling).sum() / \
                                    df[f'buy_order_aggressive_6_{rolling}'].apply(lambda x: x == x).rolling(rolling).sum()
        df[f'sell_pct_1_{rolling}'] = df[f'sell_order_aggressive_1_{rolling}'].apply(lambda x: x == 1).rolling(rolling).sum() / \
                                    df[f'sell_order_aggressive_1_{rolling}'].apply(lambda x: x == x).rolling(rolling).sum()
        df[f'sell_pct_2_{rolling}'] = df[f'sell_order_aggressive_2_{rolling}'].apply(lambda x: x == 2).rolling(rolling).sum() / \
                                    df[f'sell_order_aggressive_2_{rolling}'].apply(lambda x: x == x).rolling(rolling).sum()
        df[f'sell_pct_3_{rolling}'] = df[f'sell_order_aggressive_3_{rolling}'].apply(lambda x: x == 3).rolling(rolling).sum() / \
                                    df[f'sell_order_aggressive_3_{rolling}'].apply(lambda x: x == x).rolling(rolling).sum()
        df[f'sell_pct_4_{rolling}'] = df[f'sell_order_aggressive_4_{rolling}'].apply(lambda x: x == 4).rolling(rolling).sum() / \
                                    df[f'sell_order_aggressive_4_{rolling}'].apply(lambda x: x == x).rolling(rolling).sum()
        df[f'sell_pct_5_{rolling}'] = df[f'sell_order_aggressive_5_{rolling}'].apply(lambda x: x == 6).rolling(rolling).sum() / \
                                    df[f'sell_order_aggressive_5_{rolling}'].apply(lambda x: x == x).rolling(rolling).sum()
        df[f'sell_pct_6_{rolling}'] = df[f'sell_order_aggressive_6_{rolling}'].apply(lambda x: x == 6).rolling(rolling).sum() / \
                                    df[f'sell_order_aggressive_6_{rolling}'].apply(lambda x: x == x).rolling(rolling).sum()
        # df = df.loc[:,['closetime',f'buy_pct_1_{rolling}', f'buy_pct_2_{rolling}', f'buy_pct_3_{rolling}', f'buy_pct_4_{rolling}', f'buy_pct_5_{rolling}',
        #            f'buy_pct_6_{rolling}', f'sell_pct_1_{rolling}', f'sell_pct_2_{rolling}', f'sell_pct_3_{rolling}', f'sell_pct_4_{rolling}',
        #            f'sell_pct_5_{rolling}', f'sell_pct_6_{rolling}']]
    return df
#%%
start = time.time()

depth_factor = depth_factor_process(depth, rolling=60)
# trade_factor = trade_factor_process(trade, rolling=10)
# add_factor = add_factor_process(depth=depth, trade=trade)
# aggre_factor = order_aggressiveness(depth, rolling=10)
end = time.time()
print('Total Time = %s' % (end - start))

