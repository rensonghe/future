import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
#%%  计算这一行基于bid和ask的wap
def calc_wap1(df):
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap

# Function to calculate second WAP
def calc_wap2(df):
    wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap

def calc_wap3(df):
    wap = (df['bid_price1'] * df['bid_size1'] + df['ask_price1'] * df['ask_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap

def calc_wap4(df):
    wap = (df['bid_price2'] * df['bid_size2'] + df['ask_price2'] * df['ask_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap

def calc_wap12(df):
    var1 = df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']
    var2 = df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']
    den = df['bid_size1'] + df['ask_size1'] + df['bid_size2'] + df['ask_size2']
    return (var1+var2) / den

def calc_wap34(df):
    var1 = df['bid_price1'] * df['bid_size1'] + df['ask_price1'] * df['ask_size1']
    var2 = df['bid_price2'] * df['bid_size2'] + df['ask_price2'] * df['ask_size2']
    den = df['bid_size1'] + df['ask_size1'] + df['bid_size2'] + df['ask_size2']
    return (var1+var2) / den

def calc_swap1(df):
    return df['wap1'] - df['wap3']

def calc_swap12(df):
    return df['wap12'] - df['wap34']

def calc_tswap1(df):
    return -df['swap1'].diff()

def calc_tswap12(df):
    return -df['swap12'].diff()

def calc_wss12(df):
    ask = (df['ask_price1'] * df['ask_size1'] + df['ask_price2'] * df['ask_size2'])/(df['ask_size1']+df['ask_size2'])
    bid = (df['bid_price1'] * df['bid_size1'] + df['bid_price2'] * df['bid_size2'])/(df['bid_size1']+df['bid_size2'])
    mid = (df['ask_price1'] + df['bid_price1']) / 2
    return (ask - bid) / mid

def calc_tt1(df):
    p1 = df['ask_price1'] * df['ask_size1'] + df['bid_price1'] * df['bid_size1']
    p2 = df['ask_price2'] * df['ask_size2'] + df['bid_price2'] * df['bid_size2']
    return p2 - p1

def calc_price_impact(df):
    ask = (df['ask_price1'] * df['ask_size1'] + df['ask_price2'] * df['ask_size2'])/(df['ask_size1']+df['ask_size2'])
    bid = (df['bid_price1'] * df['bid_size1'] + df['bid_price2'] * df['bid_size2'])/(df['bid_size1']+df['bid_size2'])
    return (df['ask_price1'] - ask)/df['ask_price1'], (df['bid_price1'] - bid)/df['bid_price1']

# Calculate order book slope
def calc_slope(df):
    v0 = (df['bid_size1']+df['ask_size1'])/2
    p0 = (df['bid_price1']+df['ask_price1'])/2
    slope_bid = ((df['bid_size1']/v0)-1)/abs((df['bid_price1']/p0)-1)+(
                (df['bid_size2']/df['bid_size1'])-1)/abs((df['bid_price2']/df['bid_price1'])-1)
    slope_ask = ((df['ask_size1']/v0)-1)/abs((df['ask_price1']/p0)-1)+(
                (df['ask_size2']/df['ask_size1'])-1)/abs((df['ask_price2']/df['ask_price1'])-1)
    return (slope_bid+slope_ask)/2, abs(slope_bid-slope_ask)

# Calculate order book dispersion
def calc_dispersion(df):
    bspread = df['bid_price1'] - df['bid_price2']
    aspread = df['ask_price2'] - df['ask_price1']
    bmid = (df['bid_price1'] + df['ask_price1'])/2  - df['bid_price1']
    bmid2 = (df['bid_price1'] + df['ask_price1'])/2  - df['bid_price2']
    amid = df['ask_price1'] - (df['bid_price1'] + df['ask_price1'])/2
    amid2 = df['ask_price2'] - (df['bid_price1'] + df['ask_price1'])/2
    bdisp = (df['bid_size1']*bmid + df['bid_size2']*bspread)/(df['bid_size1']+df['bid_size2'])
    bdisp2 = (df['bid_size1']*bmid + df['bid_size2']*bmid2)/(df['bid_size1']+df['bid_size2'])
    adisp = (df['ask_size1']*amid + df['ask_size2']*aspread)/(df['ask_size1']+df['ask_size2'])
    adisp2 = (df['ask_size1']*amid + df['ask_size2']*amid2)/(df['ask_size1']+df['ask_size2'])
    return bspread, aspread, bmid, amid, bdisp, adisp, (bdisp + adisp)/2, (bdisp2 + adisp2)/2

# Calculate order book depth
def calc_depth(df):
    depth = df['bid_price1'] * df['bid_size1'] + df['ask_price1'] * df['ask_size1'] + df['bid_price2'] * df[
               'bid_size2'] + df['ask_price2'] * df['ask_size2']
    return depth

#  order flow imbalance
def calc_ofi(df):
    a = df['bid_size1']*np.where(df['bid_price1'].diff()>=0,1,0)
    b = df['bid_size1'].shift()*np.where(df['bid_price1'].diff()<=0,1,0)
    c = df['ask_size1']*np.where(df['ask_price1'].diff()<=0,1,0)
    d = df['ask_size1'].shift()*np.where(df['ask_price1'].diff()>=0,1,0)
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
    return np.sqrt(1/series)*100000

def square_root_translation(series):
    return series**(1/2)

# Calculate the realized absolute variation
def realized_absvar(series):
    return np.sqrt(np.pi/(2*series.count()))*np.sum(np.abs(series))

# Calculate the realized skew
def realized_skew(series):
    return np.sqrt(series.count())*np.sum(series**3)/(realized_volatility(series)**3)

# Calculate the realized kurtosis
def realized_kurtosis(series):
    return series.count()*np.sum(series**4)/(realized_volatility(series)**4)

def get_age(prices):
    last_value = prices[-1]
    age = 0
    for i in range(2, len(prices)):
        if prices[-i] != last_value:
            return age
        age += 1
    return age

def bid_age(depth, rolling=10):
    bp1 = depth['bid_price1']
    bp1_changes = bp1.rolling(rolling).apply(get_age, engine='numba', raw=True).fillna(0)
    return bp1_changes

def ask_age(depth, rolling=10):
    ap1 = depth['ask_price1']
    ap1_changes = ap1.rolling(rolling).apply(get_age, engine='numba', raw=True).fillna(0)
    return ap1_changes

def inf_ratio(depth, trade, rolling=100):
    quasi = trade.price.diff().abs().rolling(rolling).sum().fillna(10)
    dif = trade.price.diff(rolling).abs().fillna(10)
    return quasi / (dif + quasi)

def depth_price_range(depth, trade, rolling=100):
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
    return pd.Series(-(depth['ask_price1'] - ask) / depth['ask_price1'] - (depth['bid_price1'] - bid) / depth['bid_price1'], name="price_impact")

def depth_price_skew(depth, trade):
    prices = ["bid_price5", "bid_price4", "bid_price3", "bid_price2", "bid_price1", "ask_price1", "ask_price2", "ask_price3", "ask_price4", "ask_price5"]
    return depth[prices].skew(axis=1)

def depth_price_kurt(depth, trade):
    prices = ["bid_price5", "bid_price4", "bid_price3", "bid_price2", "bid_price1", "ask_price1", "ask_price2", "ask_price3", "ask_price4", "ask_price5"]
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

def first_location_of_maximum(x):
    max_value = max(x)  # 一个for 循环
    for loc in range(len(x)):
        if x[loc] == max_value:
            return loc + 1

def price_idxmax(depth, trade, rolling=20):
    return depth['ask_price1'].rolling(rolling).apply(first_location_of_maximum, engine='numba', raw=True).fillna(0)

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

def arrive_rate_2(depth, trade, rolling=100):
    return (trade.closetime.shift(rolling) - trade.closetime).fillna(0)

def avg_trade_volume(depth, trade, rolling=100):
    return (trade['size'][::-1].abs().rolling(rolling).sum().shift(-rolling + 1)).fillna(0)[::-1]

def avg_spread(depth, trade, rolling=200):
    return (depth.ask_price1 - depth.bid_price1).rolling(rolling).mean().fillna(0)

def avg_turnover(depth, trade, rolling=500):
    return depth[['ask_size1', 'ask_size2', 'ask_size3', 'ask_size4','ask_size5','bid_size1', 'bid_size2', 'bid_size3', 'bid_size4', 'bid_size5']].sum(axis=1)

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
        return trade['size'].copy()

    v = volume(depth=depth, trade=trade, rolling=rolling)
    v[v > 0] = 0
    return (v.rolling(rolling).sum() / bvs).fillna(0)

def av_divide_tn(depth, trade, rolling=10):
    avs = depth.ask_size1 + depth.ask_size2 + depth.ask_size3 + depth.ask_size4 + depth.ask_size5

    def volume(depth, trade, n):
        return trade['size'].copy()

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

def _bid_withdraws_volume(l, n, levels=5):
    withdraws = 0
    for price_index in range(2, 2 + 4 * levels, 4):
        now_p = n[price_index]
        for price_last_index in range(2, 2 + 4 * levels, 4):
            if l[price_last_index] == now_p:
                withdraws -= min(n[price_index + 1] - l[price_last_index + 1], 0)

    return withdraws

def _ask_withdraws_volume(l, n, levels=5):
    withdraws = 0
    for price_index in range(0, 4 * levels, 4):
        now_p = n[price_index]
        for price_last_index in range(0, 4 * levels, 4):
            if l[price_last_index] == now_p:
                withdraws -= min(n[price_index + 1] - l[price_last_index + 1], 0)

    return withdraws

def ask_withdraws(depth, trade):
    ob_values = depth.iloc[:,1:].values
    flows = np.zeros(len(ob_values))
    for i in range(1, len(ob_values)):
        flows[i] = _ask_withdraws_volume(ob_values[i - 1], ob_values[i])
    return flows

def bid_withdraws(depth, trade):
    ob_values = depth.iloc[:,1:].values
    flows = np.zeros(len(ob_values))
    for i in range(1, len(ob_values)):
        flows[i] = _bid_withdraws_volume(ob_values[i - 1], ob_values[i])
    return flows

def z_t(trade, depth):
    """初探市场微观结构：指令单薄与指令单流——资金交易策略之四 成交价的对数减去中间价的对数"""
    # data_dic = self.data_dic  # 调用的是属性
    tick_fac_data = np.log(trade['price']) - np.log((depth['bid_price1'] + depth['ask_price1']) / 2)
    return tick_fac_data

def voi(depth,trade):
    """voi订单失衡 Volume Order Imbalance20200709-中信建投-因子深度研究系列：高频量价选股因子初探"""
    # data_dic = self.data_dic
    bid_sub_price = depth['bid_price1'] - depth['bid_price1'].shift(1)
    ask_sub_price = depth['ask_price1'] - depth['ask_price1'].shift(1)

    bid_sub_volume = depth['bid_size1'] - depth['bid_size1'].shift(1)
    ask_sub_volume = depth['ask_size1'] - depth['ask_size1'].shift(1)
    bid_volume_change = bid_sub_volume
    ask_volume_change = ask_sub_volume
    # bid_volume_change[bid_sub_price == 0] = bid_sub_volume[bid_sub_price == 0]
    bid_volume_change[bid_sub_price < 0] = 0
    bid_volume_change[bid_sub_price > 0] = depth['bid_size1'][bid_sub_price > 0]
    ask_volume_change[ask_sub_price < 0] = depth['ask_size1'][ask_sub_price < 0]
    ask_volume_change[ask_sub_price > 0] = 0
    tick_fac_data = (bid_volume_change - ask_volume_change) / trade['volume']
    return tick_fac_data

def cal_weight_volume(depth):
    """计算加权的盘口挂单量"""
    # data_dic = self.data_dic
    w = [1 - (i - 1) / 5 for i in range(1, 6)]
    w = np.array(w) / sum(w)
    wb = depth['bid_size1'] * w[0] + depth['bid_size2'] * w[1] + depth['bid_size3'] * w[2] + depth['bid_size4'] * w[3] + depth['bid_size5'] * w[4]
    wa = depth['ask_size1'] * w[0] + depth['ask_size2'] * w[1] + depth['ask_size3'] * w[2] + depth['ask_size4'] * w[3] + depth['ask_size5'] * w[4]
    return wb, wa

def oir(depth, trade):
    wb, wa = cal_weight_volume(depth)
    ori = (wb-wa)/(wa+wb)
    return ori

def voi2(depth, trade):
    """同voi，衰减加权，"""
    # data_dic = self.data_dic
    wb, wa = cal_weight_volume(depth)
    bid_sub_price = depth['bid_price1'] - depth['bid_price1'].shift(1)
    ask_sub_price = depth['ask_price1'] - depth['ask_price1'].shift(1)

    bid_sub_volume = wb - wb.shift(1)
    ask_sub_volume = wa - wa.shift(1)
    bid_volume_change = bid_sub_volume
    ask_volume_change = ask_sub_volume
    # bid_volume_change[bid_sub_price == 0] = bid_sub_volume[bid_sub_price == 0]
    bid_volume_change[bid_sub_price < 0] = 0
    bid_volume_change[bid_sub_price > 0] = wb[bid_sub_price > 0]
    ask_volume_change[ask_sub_price < 0] = wa[ask_sub_price < 0]
    ask_volume_change[ask_sub_price > 0] = 0
    tick_fac_data = (bid_volume_change - ask_volume_change) / trade['volume']  # 自动行列对齐
    return tick_fac_data

def voi2_level2(depth, trade):
    """同voi，衰减加权，"""
    # data_dic = self.data_dic
    wb, wa = cal_weight_volume(depth)
    bid_sub_price = depth['bid_price2'] - depth['bid_price2'].shift(1)
    ask_sub_price = depth['ask_price2'] - depth['ask_price2'].shift(1)

    bid_sub_volume = wb - wb.shift(1)
    ask_sub_volume = wa - wa.shift(1)
    bid_volume_change = bid_sub_volume
    ask_volume_change = ask_sub_volume
    # bid_volume_change[bid_sub_price == 0] = bid_sub_volume[bid_sub_price == 0]
    bid_volume_change[bid_sub_price < 0] = 0
    bid_volume_change[bid_sub_price > 0] = wb[bid_sub_price > 0]
    ask_volume_change[ask_sub_price < 0] = wa[ask_sub_price < 0]
    ask_volume_change[ask_sub_price > 0] = 0
    tick_fac_data = (bid_volume_change - ask_volume_change) / trade['volume']  # 自动行列对齐
    return tick_fac_data

def voi2_level3(depth, trade):
    """同voi，衰减加权，"""
    # data_dic = self.data_dic
    wb, wa = cal_weight_volume(depth)
    bid_sub_price = depth['bid_price3'] - depth['bid_price3'].shift(1)
    ask_sub_price = depth['ask_price3'] - depth['ask_price3'].shift(1)

    bid_sub_volume = wb - wb.shift(1)
    ask_sub_volume = wa - wa.shift(1)
    bid_volume_change = bid_sub_volume
    ask_volume_change = ask_sub_volume
    # bid_volume_change[bid_sub_price == 0] = bid_sub_volume[bid_sub_price == 0]
    bid_volume_change[bid_sub_price < 0] = 0
    bid_volume_change[bid_sub_price > 0] = wb[bid_sub_price > 0]
    ask_volume_change[ask_sub_price < 0] = wa[ask_sub_price < 0]
    ask_volume_change[ask_sub_price > 0] = 0
    tick_fac_data = (bid_volume_change - ask_volume_change) / trade['volume']  # 自动行列对齐
    return tick_fac_data

def voi2_level4(depth, trade):
    """同voi，衰减加权，"""
    # data_dic = self.data_dic
    wb, wa = cal_weight_volume(depth)
    bid_sub_price = depth['bid_price4'] - depth['bid_price4'].shift(1)
    ask_sub_price = depth['ask_price4'] - depth['ask_price4'].shift(1)

    bid_sub_volume = wb - wb.shift(1)
    ask_sub_volume = wa - wa.shift(1)
    bid_volume_change = bid_sub_volume
    ask_volume_change = ask_sub_volume
    # bid_volume_change[bid_sub_price == 0] = bid_sub_volume[bid_sub_price == 0]
    bid_volume_change[bid_sub_price < 0] = 0
    bid_volume_change[bid_sub_price > 0] = wb[bid_sub_price > 0]
    ask_volume_change[ask_sub_price < 0] = wa[ask_sub_price < 0]
    ask_volume_change[ask_sub_price > 0] = 0
    tick_fac_data = (bid_volume_change - ask_volume_change) / trade['volume']  # 自动行列对齐
    return tick_fac_data

def voi2_level5(depth, trade):
    """同voi，衰减加权，"""
    # data_dic = self.data_dic
    wb, wa = cal_weight_volume(depth)
    bid_sub_price = depth['bid_price5'] - depth['bid_price5'].shift(1)
    ask_sub_price = depth['ask_price5'] - depth['ask_price5'].shift(1)

    bid_sub_volume = wb - wb.shift(1)
    ask_sub_volume = wa - wa.shift(1)
    bid_volume_change = bid_sub_volume
    ask_volume_change = ask_sub_volume
    # bid_volume_change[bid_sub_price == 0] = bid_sub_volume[bid_sub_price == 0]
    bid_volume_change[bid_sub_price < 0] = 0
    bid_volume_change[bid_sub_price > 0] = wb[bid_sub_price > 0]
    ask_volume_change[ask_sub_price < 0] = wa[ask_sub_price < 0]
    ask_volume_change[ask_sub_price > 0] = 0
    tick_fac_data = (bid_volume_change - ask_volume_change) / trade['volume']  # 自动行列对齐
    return tick_fac_data

def mpb(depth, trade):
    """市价偏离度 Mid-Price Basis 中信建投-因子深度研究系列：高频量价选股因子初探"""
    # data_dic = self.data_dic
    tp = trade['amount'] / trade['volume']  # 注意单位，
    # print(tp)
    tp[np.isinf(tp)] = np.nan
    tp.fillna(method='ffill', inplace=True)
    mid = (depth['bid_price1'] + depth['ask_price1']) / 2
    tick_fac_data = tp - (mid + mid.shift(1)) / 1000 / 2
    return tick_fac_data

def mpb_5min(depth, trade, rolling=120*5):
    """市价偏离度 Mid-Price Basis 中信建投-因子深度研究系列：高频量价选股因子初探"""
    # data_dic = self.data_dic
    tp = trade['amount'] / trade['volume']  # 注意单位，
    # print(tp)
    tp[np.isinf(tp)] = np.nan
    tp.fillna(method='ffill', inplace=True)
    mid = (depth['bid_price1'] + depth['ask_price1']) / 2
    tick_fac_data = tp - (mid + mid.shift(rolling)) / 1000 / 2
    return tick_fac_data

def mpc(depth, trade, rolling=120*5):
    mid = (depth['ask_price1']+depth['bid_price1'])/2
    mpc = (mid-mid.shift(rolling))/mid.shift(rolling)
    return mpc


def slope(depth):
    """斜率 价差/深度"""
    # data_dic = self.data_dic
    tick_fac_data = (depth['ask_price1'] - depth['bid_price1']) / (depth['ask_size1'] + depth['bid_size1']) * 2
    return tick_fac_data

def positive_ratio(depth, trade,rolling=20 * 3):
    """积极买入成交额占总成交额的比例"""
    # data_dic = self.data_dic
    buy_positive = pd.DataFrame(0, columns=['amount'], index=trade['amount'].index)
    buy_positive['amount'] = trade['amount']
    # buy_positive[trade['price'] >= depth['ask_price1'].shift(1)] = trade['turnover'][trade['price'] >= depth['ask_price1'].shift(1)]
    buy_positive['amount'] = np.where(trade['price']>=depth['ask_price1'].shift(1), buy_positive['amount'], 0)
    tick_fac_data = buy_positive['amount'].rolling(rolling).sum() / trade['amount'].rolling(rolling).sum()
    return tick_fac_data

def positive_buying(depth, trade, rolling = 60):
    '''
    买入情绪因子：根据积极买入和保守买入
    '''
    positive_buy = np.where(trade['price']>=depth['ask_price1'].shift(1), trade['volume'], 0)
    caustious_buy = np.where(trade['price']<=depth['bid_price1'].shift(1), trade['volume'], 0)
    bm = pd.Series(positive_buy, index=trade.index).rolling(rolling).sum()/pd.Series(caustious_buy, index=trade.index).rolling(rolling).sum()
    return bm
def positive_selling(depth, trade, rolling = 60):
    '''
    卖出情绪因子：根据积极卖出和保守卖出
    '''
    positive_sell = np.where(trade['price']<=depth['bid_price1'].shift(1), trade['volume'], 0)
    caustious_sell = np.where(trade['price']>=depth['ask_price1'].shift(1), trade['volume'], 0)
    sm = pd.Series(positive_sell, index=trade.index).rolling(rolling).sum()/pd.Series(caustious_sell, index=trade.index).rolling(rolling).sum()
    return sm

def buying_amplification_ratio(depth, trade, rolling):
    '''

    :param depth:
    :param trade:
    :param rolling:
    :return:
    '''
    biding = depth['bid_size1']*depth['bid_price1'] + depth['bid_size2']*depth['bid_price2'] + depth['bid_size3']*depth['bid_price3'] + depth['bid_size4']*depth['bid_price4'] + depth['bid_size5']*depth['bid_price5']
    asking = depth['ask_size1']*depth['ask_price1'] + depth['ask_size2']*depth['ask_price2'] + depth['ask_size3']*depth['ask_price3'] + depth['ask_size4']*depth['ask_price4'] + depth['ask_size5']*depth['ask_price5']
    amplify_biding = np.where(biding>biding.shift(1), biding-biding.shift(1),0)
    amplify_asking = np.where(asking>asking.shift(1), asking-asking.shift(1),0)
    diff = amplify_biding - amplify_asking
    buying_ratio = (pd.Series(diff, index=trade.index).rolling(rolling).sum())/(trade['amount'])/rolling
    return buying_ratio

def buying_amount_ratio(depth, trade, rolling):
    positive_buy = np.where(trade['price']>=depth['ask_price1'].shift(1), trade['amount'], 0)
    positive_sell = np.where(trade['price']<=depth['bid_price1'].shift(1), trade['amount'], 0)
    diff = positive_buy - positive_sell
    buying_amount_ratio = ((pd.Series(diff, index=trade.index).rolling(rolling).sum())/(trade['amount'].rolling(rolling).sum()))/rolling
    return buying_amount_ratio

def buying_willing(depth, trade, rolling):
    biding = depth['bid_size1'] * depth['bid_price1'] + depth['bid_size2'] * depth['bid_price2'] + depth['bid_size3'] * \
             depth['bid_price3'] + depth['bid_size4'] * depth['bid_price4'] + depth['bid_size5'] * depth['bid_price5']
    asking = depth['ask_size1'] * depth['ask_price1'] + depth['ask_size2'] * depth['ask_price2'] + depth['ask_size3'] * \
             depth['ask_price3'] + depth['ask_size4'] * depth['ask_price4'] + depth['ask_size5'] * depth['ask_price5']
    amplify_biding = np.where(biding > biding.shift(1), biding - biding.shift(1), 0)
    amplify_asking = np.where(asking > asking.shift(1), asking - asking.shift(1), 0)
    positive_buy = np.where(trade['price'] >= depth['ask_price1'].shift(1), trade['amount'], 0)
    positive_sell = np.where(trade['price'] <= depth['bid_price1'].shift(1), trade['amount'], 0)
    diff = (amplify_biding - amplify_asking) + (positive_buy - positive_sell)
    buying_willing = pd.Series((pd.Series(diff, index=trade.index).rolling(rolling).sum())/trade['amount'].rolling(rolling).sum())/rolling
    return buying_willing

def buying_willing_stength(depth, trade, rolling):
    biding = (depth['bid_size1'] + depth['bid_size2'] + depth['bid_size3'] + depth['bid_size4'] + depth['bid_size5'])
    asking = (depth['ask_size1'] + depth['ask_size2'] + depth['ask_size3'] + depth['ask_size4'] + depth['ask_size5'])
    positive_buy = np.where(trade['price'] >= depth['ask_price1'].shift(1), trade['amount'], 0)
    positive_sell = np.where(trade['price'] <= depth['bid_price1'].shift(1), trade['amount'], 0)
    diff = (biding - asking) + (positive_buy - positive_sell)
    buying_stength = pd.Series((pd.Series(diff, index=trade.index).rolling(rolling).mean())/(pd.Series(diff, index=trade.index).rolling(rolling).std())).rolling(rolling).std()/rolling
    return buying_stength

def buying_amount_strength(depth, trade, rolling):
    positive_buy = np.where(trade['price']>=depth['ask_price1'].shift(1), trade['amount'], 0)
    positive_sell = np.where(trade['price'] <= depth['bid_price1'].shift(1), trade['amount'], 0)
    diff = positive_buy - positive_sell
    buying_amount_strength = (pd.Series(((pd.Series(diff, index=trade.index).rolling(rolling).mean())/(pd.Series(diff, index=trade.index).rolling(rolling).std()))).rolling(rolling).std())/rolling
    return buying_amount_strength

def selling_ratio(depth, trade, rolling):
    biding = depth['bid_size1'] * depth['bid_price1'] + depth['bid_size2'] * depth['bid_price2'] + depth['bid_size3'] * \
             depth['bid_price3'] + depth['bid_size4'] * depth['bid_price4'] + depth['bid_size5'] * depth['bid_price5']
    asking = depth['ask_size1'] * depth['ask_price1'] + depth['ask_size2'] * depth['ask_price2'] + depth['ask_size3'] * \
             depth['ask_price3'] + depth['ask_size4'] * depth['ask_price4'] + depth['ask_size5'] * depth['ask_price5']
    amplify_biding = np.where(biding > biding.shift(1), biding - biding.shift(1), 0)
    amplify_asking = np.where(asking > asking.shift(1), asking - asking.shift(1), 0)
    diff = amplify_asking - amplify_biding
    # amount = trade['amount'].copy().reset_index(drop=True)
    selling_ratio = (pd.Series(diff, index=trade.index).rolling(rolling).sum())/(trade['amount'])/rolling
    return selling_ratio

def large_order_ratio(depth, trade, rolling=120*2):
    mean = (trade['volume'] - trade['volume'].shift(rolling)).rolling(rolling).mean()
    std = (trade['volume'] - trade['volume'].shift(rolling)).rolling(rolling).std()
    large = np.where(np.abs(trade['size'])>(mean+std),trade['amount'],0)
    # amount = trade['amount'].copy().reset_index(drop=True)
    ratio = large/trade['amount']
    large_order_ratio = (pd.Series(ratio, index=trade.index).rolling(rolling).sum())/rolling
    return large_order_ratio

def buy_order_aggressivenes_level1(depth, trade, rolling=1000):
    '''
    买单订单侵略性因子 aggressive level1
    '''
    v = trade['size'].copy()
    p = trade['price'].copy()
    biding = depth['bid_size1'] * depth['bid_price1'] + depth['bid_size2'] * depth['bid_price2'] + depth['bid_size3'] * \
             depth['bid_price3'] + depth['bid_size4'] * depth['bid_price4'] + depth['bid_size5'] * depth['bid_price5']
    # 买家激进程度
    p[v<0] = 0
    mid = (depth['ask_price1']+depth['bid_price1'])/2
    buy_price = np.where((p>=depth['ask_price1'].shift(1))&(v>=depth['ask_size1'].shift(1)), p,0)
    amount = np.where((p>=depth['ask_price1'].shift(1))&(v>=depth['ask_size1'].shift(1)), trade['amount']-trade['amount'].shift(1),np.nan)
    buy_amount_agg_ratio = biding.rolling(rolling).sum()/amount
    buy_price_bias =abs(buy_price-mid.shift(rolling))/mid.shift(rolling)
    return buy_price_bias, buy_amount_agg_ratio

def buy_order_aggressivenes_level2(depth, trade, rolling=1000):
    v = trade['size'].copy()
    p = trade['price'].copy()
    biding = depth['bid_size1'] * depth['bid_price1'] + depth['bid_size2'] * depth['bid_price2'] + depth['bid_size3'] * \
             depth['bid_price3'] + depth['bid_size4'] * depth['bid_price4'] + depth['bid_size5'] * depth['bid_price5']
    # 买家激进程度
    p[v<0] = 0
    mid = (depth['ask_price1']+depth['bid_price1'])/2
    buy_price = np.where((p>=depth['ask_price1'].shift(1))&(v<=depth['ask_size1'].shift(1)), p,0)
    amount = np.where((p>=depth['ask_price1'].shift(1))&(v<=depth['ask_size1'].shift(1)), trade['amount']-trade['amount'].shift(1),np.nan)
    buy_amount_agg_ratio = biding.rolling(rolling).sum()/amount
    buy_price_bias =abs(buy_price-mid.shift(rolling))/mid.shift(rolling)
    return buy_price_bias, buy_amount_agg_ratio

def sell_order_aggressivenes_level1(depth, trade, rolling=1000):
    v = trade['size'].copy()
    p = trade['price'].copy()
    asking = depth['ask_size1'] * depth['ask_price1'] + depth['ask_size2'] * depth['ask_price2'] + depth['ask_size3'] * \
             depth['ask_price3'] + depth['ask_size4'] * depth['ask_price4'] + depth['ask_size5'] * depth['ask_price5']
    # 卖家激进程度
    p[v>0] = 0
    mid = (depth['ask_price1']+depth['bid_price1'])/2
    sell_price = np.where((p<=depth['bid_price1'].shift(1))&(abs(v)>=depth['bid_size1'].shift(1)), p,0)
    amount = np.where((p<=depth['bid_price1'].shift(1))&(abs(v)>=depth['bid_size1'].shift(1)), trade['amount']-trade['amount'].shift(1),np.nan)
    sell_amount_agg_ratio = asking.rolling(rolling).sum()/amount
    sell_price_bias = abs(sell_price-mid.shift(rolling))/mid.shift(rolling)
    return sell_price_bias, sell_amount_agg_ratio

def sell_order_aggressivenes_level2(depth, trade, rolling=1000):
    v = trade['size'].copy()
    p = trade['price'].copy()
    asking = depth['ask_size1'] * depth['ask_price1'] + depth['ask_size2'] * depth['ask_price2'] + depth['ask_size3'] * \
             depth['ask_price3'] + depth['ask_size4'] * depth['ask_price4'] + depth['ask_size5'] * depth['ask_price5']
    # 卖家激进程度
    p[v>0] = 0
    mid = (depth['ask_price1']+depth['bid_price1'])/2
    sell_price = np.where((p<=depth['bid_price1'].shift(1))&(abs(v)<=depth['bid_size1'].shift(1)), p,0)
    amount = np.where((p<=depth['bid_price1'].shift(1))&(abs(v)<=depth['bid_size1'].shift(1)), trade['amount']-trade['amount'].shift(1),np.nan)
    sell_amount_agg_ratio = asking.rolling(rolling).sum()/amount
    sell_price_bias = abs(sell_price-mid.shift(rolling))/mid.shift(rolling)
    return sell_price_bias, sell_amount_agg_ratio

def length_imbalance(depth,trade, level=5):
    _ = np.arange(1, level + 1)

    imb = {s: (depth["bid_size%s" % s] - depth["ask_size%s" % s]) / (depth["bid_size%s" % s] + depth["ask_size%s" % s]) for s in _}

    return pd.concat(imb.values(), keys=imb.keys()).unstack().T

def height_imbalance(depth,trade, level=5):
    _ = np.arange(2, level + 1)

    bid_height = [(depth['bid_price%s' % (i - 1)] - depth['bid_price%s' % i]) for i in _]
    ask_height = [(depth['ask_price%s' % i] - depth['ask_price%s' % (i - 1)]) for i in _]

    r = {i + 2: (b - ask_height[i]) / (b + ask_height[i]) for i, b in enumerate(bid_height)}

    r = pd.concat(r.values(), keys=r.keys()).unstack().T

    return r
#%%
def corr_pv(depth, trade, rolling=120):
    '''
    高频量价相关性
    '''
    pv_ic = trade['price'].rolling(rolling).corr((trade['volume']-trade['volume'].shift(rolling))/trade['volume'])
    oi_ic = trade['price'].rolling(rolling).corr(trade['open_interest'])
    return pv_ic, oi_ic

def flowInRatio(depth, trade, rolling=120):
    flowInRatio = trade['volume']*trade['price']*((trade['price']-trade['price'].shift(1))/abs(trade['price']-trade['price'].shift(rolling)))/trade['amount']
    # flowInRatio2 = (trade['open_interest']-trade['open_interest'].shift(1))*trade['price']*((trade['price']-trade['price'].shift(1))/abs(trade['price']-trade['price'].shift(1)))
    return flowInRatio

def large_order(depth, trade, rolling=120*10):
    '''
    大单买入卖出因子
    '''
    buy = np.where(trade['size']>0, trade['amount']-trade['amount'].shift(1),0)
    sell = np.where(trade['size']<0, trade['amount']-trade['amount'].shift(1),0)
    large_buy = np.where(pd.Series(buy, index=trade.index) > pd.Series(buy, index=trade.index).rolling(rolling).quantile(0.95),pd.Series(buy, index=trade.index),0)
    large_sell = np.where(pd.Series(sell, index=trade.index) > pd.Series(sell, index=trade.index).rolling(rolling).quantile(0.95),pd.Series(sell, index=trade.index),0)
    large_buy_ratio = pd.Series(large_buy, index=trade.index).rolling(rolling).sum()/(pd.Series(buy, index=trade.index).rolling(rolling).sum()+pd.Series(sell, index=trade.index).rolling(rolling).sum())
    large_sell_ratio = pd.Series(large_sell, index=trade.index).rolling(rolling).sum() / (
                pd.Series(buy, index=trade.index).rolling(rolling).sum() + pd.Series(sell, index=trade.index).rolling(rolling).sum())
    return large_sell_ratio,large_buy_ratio

# def rwR(depth, trade, rolling=120):
#     o, h, l,c  = trade['price'].resample(rolling).agg({''})
#%%
def price_weighted_pressure(depth, kws):
    n1 = kws.setdefault("n1", 1)
    n2 = kws.setdefault("n2", 5)

    bench = kws.setdefault("bench_type","MID")

    _ = np.arange(n1, n2 + 1)

    if bench == "MID":
        bench_prices = depth['ask_price1']+depth['bid_price1']
    elif bench == "SPECIFIC":
        bench_prices = kws.get("bench_price")
    else:
        raise Exception("")

    def unit_calc(bench_price):
        """比结算价高的价单立马成交，权重=0"""

        bid_d = [bench_price / (bench_price - depth["bid_price%s" % s]) for s in _]
        # bid_d = [_.replace(np.inf,0) for _ in bid_d]
        bid_denominator = sum(bid_d)

        bid_weights = [(d / bid_denominator).replace(np.nan,1) for d in bid_d]

        press_buy = sum([depth["bid_size%s" % (i + 1)] * w for i, w in enumerate(bid_weights)])

        ask_d = [bench_price / (depth['ask_price%s' % s] - bench_price) for s in _]
        # ask_d = [_.replace(np.inf,0) for _ in ask_d]
        ask_denominator = sum(ask_d)

        ask_weights = [d / ask_denominator for d in ask_d]

        press_sell = sum([depth['ask_size%s' % (i + 1)] * w for i, w in enumerate(ask_weights)])

        return (np.log(press_buy) - np.log(press_sell)).replace([-np.inf, np.inf], np.nan)

    return unit_calc(bench_prices)

def volume_order_imbalance(depth, kws):

    """
    Reference From <Order imbalance Based Strategy in High Frequency Trading>
    :param data:
    :param kws:
    :return:
    """
    drop_first = kws.setdefault("drop_first", True)

    current_bid_price = depth['bid_price1']

    bid_price_diff = current_bid_price - current_bid_price.shift()

    current_bid_vol = depth['bid_size1']

    nan_ = current_bid_vol[current_bid_vol == 0].index

    bvol_diff = current_bid_vol - current_bid_vol.shift()

    bid_increment = np.where(bid_price_diff > 0, current_bid_vol,
                             np.where(bid_price_diff < 0, 0, np.where(bid_price_diff == 0, bvol_diff, bid_price_diff)))

    current_ask_price = depth['ask_price1']

    ask_price_diff = current_ask_price - current_ask_price.shift()

    current_ask_vol = depth['ask_size1']

    avol_diff = current_ask_vol - current_ask_vol.shift()

    ask_increment = np.where(ask_price_diff < 0, current_ask_vol,
                             np.where(ask_price_diff > 0, 0, np.where(ask_price_diff == 0, avol_diff, ask_price_diff)))

    _ = pd.Series(bid_increment - ask_increment, index=depth.index)

    if drop_first:
        _.loc[_.groupby(_.index.date).apply(lambda x: x.index[0])] = np.nan

    _.loc[nan_] = np.nan

    return _

def get_mid_price_change(depth, drop_first=True):
    mid = (depth['ask_price1']+depth['bid_price1'])/2
    _ = mid.pct_change()
    if drop_first:
        _.loc[_.groupby(_.index.date).apply(lambda x: x.index[0])] = np.nan
    return _

def Open_Close_Percentage(trade, depth, rolling=120*5):
    openInterestChg = trade['open_interest'] - trade['open_interest'].shift(1)
    volumeChg = trade['volume'] - trade['volume'].shift(1)
    closeContract = (volumeChg - openInterestChg)/2
    openContract = volumeChg - closeContract

    return openContract.rolling(rolling).sum()/closeContract.rolling(rolling).sum()

def Open_Interest_Change(trade, depth, rolling=120*5):
    return trade['open_interest']/trade['open_interest'].shift(rolling)

def Order_Type1(l):
    openInterestChg = l['open_interest'] - l['open_interest'].shift(1)
    priceChg = l['price'] - l['price'].shift(1)
    return np.where((openInterestChg>0)&(priceChg>0), 1, 0)

def Order_Type1_large(l):
    openInterestChg = l['open_interest'] - l['open_interest'].shift(1)
    priceChg = l['price'] - l['price'].shift(1)
    size = l['volume'] - l['volume'].shift(1)
    return np.where((openInterestChg>0)&(priceChg>0)&(abs(l.size)>10), 1, 0)

def Order_Type1_Pct(trade, depth, rolling=120*5):
    type = Order_Type1(trade)
    pct = pd.Series(type).apply(lambda x:x ==1).rolling(rolling).sum()/pd.Series(type).apply(lambda x:x ==x).rolling(rolling).sum()
    return pct.fillna(0)

def Order_Type1_large_Pct(trade, depth, rolling=120*5):
    type = Order_Type1_large(trade)
    pct = pd.Series(type).apply(lambda x:x ==1).rolling(rolling).sum()/pd.Series(type).apply(lambda x:x ==x).rolling(rolling).sum()
    return pct.fillna(0)

def Order_Type2(l):
    openInterestChg = l['open_interest'] - l['open_interest'].shift(1)
    priceChg = l['price'] - l['price'].shift(1)
    return np.where((openInterestChg>0)&(priceChg<0), 2, 0)

def Order_Type2_large(l):
    openInterestChg = l['open_interest'] - l['open_interest'].shift(1)
    priceChg = l['price'] - l['price'].shift(1)
    size = l['volume'] - l['volume'].shift(1)
    return np.where((openInterestChg>0)&(priceChg<0)&(abs(l.size)>10), 2, 0)

def Order_Type2_Pct(trade, depth, rolling=120*5):
    type = Order_Type2(trade)
    pct = pd.Series(type).apply(lambda x:x == 2).rolling(rolling).sum()/pd.Series(type).apply(lambda x:x ==x).rolling(rolling).sum()
    return pct.fillna(0)

def Order_Type2_large_Pct(trade, depth, rolling=120*5):
    type = Order_Type2_large(trade)
    pct = pd.Series(type).apply(lambda x:x ==2).rolling(rolling).sum()/pd.Series(type).apply(lambda x:x ==x).rolling(rolling).sum()
    return pct.fillna(0)

def Order_Type3(l):
    openInterestChg = l['open_interest'] - l['open_interest'].shift(1)
    priceChg = l['price'] - l['price'].shift(1)
    return np.where((openInterestChg<0)&(priceChg>0), 3, 0)

def Order_Type3_large(l):
    openInterestChg = l['open_interest'] - l['open_interest'].shift(1)
    priceChg = l['price'] - l['price'].shift(1)
    # size = l['volume'] - l['volume'].shift(1)
    return np.where((openInterestChg<0)&(priceChg>0)&(abs(l.size)>10), 3, 0)

def Order_Type3_Pct(trade, depth, rolling=120*5):
    type = Order_Type3(trade)
    pct = pd.Series(type).apply(lambda x:x ==3).rolling(rolling).sum()/pd.Series(type).apply(lambda x:x ==x).rolling(rolling).sum()
    return pct.fillna(0)

def Order_Type3_large_Pct(trade, depth, rolling=120*5):
    type = Order_Type3_large(trade)
    pct = pd.Series(type).apply(lambda x:x ==3).rolling(rolling).sum()/pd.Series(type).apply(lambda x:x ==x).rolling(rolling).sum()
    return pct.fillna(0)

def Order_Type4(l):
    openInterestChg = l['open_interest'] - l['open_interest'].shift(1)
    priceChg = l['price'] - l['price'].shift(1)
    return np.where((openInterestChg<0)&(priceChg<0), 4, 0)

def Order_Type4_large(l):
    openInterestChg = l['open_interest'] - l['open_interest'].shift(1)
    priceChg = l['price'] - l['price'].shift(1)
    size = l['volume'] - l['volume'].shift(1)
    return np.where((openInterestChg<0)&(priceChg<0)&(abs(l.size)>10), 4, 0)

def Order_Type4_Pct(trade, depth, rolling=120*5):
    type = Order_Type4(trade)
    pct = pd.Series(type).apply(lambda x:x ==4).rolling(rolling).sum()/pd.Series(type).apply(lambda x:x ==x).rolling(rolling).sum()
    return pct.fillna(0)

def Order_Type4_large_Pct(trade, depth, rolling=120*5):
    type = Order_Type4_large(trade)
    pct = pd.Series(type).apply(lambda x:x ==4).rolling(rolling).sum()/pd.Series(type).apply(lambda x:x ==x).rolling(rolling).sum()
    return pct.fillna(0)

#%%
lags = [5,10,15,30]
def depth_process(data, rolling=60):

    df = data.loc[:,['closetime', 'ask_price1', 'ask_size1', 'bid_price1', 'bid_size1',
       'ask_price2', 'ask_size2', 'bid_price2', 'bid_size2', 'ask_price3',
       'ask_size3', 'bid_price3', 'bid_size3', 'ask_price4', 'ask_size4',
       'bid_price4', 'bid_size4']]

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

    df['depth_1s_bid_size1_shift_1_diff'] = df['bid_size1'] - df['bid_size1'].shift()
    df['depth_1s_ask_size1_shift_1_diff'] = df['ask_size1'] - df['ask_size1'].shift()
    df['depth_1s_bid_size2_shift_1_diff'] = df['bid_size2'] - df['bid_size2'].shift()
    df['depth_1s_ask_size2_shift_1_diff'] = df['ask_size2'] - df['ask_size2'].shift()
    df['depth_1s_bid_size3_shift_1_diff'] = df['bid_size3'] - df['bid_size3'].shift()
    df['depth_1s_ask_size3_shift_1_diff'] = df['ask_size3'] - df['ask_size3'].shift()

    df['bid_ask_size1_spread'] = df['bid_ask_size1_minus'] / df['bid_ask_size1_plus']
    df['bid_ask_size2_spread'] = df['bid_ask_size2_minus'] / df['bid_ask_size2_plus']
    df['bid_ask_size3_spread'] = df['bid_ask_size3_minus'] / df['bid_ask_size3_plus']
    df['bid_ask_size4_spread'] = df['bid_ask_size4_minus'] / df['bid_ask_size4_plus']

    df['HR2'] = ((df['bid_price2'] - df['bid_price2'].shift(1)) - (df['ask_price2'] - df['ask_price2'].shift(1))) / (
            (df['bid_price2'] - df['bid_price2'].shift(1)) + (df['ask_price2'] - df['ask_price2'].shift(1)))

    df['QR1'] = (df['bid_size1'] - df['ask_size1']) / (df['bid_size1'] + df['ask_size1'])
    df['QR2'] = (df['bid_size2'] - df['ask_size2']) / (df['bid_size2'] + df['ask_size2'])

    for rolling in lags:

        # wap1 genetic functions
        df[f'depth_1s_wap1_shift_{rolling}_log_return'] = np.log(df['wap1'].shift(1) / df['wap1'].shift(rolling))
        df[f'depth_1s_wap1_rolling_{rolling}_realized_volatility'] = df['wap1'].rolling(rolling).apply(realized_volatility)
        df[f'depth_1s_wap1_rolling_{rolling}_realized_absvar'] = df['wap1'].rolling(rolling).apply(realized_absvar)
        df[f'depth_1s_wap1_rolling_{rolling}_realized_skew'] = df['wap1'].rolling(rolling).skew()
        df[f'depth_1s_wap1_rolling_{rolling}_realized_kurtosis'] = df['wap1'].rolling(rolling).kurt()

        df[f'depth_1s_wap1_rolling_{rolling}_mean'] = df['wap1'].rolling(rolling).mean()
        df[f'depth_1s_wap1_rolling_{rolling}_std'] = df['wap1'].rolling(rolling).std()
        df[f'depth_1s_wap1_rolling_{rolling}_min'] = df['wap1'].rolling(rolling).min()
        df[f'depth_1s_wap1_rolling_{rolling}_max'] = df['wap1'].rolling(rolling).max()

        df[f'depth_1s_wap1_rolling_{rolling}_mean/std'] = df[f'depth_1s_wap1_rolling_{rolling}_mean']/df[f'depth_1s_wap1_rolling_{rolling}_std']

        df[f'depth_1s_wap1_rolling_{rolling}_quantile_25'] = df['wap1'].rolling(rolling).quantile(.25)
        df[f'depth_1s_wap1_rolling_{rolling}_quantile_75'] = df['wap1'].rolling(rolling).quantile(.75)

        # wap2
        df[f'depth_1s_wap2_shift1_{rolling}_log_return'] = np.log(df['wap2'].shift(1) / df['wap2'].shift(rolling))
        df[f'depth_1s_wap2_rolling_{rolling}_realized_volatility'] = df['wap2'].rolling(rolling).apply(realized_volatility)
        df[f'depth_1s_wap2_rolling_{rolling}_realized_absvar'] = df['wap2'].rolling(rolling).apply(realized_absvar)
        df[f'depth_1s_wap2_rolling_{rolling}_realized_skew'] = df['wap2'].rolling(rolling).skew()
        df[f'depth_1s_wap2_rolling_{rolling}_realized_kurtosis'] = df['wap2'].rolling(rolling).kurt()

        df[f'depth_1s_wap2_rolling_{rolling}_mean'] = df['wap2'].rolling(rolling).mean()
        df[f'depth_1s_wap2_rolling_{rolling}_std'] = df['wap2'].rolling(rolling).std()
        df[f'depth_1s_wap2_rolling_{rolling}_min'] = df['wap2'].rolling(rolling).min()
        df[f'depth_1s_wap2_rolling_{rolling}_max'] = df['wap2'].rolling(rolling).max()

        df[f'depth_1s_wap2_rolling_{rolling}_mean/std'] = df[f'depth_1s_wap2_rolling_{rolling}_mean']/df[f'depth_1s_wap2_rolling_{rolling}_std']

        df[f'depth_1s_wap2_rolling_{rolling}_quantile_25'] = df['wap2'].rolling(rolling).quantile(.25)
        df[f'depth_1s_wap2_rolling_{rolling}_quantile_75'] = df['wap2'].rolling(rolling).quantile(.75)

        df[f'depth_1s_wap3_shift1_{rolling}_log_return'] = np.log(df['wap3'].shift(1) / df['wap3'].shift(rolling))
        df[f'depth_1s_wap3_rolling_{rolling}_realized_volatility'] = df['wap3'].rolling(rolling).apply(realized_volatility)
        df[f'depth_1s_wap3_rolling_{rolling}_realized_absvar'] = df['wap3'].rolling(rolling).apply(realized_absvar)
        df[f'depth_1s_wap3_rolling_{rolling}_realized_skew'] = df['wap3'].rolling(rolling).skew()
        df[f'depth_1s_wap3_rolling_{rolling}_realized_kurtosis'] = df['wap3'].rolling(rolling).kurt()

        df[f'depth_1s_wap3_rolling_{rolling}_mean'] = df['wap3'].rolling(rolling).mean()
        df[f'depth_1s_wap3_rolling_{rolling}_std'] = df['wap3'].rolling(rolling).std()
        df[f'depth_1s_wap3_rolling_{rolling}_min'] = df['wap3'].rolling(rolling).min()
        df[f'depth_1s_wap3_rolling_{rolling}_max'] = df['wap3'].rolling(rolling).max()

        df[f'depth_1s_wap3_rolling_{rolling}_mean/std'] = df[f'depth_1s_wap3_rolling_{rolling}_mean']/df[f'depth_1s_wap3_rolling_{rolling}_std']

        df[f'depth_1s_wap3_rolling_{rolling}_quantile_25'] = df['wap3'].rolling(rolling).quantile(.25)
        df[f'depth_1s_wap3_rolling_{rolling}_quantile_75'] = df['wap3'].rolling(rolling).quantile(.75)

        # wap4 genetic functions
        df[f'depth_1s_wap4_shift_{rolling}_log_return'] = np.log(df['wap4'].shift(1) / df['wap4'].shift(rolling))
        df[f'depth_1s_wap4_rolling_{rolling}_realized_volatility'] = df['wap4'].rolling(rolling).apply(realized_volatility)
        df[f'depth_1s_wap4_rolling_{rolling}_realized_absvar'] = df['wap4'].rolling(rolling).apply(realized_absvar)
        df[f'depth_1s_wap4_rolling_{rolling}_realized_skew'] = df['wap4'].rolling(rolling).skew()
        df[f'depth_1s_wap4_rolling_{rolling}_realized_kurtosis'] = df['wap4'].rolling(rolling).kurt()

        df[f'depth_1s_wap4_rolling_{rolling}_mean'] = df['wap4'].rolling(rolling).mean()
        df[f'depth_1s_wap4_rolling_{rolling}_std'] = df['wap4'].rolling(rolling).std()
        df[f'depth_1s_wap4_rolling_{rolling}_min'] = df['wap4'].rolling(rolling).min()
        df[f'depth_1s_wap4_rolling_{rolling}_max'] = df['wap4'].rolling(rolling).max()

        df[f'depth_1s_wap4_rolling_{rolling}_mean/std'] = df[f'depth_1s_wap4_rolling_{rolling}_mean']/df[f'depth_1s_wap4_rolling_{rolling}_std']

        df[f'depth_1s_wap4_rolling_{rolling}_quantile_25'] = df['wap4'].rolling(rolling).quantile(.25)
        df[f'depth_1s_wap4_rolling_{rolling}_quantile_75'] = df['wap4'].rolling(rolling).quantile(.75)


        df[f'depth_1s_HR1_rolling_{rolling}_mean'] = df['HR1'].rolling(rolling).mean()
        df[f'depth_1s_HR1_rolling_{rolling}_std'] = df['HR1'].rolling(rolling).std()
        df[f'depth_1s_HR1_rolling_{rolling}_mean/std'] = df[f'depth_1s_HR1_rolling_{rolling}_mean']/ df[f'depth_1s_HR1_rolling_{rolling}_std']

        df[f'depth_1s_vtA_rolling_{rolling}_mean'] = df['vtA'].rolling(rolling).mean()
        df[f'depth_1s_vtA_rolling_{rolling}_std'] = df['vtA'].rolling(rolling).std()
        df[f'depth_1s_vtA_rolling_{rolling}_mean/std'] = df[f'depth_1s_vtA_rolling_{rolling}_mean']/df[f'depth_1s_vtA_rolling_{rolling}_std']

        df[f'depth_1s_vtB_rolling_{rolling}_mean'] = df['vtB'].rolling(rolling).mean()
        df[f'depth_1s_vtB_rolling_{rolling}_std'] = df['vtB'].rolling(rolling).std()
        df[f'depth_1s_vtB_rolling_{rolling}_mean/std'] = df[f'depth_1s_vtB_rolling_{rolling}_mean']/df[f'depth_1s_vtB_rolling_{rolling}_std']

        df['Oiab'] = df['vtB'] - df['vtA']
        df[f'Oiab_{rolling}'] = df[f'depth_1s_vtB_rolling_{rolling}_mean'] - df[f'depth_1s_vtA_rolling_{rolling}_mean']
        df[f'Oiab_{rolling}'] = df[f'depth_1s_vtB_rolling_{rolling}_mean'] - df[f'depth_1s_vtA_rolling_{rolling}_mean']
        df[f'Oiab_{rolling}'] = df[f'depth_1s_vtB_rolling_{rolling}_mean'] - df[f'depth_1s_vtA_rolling_{rolling}_mean']


        df[f'bidprice1_press_rolling_{rolling}'] = (df['mid_price1'] / (df['bid_price1'] - df['mid_price1'])) / (
                    df['mid_price1'] / (df['bid_price1'] - df['mid_price1'])).rolling(rolling).sum()
        df[f'askprice1_press_rolling_{rolling}'] = (df['mid_price1'] / (df['ask_price1'] - df['mid_price1'])) / (
                df['mid_price1'] / (df['ask_price1'] - df['mid_price1'])).rolling(rolling).sum()
        df[f'bidprice2_press_rolling_{rolling}'] = (df['mid_price2'] / (df['bid_price2'] - df['mid_price2'])) / (
                df['mid_price2'] / (df['bid_price2'] - df['mid_price2'])).rolling(rolling).sum()
        df[f'askprice2_press_rolling_{rolling}'] = (df['mid_price2'] / (df['ask_price2'] - df['mid_price2'])) / (
                df['mid_price2'] / (df['ask_price2'] - df['mid_price2'])).rolling(rolling).sum()

        df[f'bidask1_press_rolling_{rolling}'] = np.log(
            (df[f'bidprice1_press_rolling_{rolling}'] * df['bid_size1'].rolling(rolling).sum()) / (df[f'askprice1_press_rolling_{rolling}']) * df[
                'ask_size1'].rolling(rolling).sum())
        df[f'bidask2_press_rolling_{rolling}'] = np.log(
            (df[f'bidprice2_press_rolling_{rolling}'] * df['bid_size2'].rolling(rolling).sum()) / (df[f'askprice2_press_rolling_{rolling}']) * df[
                'ask_size2'].rolling(rolling).sum())


    df = df.fillna(0)
    df = df.replace(np.inf, 1)
    df = df.replace(-np.inf, -1)

    return df

def trade_process(trade, rolling=60):

    df = trade.loc[:, ['closetime', 'price', 'volume', 'open_interest', 'amount']]
    df['size'] = df['volume'] - df['volume'].shift(1)
    df['BS'] = np.where((df['open_interest']-df['open_interest'].shift(1)) > 0, 'B', np.where((df['open_interest']-df['open_interest'].shift(1)) < 0, 'S', 0))
    df['active_buy'] = np.where(df['BS'] == 'B', df['price'], 0)
    df['active_sell'] = np.where(df['BS'] == 'S', df['price'], 0)
    df = df.drop(['BS'], axis=1)
    for rolling in lags:

        df[f'buy_ratio_rolling_{rolling}'] = (df['active_buy'] * df['size']).rolling(rolling).mean() / (
                df['active_buy'] * df['size']).rolling(rolling).std()
        df[f'sell_ratio_rolling_{rolling}'] = (df['active_sell'] * abs(df['size'])).rolling(rolling).mean() / (
                df['active_buy'] * abs(df['size'])).rolling(rolling).std()

        df[f'depth_1s_last_price_shift_{rolling}_60_log_return'] = np.log(df['price'].shift(1) / df['price'].shift(rolling))
        # realized volatility
        df[f'depth_1s_log_return_rolling_{rolling}_realized_volatility'] = df[f'depth_1s_last_price_shift_{rolling}_60_log_return'].rolling(rolling).apply(realized_volatility)
        # realized absvar
        df[f'depth_1s_log_return_rolling_{rolling}_realized_absvar'] = df[f'depth_1s_last_price_shift_{rolling}_60_log_return'].rolling(rolling).apply(realized_absvar)
        # realized skew
        df[f'depth_1s_log_return_rolling_{rolling}_realized_skew'] = df[f'depth_1s_last_price_shift_{rolling}_60_log_return'].rolling(rolling).skew()
        # realized kurt
        df[f'depth_1s_log_return_rolling_{rolling}_realized_skew'] = df[f'depth_1s_last_price_shift_{rolling}_60_log_return'].rolling(rolling).kurt()

        df[f'depth_1s_log_rolling_{rolling}_quantile_25'] = df[f'depth_1s_last_price_shift_{rolling}_60_log_return'].rolling(rolling).quantile(.25)
        df[f'depth_1s_log_rolling_{rolling}_quantile_75'] = df[f'depth_1s_last_price_shift_{rolling}_60_log_return'].rolling(rolling).quantile(.75)

        df[f'depth_1s_log_percentile_rolling_{rolling}'] = df[f'depth_1s_log_rolling_{rolling}_quantile_75'] - df[f'depth_1s_log_rolling_{rolling}_quantile_25']


        df[f'depth_1s_size_rolling_{rolling}_realized_absvar'] = df['size'].rolling(rolling).apply(realized_absvar)

        df[f'depth_1s_size_rolling_{rolling}_quantile_25'] = df['size'].rolling(rolling).quantile(.25)
        df[f'depth_1s_size_rolling_{rolling}_quantile_75'] = df['size'].rolling(rolling).quantile(.75)
        df[f'depth_1s_size_percentile_rolling_{rolling}'] = df[f'depth_1s_size_rolling_{rolling}_quantile_75'] - df[f'depth_1s_size_rolling_{rolling}_quantile_25']


        # amount genetic functions
        # df['amount'] = df['last_price'] * df['size']

        df['trade_mid_price'] = np.where(df['size'] > 0, (df['amount'] - df['amount'].shift(1)) / df['size'], df['price'])
        df[f'depth_1s_mid_price_rolling_{rolling}_mean'] = df['mid_price'].rolling(rolling).mean()
        df[f'depth_1s_mid_price_rolling_{rolling}_std'] = df['mid_price'].rolling(rolling).std()
        df[f'depth_1s_mid_price_rolling_{rolling}_mean/std'] = df[f'depth_1s_mid_price_rolling_{rolling}_mean']/df[f'depth_1s_mid_price_rolling_{rolling}_std']

        df[f'depth_1s_amount_rolling_{rolling}_mean'] = df['amount'].rolling(rolling).mean()
        df[f'depth_1s_amount_rolling_{rolling}_std'] = df['amount'].rolling(rolling).std()
        df[f'depth_1s_amount_rolling_{rolling}_mean/std'] = df[f'depth_1s_amount_rolling_{rolling}_mean']/df[f'depth_1s_amount_rolling_{rolling}_std']
        df[f'depth_1s_amount_rolling_{rolling}_quantile_25'] = df['amount'].rolling(rolling).quantile(.25)
        df[f'depth_1s_amount_rolling_{rolling}_quantile_75'] = df['amount'].rolling(rolling).quantile(.75)


    df = df.fillna(0)
    df = df.replace(np.inf, 1)
    df = df.replace(-np.inf, -1)

    return df

def add_factor_process(depth, trade):

    df = pd.DataFrame()
    # df = depth.loc[:,
    #      ['closetime', 'ask_price1', 'ask_size1', 'bid_price1', 'bid_size1', 'ask_price2', 'ask_size2', 'bid_price2',
    #       'bid_size2', 'ask_price3', 'ask_size3', 'bid_price3', 'bid_size3', 'ask_price4', 'ask_size4', 'bid_price4',
    #       'bid_size4', 'ask_price5', 'ask_size5', 'bid_price5', 'bid_size5']]
    df = trade.loc[:,['closetime', 'price','size', 'highest', 'lowest','volume','amount']]
    # df['price'] = trade.loc[:, ['price']]
    # df['volume'] = trade.loc[:, ['volume']]
    # df['size'] = trade.loc[:, ['size']]
    # df['amount'] = trade.loc[:, ['amount']]
    # df['open_interest'] = trade.loc[:, ['open_interest']]
    df['ask_age'] = ask_age(depth=depth, rolling=60*2)
    df['bid_age'] = bid_age(depth=depth, rolling=60*2)
    df['inf_ratio'] = inf_ratio(depth=None, trade=trade, rolling=1000*10)
    df['arrive_rate'] = arrive_rate(depth=None, trade=trade, rolling=1000*10)
    df['arrive_rate_2'] = arrive_rate_2(depth=None, trade=trade, rolling=1000*10)
    df['depth_price_range'] = depth_price_range(depth=depth, trade=None)
    df['bp_rank'] = bp_rank(depth=depth, trade=None, rolling=1000*10)
    df['ap_rank'] = ap_rank(depth=depth, trade=None, rolling=1000*10)
    df['price_impact'] = price_impact(depth=depth, trade=None, level=5)
    df['depth_price_skew'] = depth_price_skew(depth=depth, trade=None)
    df['depth_price_kurt'] = depth_price_kurt(depth=depth, trade=None)
    df['rolling_return'] = rolling_return(depth=depth, trade=None, rolling=1000*10)
    df['buy_increasing'] = buy_increasing(depth=None, trade=trade, rolling=1000*10)
    df['sell_increasing'] = sell_increasing(depth=None, trade=trade, rolling=1000*10)
    df['price_idxmax'] = price_idxmax(depth=depth, trade=None, rolling=120)
    df['center_deri_two'] = center_deri_two(depth=depth, trade=None, rolling=120*10)
    df['quasi'] = quasi(depth=depth, trade=None, rolling=1000*10)
    df['last_range'] = last_range(depth=None, trade=trade, rolling=1000*10)
    df['avg_trade_volume'] = avg_trade_volume(depth=depth, trade=trade, rolling=1200*10)
    df['avg_spread'] = avg_spread(depth=depth, trade=None, rolling=1000*10)
    df['avg_turnover'] = avg_turnover(depth=depth, trade=trade, rolling=1000*10)
    df['abs_volume_kurt'] = abs_volume_kurt(depth=None, trade=trade,rolling=1000*10)
    df['abs_volume_skew'] = abs_volume_skew(depth=None, trade=trade, rolling=1000*10)
    df['volume_kurt'] = volume_kurt(depth=None, trade=trade, rolling=1000*10)
    df['volume_skew'] = volume_skew(depth=None, trade=trade, rolling=1000*10)
    df['price_kurt'] = price_kurt(depth=None, trade=trade, rolling=1000*10)
    df['price_skew'] = price_skew(depth=None, trade=trade, rolling=1000*10)
    df['bv_divide_tn'] = bv_divide_tn(depth=depth, trade=trade, rolling=120*10)
    df['av_divide_tn'] = av_divide_tn(depth=depth, trade=trade, rolling=120*10)
    df['weighted_price_to_mid'] = weighted_price_to_mid(depth=depth, trade=None, levels=5, alpha=1)
    df['ask_withdraws'] = ask_withdraws(depth=depth, trade=None)
    df['bid_withdraws'] = bid_withdraws(depth=depth, trade=None)
    df['z_t'] = z_t(trade=trade, depth=depth)
    df['voi'] = voi(trade=trade, depth=depth)
    df['voi2'] = voi2(depth=depth, trade=trade)
    df['voi2_level2'] = voi2_level2(depth=depth,trade=trade)
    df['voi2_level3'] = voi2_level3(depth=depth, trade=trade)
    df['voi2_level4'] = voi2_level4(depth=depth, trade=trade)
    df['voi2_level5'] = voi2_level5(depth=depth, trade=trade)
    df['wa'], df['wb'] = cal_weight_volume(depth=depth)
    df['slope'] = slope(depth=depth)
    df['mpb'] = mpb(depth=depth, trade=trade)
    df['mpb_5min'] = mpb_5min(depth=depth, trade=trade,rolling=120*5)
    df['mpc'] = mpc(depth=depth,trade=None)
    df['oir'] = oir(depth=depth,trade=None)
    df['price_weighted_pressure'] = price_weighted_pressure(depth=depth, kws={})
    df['volume_order_imbalance'] = volume_order_imbalance(depth=depth, kws={})
    df['get_mid_price_change'] = get_mid_price_change(depth=depth, drop_first=True)
    df['positive_buying'] = positive_buying(depth=depth,trade=trade,rolling=120)
    df['positive_selling'] = positive_selling(depth=depth,trade=trade,rolling=120)
    df['buying_amount_ratio'] = buying_amount_ratio(depth=depth,trade=trade,rolling=1000*10)
    df['selling_ratio'] = selling_ratio(depth=depth, trade=trade, rolling=1000*10)
    df['buying_amount_strength'] = buying_amount_strength(depth=depth,trade=trade,rolling=1000*10)
    df['buying_amount_ratio'] = buying_amount_ratio(depth=depth,trade=trade,rolling=1000*10)
    df['buying_amount_strength'] = buying_amount_strength(depth=depth,trade=trade,rolling=1000*10)
    df['buying_willing'] = buying_willing(depth=depth,trade=trade,rolling=1000*10)
    df['large_order_ratio'] = large_order_ratio(depth=depth,trade=trade, rolling=120*10)
    df['Open_Close_Percentage_5min'] = Open_Close_Percentage(depth=None, trade=trade, rolling=120 * 5)
    df['buy_price_bias_level1'], df['buy_amount_agg_ratio_level1'] = buy_order_aggressivenes_level1(depth=depth,trade=trade,rolling=1000*10)
    df['buy_price_bias_level2'], df['buy_amount_agg_ratio_level2'] = buy_order_aggressivenes_level2(depth=depth,trade=trade,rolling=1000*10)
    df['sell_price_bias_level1'], df['sell_amount_agg_ratio_level1'] = sell_order_aggressivenes_level1(depth=depth,trade=trade,rolling=1000*10)
    df['sell_price_bias_level2'], df['sell_amount_agg_ratio_level2'] = sell_order_aggressivenes_level2(depth=depth,trade=trade,rolling=1000*10)
    # df['Open_Close_Percentage_10min'] = Open_Close_Percentage(depth=None, trade=trade, rolling=120 * 10)
    # df['Open_Close_Percentage_15min'] = Open_Close_Percentage(depth=None, trade=trade, rolling=120 * 15)
    # df['Open_Close_Percentage_30min'] = Open_Close_Percentage(depth=None, trade=trade, rolling=120 * 30)
    df['Open_Interest_Change'] = Open_Interest_Change(depth=None, trade=trade, rolling=120 * 10*10)
    # df['li_1'],df['li_2'],df['li_3'],df['li_4'],df['li_5'] = length_imbalance(depth=depth,trade=None,level=5)
    # df['hi_2'],df['hi_3'],df['hi_4'],df['hi_5'] = height_imbalance(depth=depth,trade=None, level=5)
    df['pv_ic'], df['oi_ic'] = corr_pv(depth=None, trade=trade, rolling=120*10*10)
    df['flowInRatio'] = flowInRatio(depth=None, trade=trade, rolling=120*10*10)
    df['large_buy_ratio'], df['large_sell_ratio'] = large_order(depth=None, trade=trade, rolling=120*10*10)

    return df

def order_type_process(depth, trade):

    df = pd.DataFrame()
    trade = trade.reset_index()
    df['datetime'] = trade.loc[:,['datetime']]
    df['closetime'] = trade.loc[:,['closetime']]
    df['Order_Type1_Pct_5min'] = Order_Type1_Pct(trade=trade, depth=None, rolling=120 * 5)
    df['Order_Type2_Pct_5min'] = Order_Type2_Pct(trade=trade, depth=None, rolling=120 * 5)
    df['Order_Type3_Pct_5min'] = Order_Type3_Pct(trade=trade, depth=None, rolling=120 * 5)
    df['Order_Type4_Pct_5min'] = Order_Type4_Pct(trade=trade, depth=None, rolling=120 * 5)
    df['Order_Type1_Pct_10min'] = Order_Type1_Pct(trade=trade, depth=None, rolling=120 * 10)
    df['Order_Type2_Pct_10min'] = Order_Type2_Pct(trade=trade, depth=None, rolling=120 * 10)
    df['Order_Type3_Pct_10min'] = Order_Type3_Pct(trade=trade, depth=None, rolling=120 * 10)
    df['Order_Type4_Pct_10min'] = Order_Type4_Pct(trade=trade, depth=None, rolling=120 * 10)
    df['Order_Type1_Pct_15min'] = Order_Type1_Pct(trade=trade, depth=None, rolling=120 * 15)
    df['Order_Type2_Pct_15min'] = Order_Type2_Pct(trade=trade, depth=None, rolling=120 * 15)
    df['Order_Type3_Pct_15min'] = Order_Type3_Pct(trade=trade, depth=None, rolling=120 * 15)
    df['Order_Type4_Pct_15min'] = Order_Type4_Pct(trade=trade, depth=None, rolling=120 * 15)
    df['Order_Type1_Pct_30min'] = Order_Type1_Pct(trade=trade, depth=None, rolling=120 * 30)
    df['Order_Type2_Pct_30min'] = Order_Type2_Pct(trade=trade, depth=None, rolling=120 * 30)
    df['Order_Type3_Pct_30min'] = Order_Type3_Pct(trade=trade, depth=None, rolling=120 * 30)
    df['Order_Type4_Pct_30min'] = Order_Type4_Pct(trade=trade, depth=None, rolling=120 * 30)
    df['Order_Type1_large_Pct_5min'] = Order_Type1_large_Pct(trade=trade, depth=None, rolling=120 * 5)
    df['Order_Type2_large_Pct_5min'] = Order_Type2_large_Pct(trade=trade, depth=None, rolling=120 * 5)
    df['Order_Type3_large_Pct_5min'] = Order_Type3_large_Pct(trade=trade, depth=None, rolling=120 * 5)
    df['Order_Type4_large_Pct_5min'] = Order_Type4_large_Pct(trade=trade, depth=None, rolling=120 * 5)
    df['Order_Type1_large_Pct_10min'] = Order_Type1_large_Pct(trade=trade, depth=None, rolling=120 * 10)
    df['Order_Type2_large_Pct_10min'] = Order_Type2_large_Pct(trade=trade, depth=None, rolling=120 * 10)
    df['Order_Type3_large_Pct_10min'] = Order_Type3_large_Pct(trade=trade, depth=None, rolling=120 * 10)
    df['Order_Type4_large_Pct_10min'] = Order_Type4_large_Pct(trade=trade, depth=None, rolling=120 * 10)
    df['Order_Type1_large_Pct_15min'] = Order_Type1_large_Pct(trade=trade, depth=None, rolling=120 * 15)
    df['Order_Type2_large_Pct_15min'] = Order_Type2_large_Pct(trade=trade, depth=None, rolling=120 * 15)
    df['Order_Type3_large_Pct_15min'] = Order_Type3_large_Pct(trade=trade, depth=None, rolling=120 * 15)
    df['Order_Type4_large_Pct_15min'] = Order_Type4_large_Pct(trade=trade, depth=None, rolling=120 * 15)
    df['Order_Type1_large_Pct_30min'] = Order_Type1_large_Pct(trade=trade, depth=None, rolling=120 * 30)
    df['Order_Type2_large_Pct_30min'] = Order_Type2_large_Pct(trade=trade, depth=None, rolling=120 * 30)
    df['Order_Type3_large_Pct_30min'] = Order_Type3_large_Pct(trade=trade, depth=None, rolling=120 * 30)
    df['Order_Type4_large_Pct_30min'] = Order_Type4_large_Pct(trade=trade, depth=None, rolling=120 * 30)

    return df


