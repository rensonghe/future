#%%
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,roc_auc_score, classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
import time
from functools import reduce
import pyarrow
import datetime
from pyarrow import fs
import pyarrow.parquet as pq
from bayes_opt import BayesianOptimization
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import pearsonr
#%%
test_high_low = final_data.groupby(pd.Grouper(freq='5s')).agg({'price':['min','max']})
test_high_low.columns = ['_'.join(col) for col in test_high_low.columns]
test_high_low['diff'] = test_high_low['price_max']-test_high_low['price_min']
#%%
test_high_low = test_high_low.dropna(axis=0)