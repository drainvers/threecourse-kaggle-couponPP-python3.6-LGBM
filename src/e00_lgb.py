"""
    This file runs lightgbm, train with train data and predict test data.
"""

import pandas as pd
import numpy as np
import gc
import lightgbm as lgb
import joblib
from sklearn.model_selection import StratifiedKFold
from util_logger import get_logger

import sys
argvs = sys.argv 
_ , runtype, version = argvs
LOG = get_logger()
LOG.info("start e00")

def run_lightgbm(labels, weights, data):

    # convert data into lgb.Dataset. 
    # train using 80% of the data, 20% of the data is used for watchlist
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 123)
    idx_train, idx_test = list(skf.split(data, labels))[0]

    dtrain = lgb.Dataset(data[idx_train,:], weight=weights[idx_train], label=labels[idx_train])
    dvalid = lgb.Dataset(data[idx_test,:], weight=weights[idx_test], label=labels[idx_test])
    watchlist_sets = [dtrain, dvalid]
    watchlist_names = ['train', 'eval']

    # set parameters
    # logistic regression with gradient boosted decision trees
    num_rounds = 1000
    # Old XGBoost params for reference
    # params={'objective': 'binary',
    #         'eta': 0.05,
    #         'max_depth': 6,
    #         'subsample': 0.85,
    #         'colsample_bytree': 0.8,
    #         "silent" : 1,
    #         "seed" : 12345,
    #         "min_child_weight" : 1
    #         'metric': 'binary'
    #         }
    params = {
        'boosting_type': 'gbdt',
        'colsample_bytree': 1.0,
        'importance_type': 'split',
        'learning_rate': 0.1,
        'max_depth': -1,
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'min_split_gain': 0.0,
        'num_iterations': 100,
        'n_jobs': -1,
        'num_leaves': 31,
        'objective': 'binary',
        'random_state': 12345,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'subsample': 1.0,
        'subsample_for_bin': 200000,
        'subsample_freq': 0,
        'metric': 'binary_logloss'
    }

    # run lightgbm
    LOG.info("started lgb")
    model = lgb.train(params, dtrain, num_boost_round=num_rounds, valid_sets=watchlist_sets, valid_names=watchlist_names, early_stopping_rounds=10)

    return model

def predict_and_write(in_fname, out_fname):
    lgbdata = joblib.load(in_fname) 

    # to reduce memory consumption, predict by small chunks.
    def predict(data):
        return pd.Series(model.predict(data)) 

    data_list = np.array_split(lgbdata["data"], 8)
    preds = pd.concat([predict(data) for data in data_list])
    preds.to_csv(out_fname)

# train model
lgbdata_train = joblib.load( '../model/lgbdata_train.pkl') 
model = run_lightgbm(lgbdata_train["labels"], lgbdata_train["weights"], lgbdata_train["data"])
model.save_model('../model/lgb.model')

# release memory
lgbdata_train = None
gc.collect()

# predict test data and write
predict_and_write( '../model/lgbdata_test.pkl', "../model/lgb_predicted.txt") 

LOG.info("finished")

