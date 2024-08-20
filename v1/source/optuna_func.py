from pyspark.sql.types import *
import pandas as pd
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from optuna import Trial, visualization
import pandas as pd

import numpy as np
from functools import partial
from sklearn.metrics import log_loss
from scipy.stats import norm
import pandas as pd
from google.cloud import storage


"""
This paragraph contains several metrics for use in Optuna studies as well as objective functions.
"""

# Mean Absolute Error (MAE)
def return_mae(params, dtrain, dvalidation):
    model = xgb.train(params, dtrain, num_boost_round=4000, evals=[(dvalidation, 'eval')], early_stopping_rounds=20, verbose_eval=0)
    preds = model.predict(dvalidation)
    dvalid_labels = dvalidation.get_label()
    mae = np.mean(np.abs(dvalid_labels - preds))
    return mae

# Root Mean Squared Error (RMSE)
def return_rmse(params, dtrain, dvalidation):
    model = xgb.train(params, dtrain, num_boost_round=4000, evals=[(dvalidation, 'eval')], early_stopping_rounds=20, verbose_eval=0)
    preds = model.predict(dvalidation)
    dvalid_labels = dvalidation.get_label()
    rmse = np.sqrt(np.mean((dvalid_labels - preds) ** 2))
    return rmse

# Mean Squared Error (MSE)
def return_mse(params, dtrain, dvalidation):
    model = xgb.train(params, dtrain, num_boost_round=4000, evals=[(dvalidation, 'eval')], early_stopping_rounds=20, verbose_eval=0)
    preds = model.predict(dvalidation)
    dvalid_labels = dvalidation.get_label()
    mse = np.mean((dvalid_labels - preds) ** 2)
    return mse

# Mean Absolute Scaled Error (MASE)
def return_mase(params, dtrain, dvalidation, train_labels):
    model = xgb.train(params, dtrain, num_boost_round=4000, evals=[(dvalidation, 'eval')], early_stopping_rounds=20, verbose_eval=0)
    preds = model.predict(dvalidation)
    dvalid_labels = dvalidation.get_label()
    naive_forecast_error = np.mean(np.abs(train_labels[1:] - train_labels[:-1]))
    mase = np.mean(np.abs(dvalid_labels - preds)) / naive_forecast_error
    return mase

# Mean Squared Logarithmic Error (MSLE)
def return_msle(params, dtrain, dvalidation):
    model = xgb.train(params, dtrain, num_boost_round=4000, evals=[(dvalidation, 'eval')], early_stopping_rounds=20, verbose_eval=0)
    preds = model.predict(dvalidation)
    dvalid_labels = dvalidation.get_label()
    msle = np.mean((np.log1p(dvalid_labels) - np.log1p(preds)) ** 2)
    return msle

# Mean Absolute Percent Error
def return_mape(params, dtrain, dvalidation):
    model = xgb.train(params, dtrain, num_boost_round=4000, evals = [(dvalidation, 'eval')], early_stopping_rounds = 20, verbose_eval = 0)
    preds = model.predict(dvalidation)
    dvalid_labels  = dmvalidationhome.get_label()
    epsilon = 3
    mape = np.mean(np.abs((dvalid_labels - preds) / (dvalid_labels + epsilon)))
    return mape

def return_log_loss(params, dtrain, dvalidation):
    # Train the model
    model = xgb.train(params, dtrain, num_boost_round=4000, evals=[(dvalidation, 'eval')], early_stopping_rounds=20, verbose_eval=0)
    
    # Predict the probabilities for the validation set
    preds = model.predict(dvalidation)
    
    # Get the labels
    dvalid_labels = dvalidation.get_label()
    
    # Calculate log loss
    logloss = log_loss(dvalid_labels, preds)
    
    return logloss

def objective_binary_log_loss(trial, dtrain, dvalidation):
    # Define the parameter search space
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': 'gbtree',
        'eta': trial.suggest_loguniform('eta', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
    }
    
    # Return the log loss
    return return_log_loss(params, dtrain, dvalidation)

def objective_mae(trial, max_depth_min, max_depth_max, reg_alpha_min, reg_alpha_max, min_child_weight_min, min_child_weight_max, gamma_min, gamma_max, learning_rate_min, learning_rate_max, colsample_bytree_min, colsample_bytree_max, subsample_min, subsample_max, dtrain, dvalidation):
    param = {
        "max_depth": trial.suggest_int('max_depth', max_depth_min, max_depth_max),
        "reg_alpha": trial.suggest_float('reg_alpha', reg_alpha_min, reg_alpha_max),
        "min_child_weight": trial.suggest_int('min_child_weight', min_child_weight_min, min_child_weight_max),
        "gamma": trial.suggest_float('gamma', gamma_min, gamma_max),
        "learning_rate": trial.suggest_float('learning_rate', learning_rate_min, learning_rate_max),
        "colsample_bytree": trial.suggest_float('colsample_bytree', colsample_bytree_min, colsample_bytree_max),
        "subsample": trial.suggest_float('subsample', subsample_min, subsample_max),
        "nthread": -1,#Specify an objective if needed, 'reg:squarederror' for regression
    }

    return return_mae(param, dtrain, dvalidation)

def optuna_mlb_study(df_training, df_validation, df_test, dmtraining, dmvalidation, dmtrainval, dmtest):
    study = optuna.create_study(direction = 'minimize', sampler = TPESampler(38))

    objective_with_args = partial(objective_mae, max_depth_min = 3, max_depth_max = 10, reg_alpha_min = 0, reg_alpha_max = 10, min_child_weight_min = 0, min_child_weight_max = 10, gamma_min = 0, gamma_max = 5, learning_rate_min = 0, learning_rate_max = .5, colsample_bytree_min = .4, colsample_bytree_max = .9, subsample_min = .4, subsample_max = .9, dtrain = dmtraining ,dvalidation = dmvalidation)

    study.optimize(objective_with_args, n_trials = 100)
    best_trial = study.best_trial

    best_model = xgb.train(best_trial.params, dmtrainval, num_boost_round=4000, evals=[(dmtest, 'eval')], early_stopping_rounds=20, verbose_eval=0)

    preds_training = best_model.predict(dmtraining)
    preds_validation = best_model.predict(dmvalidation)
    preds_test = best_model.predict(dmtest)

    df_training_ = df_training.copy()
    df_validation_ = df_validation.copy()
    df_test_ = df_test.copy()

    df_training_['model_prediction'] = preds_training
    df_training_['dataset']='train'
    df_validation_['model_prediction'] = preds_validation
    df_validation_['dataset']='validation'
    df_test_['model_prediction'] = preds_test
    df_test_['dataset']='test'

    dfs = [df_training_,df_validation_,df_test_]
    union_df = pd.concat(dfs).drop_duplicates().reset_index(drop=True)

    return union_df, study, best_model

def optuna_mlb_study_binary(df_training, df_validation, df_test, dmtraining, dmvalidation, dmtrainval, dmtest):
    study = optuna.create_study(direction = 'minimize', sampler = TPESampler(38))

    objective_with_args = partial(objective_binary_log_loss, dtrain = dmtraining ,dvalidation = dmvalidation)

    study.optimize(objective_with_args, n_trials = 100)
    best_trial = study.best_trial

    best_model = xgb.train(best_trial.params, dmtrainval, num_boost_round=4000, evals=[(dmtest, 'eval')], early_stopping_rounds=20, verbose_eval=0)

    preds_training = best_model.predict(dmtraining)
    preds_validation = best_model.predict(dmvalidation)
    preds_test = best_model.predict(dmtest)

    df_training_ = df_training.copy()
    df_validation_ = df_validation.copy()
    df_test_ = df_test.copy()

    df_training_['model_prediction'] = preds_training
    df_training_['dataset']='train'
    df_validation_['model_prediction'] = preds_validation
    df_validation_['dataset']='validation'
    df_test_['model_prediction'] = preds_test
    df_test_['dataset']='test'

    dfs = [df_training_,df_validation_,df_test_]
    union_df = pd.concat(dfs).drop_duplicates().reset_index(drop=True)

    return union_df, study, best_model