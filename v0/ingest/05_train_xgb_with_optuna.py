import argparse
import pyspark

from pyspark.sql.types import *
import pyspark.sql.functions as f
import pandas as pd
import xgboost as xgb
import shap
import optuna
from optuna.samplers import TPESampler
from optuna import Trial, visualization
import pandas as pd
import re
import numpy as np
from functools import partial
from sklearn.metrics import log_loss
from scipy.stats import norm
import pandas as pd
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pickle
from google.cloud import storage

def upload_pickled_model(model, bucket_name, destination_path):
    """
    Pickles the model and uploads it to the specified Google Cloud Storage path.

    :param model: The model to be pickled and uploaded.
    :param bucket_name: The name of the GCS bucket.
    :param destination_path: The destination path within the GCS bucket.
    """
    # Serialize the model to a bytes object
    pickled_model = pickle.dumps(model)

    # Initialize a storage client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Create a new blob in the bucket
    blob = bucket.blob(destination_path)

    # Upload the serialized model
    blob.upload_from_string(pickled_model)

    print(f"Model uploaded to {bucket_name}/{destination_path}.")

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
    
def df_to_split_dmatrix(pdf, split_col = 'season', validation_set_value = 2023, test_set_value = 2024):
    """
    Accepts inputs of a pandas df (pdf), a column over which train/validation/test sets 
    are to be split, and the value of the validation and test sets in that column.

    Any value in the column alphanumerically before validation_set_value is used as the training data.
    """
    
    def model_dataframe_to_dmatrix(model_dataframe):
        features = model_dataframe.iloc[:,2:-1]
        label = model_dataframe.iloc[:,-1]

        return xgb.DMatrix(features,label)
    
    df_training = pdf[pdf[split_col]<validation_set_value]
    df_validation = pdf[pdf[split_col]==validation_set_value]
    df_test = pdf[pdf[split_col]==test_set_value]
    df_trainval = pdf[pdf[split_col]<=validation_set_value]

    dmtraining = model_dataframe_to_dmatrix(df_training)
    dmvalidation = model_dataframe_to_dmatrix(df_validation)
    dmtrainval = model_dataframe_to_dmatrix(df_trainval)
    dmtest = model_dataframe_to_dmatrix(df_test)

    return df_training, df_validation, df_test, dmtraining, dmvalidation, dmtrainval, dmtest

"""
Run a one-sided test to determine if the model has a >55% success rate:
"""
def display_accuracy_55_test(sdf):

    # Convert to pandas DataFrame
    df = sdf.toPandas()

    # Null hypothesis success rate
    p0 = 0.55
    alpha = 0.05  # significance level

    results = []

    for index, row in df.iterrows():
        x = row['correct']
        n = row['games']
        
        # Sample proportion
        p_hat = x / n
        
        # Standard error
        se = (p0 * (1 - p0) / n) ** 0.5
        
        # Z-score
        z = (p_hat - p0) / se
        
        # One-tailed p-value
        p_value = 1 - norm.cdf(z)
        
        # Determine if the success rate is significantly better than 55%
        better_than_55 = 'Yes' if p_value < alpha else 'No'
        
        results.append({
            'dataset': row['dataset'],
            'season': row['season'],
            'game_month': row['game_month'],
            'yrmo': row['yrmo'],
            'games': row['games'],
            'correct': row['correct'],
            'success_rate': p_hat,
            'z_score': z,
            'p_value': p_value,
            'better_than_55%': better_than_55
        })

    # Convert results to DataFrame for better visualization
    results_df = pd.DataFrame(results)

    return spark.createDataFrame(results_df).groupBy('dataset').agg((f.sum('correct')/f.sum('games')).alias('pct'))


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

def run_shap(best_model, pdf, destination_bucket_name, destination_blob_name):
    # Assuming best_model is an XGBoost Booster object
    # Convert the Booster to a compatible XGBClassifier
    model = XGBClassifier()
    model._Booster = best_model

    """
    Run a SHAP Explainer:
    """
    # Verify the objective to confirm binary classification settings
    objective = best_model.attributes().get("objective", "")
    if "multi:softprob" in objective or "binary:logistic" in objective:
        print("Model is configured for binary classification.")

    # Use TreeExplainer for the model
    explainer = shap.TreeExplainer(model)

    # Use the training features
    X_train = pdf.iloc[:, 2:-1]  # Adjust indexing based on your actual data

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_train)

    # Check the output shape to ensure correct handling
    if isinstance(shap_values, list) and len(shap_values) == 2:
        print("Multi-class SHAP values detected in a binary setting. Using only one class SHAP values.")
        shap_values = shap_values[1]  # Use SHAP values for the positive class

    # Plot summary of feature importance
    plt.figure()
    shap.summary_plot(shap_values, X_train, show=False)
    plt.savefig('shap_summary_plot.png')
    plt.close()

    # Upload the file to Google Cloud Storage
    upload_to_gcs('shap_summary_plot.png', destination_bucket_name, destination_blob_name)

def upload_to_gcs(source_file_name, bucket_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # Initialize a storage client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Create a new blob and upload the file's content
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} uploaded to {bucket_name}/{destination_blob_name}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MLB Xgboost inputs")
    parser.add_argument("--input_bucket", type=str, help="a GCS Bucket")
    parser.add_argument("--input_path", type=str, help="04_prepare.. output path")
    parser.add_argument("--output_bucket", type=str, help="a GCS Bucket")
    parser.add_argument("--output_destination", type=str, help="a location within GCS bucket where output is stored")
    parser.add_argument("--model_tag", type=str, help="folder in output_dest for all output")
    parser.add_argument("--write_mode", type=str, help="overwrite or append, as used in spark.write.*")
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder \
        .appName("Prepare MLB XGBoost Model Inputs") \
        .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

    input_df = spark.read.format("parquet").load(f'gs://{args.input_bucket}/{args.input_path}')

    df_training, df_validation, df_test, dmtraining, dmvalidation, dmtrainval, dmtest =  df_to_split_dmatrix( \
        pdf = input_df.toPandas(), split_col = 'season' \
            , validation_set_value = 2023, test_set_value = 2024)

    union_df, study, best_model =  \
        optuna_mlb_study_binary(df_training, df_validation, df_test, dmtraining, dmvalidation, dmtrainval, dmtest)
    
    df_result = spark.createDataFrame(union_df) \
        .withColumn("rounded",f.expr("round(model_prediction,1)")) \
        .withColumn('pred_binary', f.expr("case when model_prediction >.5 then 1 else 0 end")) \
        .withColumn('correct',f.expr("case when pred_binary = HomeTeamWin then 1 else 0 end"))
    
    df_result_summary = df_result \
            .groupBy("season","game_month", "dataset").agg(f.avg('model_prediction'),f.count(f.lit(1)).alias('games'), f.sum('correct').alias('correct')) \
                .withColumn('yrmo',f.expr("concat(cast(season as string),case when game_month<10 then '0' else '' end, cast(game_month as string))")).withColumn('pct',f.expr("correct/games"))

    """
    Run a one-sided test to determine if the model has a >55% success rate:
    """
    df_accuracy_55_result = display_accuracy_55_test(df_result_summary)
    run_shap(best_model, df_training, args.output_bucket, f"{args.output_destination}/{args.model_tag}/shap_plot.png")
    df_result.write.format("parquet").save(f"gs://{args.output_bucket}/{args.output_destination}/{args.model_tag}/training_data_with_predictions", mode = args.write_mode)
    df_accuracy_55_result.write.format("parquet").save(f"gs://{args.output_bucket}/{args.output_destination}/{args.model_tag}/accuracy_55_test_result", mode = args.write_mode)
    upload_pickled_model(best_model, args.output_bucket, f"{args.output_destination}/{args.model_tag}/best_model.pkl")
    spark.stop()