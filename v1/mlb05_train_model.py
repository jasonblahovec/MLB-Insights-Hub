import argparse
import pyspark

from pyspark.sql.types import *
import pyspark.sql.functions as f
import pandas as pd
import xgboost as xgb
import shap
import pandas as pd
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

def display_accuracy_55_test(sdf):
    """
    Run a one-sided test to determine if the model has a >55% success rate:
    """
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
        .appName("Train MLB Model") \
        .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

    spark.sparkContext.addPyFile("gs://dataproc-init-01/source/optuna_func.py")
    from optuna_func import optuna_mlb_study_binary

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


