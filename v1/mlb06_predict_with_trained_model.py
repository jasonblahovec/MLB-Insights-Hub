import argparse
import pyspark

from pyspark.sql.types import *
import pyspark.sql.functions as f
import xgboost as xgb
import pickle
from google.cloud import storage

def fetch_pickled_model(bucket_name, source_path):
    """
    Fetches and unpickles the model from the specified Google Cloud Storage path.

    :param bucket_name: The name of the GCS bucket.
    :param source_path: The source path within the GCS bucket where the pickled model is stored.
    :return: The unpickled model.
    """
    # Initialize a storage client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Create a blob object
    blob = bucket.blob(source_path)

    # Download the serialized model
    pickled_model = blob.download_as_string()

    # Deserialize the model
    model = pickle.loads(pickled_model)

    print(f"Model fetched from {bucket_name}/{source_path}.")

    return model 

def model_dataframe_to_dmatrix(model_dataframe):
    features = model_dataframe.iloc[:,2:-1]
    label = model_dataframe.iloc[:,-1]

    return xgb.DMatrix(features,label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MLB Xgboost inputs")
    parser.add_argument("--input_bucket", type=str, help="a GCS Bucket")
    parser.add_argument("--input_path", type=str, help="04_prepare.. output path")
    parser.add_argument("--model_bucket", type=str, help="a GCS Bucket")
    parser.add_argument("--model_path", type=str, help="04_prepare.. output path")
    parser.add_argument("--output_bucket", type=str, help="a GCS Bucket")
    parser.add_argument("--output_destination", type=str, help="a location within GCS bucket where output is stored")
    parser.add_argument("--model_tag", type=str, help="folder in output_dest for all output")
    parser.add_argument("--write_mode", type=str, help="overwrite or append, as used in spark.write.*")
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder \
        .appName("Get predictions for a dataset using an XGBoost Model") \
        .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

    input_df = spark.read.format("parquet").load(f'gs://{args.input_bucket}/{args.input_path}')
    dm =  model_dataframe_to_dmatrix(input_df.toPandas())
    model = fetch_pickled_model(bucket_name = args.model_bucket, source_path = args.model_path)

    preds = model.predict(dm)

    output_df = input_df.toPandas()

    output_df['model_prediction'] = preds

    df_result = spark.createDataFrame(output_df) \
        .withColumn("rounded",f.expr("round(model_prediction,1)")) \
        .withColumn('pred_home_win', f.expr("case when model_prediction >.5 then 1 else 0 end"))
    
    df_result.write.format("parquet").save(f"gs://{args.output_bucket}/{args.output_destination}/{args.model_tag}/game_predictions", mode = args.write_mode)

    spark.stop()