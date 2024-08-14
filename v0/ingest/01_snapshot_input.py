import pyspark
from pyspark.sql.types import *


if __name__ == "__main__":
    spark = pyspark.sql.SparkSession.builder \
        .appName("Ingest MLB Game info for a provided date range") \
        .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

    df_individual_pitching = spark.read.format('csv').load('gs://mlb_api_extracts/mlb_api_output/individual_pitching.csv', header = True, inferSchema = True)
    df_mlb_games = spark.read.format('csv').load('gs://mlb_api_extracts/mlb_api_output/games_history.csv', header = True, inferSchema = True)
    df_team_batting = spark.read.format('csv').load('gs://mlb_api_extracts/mlb_api_output/team_batting.csv', header = True, inferSchema = True)
    df_team_pitching = spark.read.format('csv').load('gs://mlb_api_extracts/mlb_api_output/team_pitching.csv', header = True, inferSchema = True)
    df_individual_batting = spark.read.format('csv').load('gs://mlb_api_extracts/mlb_api_output/individual_batting.csv', header = True, inferSchema = True)

    df_individual_pitching[df_individual_pitching.columns[2:]].write.format("parquet").mode('overwrite').save('gs://mlb_api_extracts/mlb_api_output_parquet/individual_pitching')
    df_mlb_games[df_mlb_games.columns[2:]].write.format("parquet").mode('overwrite').save('gs://mlb_api_extracts/mlb_api_output_parquet/games_history')
    df_team_batting[df_team_batting.columns[2:]].write.format("parquet").mode('overwrite').save('gs://mlb_api_extracts/mlb_api_output_parquet/team_batting')
    df_team_pitching[df_team_pitching.columns[2:]].write.format("parquet").mode('overwrite').save('gs://mlb_api_extracts/mlb_api_output_parquet/team_pitching')
    df_individual_batting[df_individual_batting.columns[2:]].write.format("parquet").mode('overwrite').save('gs://mlb_api_extracts/mlb_api_output_parquet/individual_batting')

    spark.stop()