import argparse
import pyspark

from pyspark.sql.types import *
import pyspark.sql.functions as f
from pyspark.sql.window import Window
import pandas as pd
from pyspark.sql import DataFrame
import datetime as dt

def prefix_columns(prefix: str, df: DataFrame) -> DataFrame:
    # Generate a list of expressions to rename each column
    renamed_cols = [f.col(c).alias(f"{prefix}{c}") for c in df.columns]
    # Select the renamed columns
    prefixed_df = df.select(*renamed_cols)
    return prefixed_df

def get_current_batting_ranks(df_team_batting,game_date):
    df_filtered = df_team_batting \
        .withColumn("game_batting_avg", f.expr("hits/atbats")) \
        .withColumn("gamesback", f.expr("rank() over(partition by team_name, season order by officialdate desc)")) \
        .withColumn("walks", f.expr('baseonballs+intentionalwalks as walks')).distinct()

    # Function to calculate average runs and batting average over the last N games
    def calculate_averages(df, n):
        window_spec = Window.partitionBy("team_name","season").orderBy(f.col("officialdate").desc()).rowsBetween(Window.unboundedPreceding, Window.currentRow)
        
        df_avg = df \
            .withColumn(f"runs_last_{n}", f.avg("runs").over(window_spec.rowsBetween(Window.currentRow, Window.currentRow + n - 1))) \
            .withColumn(f"batting_avg_last_{n}", f.avg("game_batting_avg").over(window_spec.rowsBetween(Window.currentRow, Window.currentRow + n - 1))) \
            .withColumn(f"strikeouts_last_{n}", f.avg("strikeouts").over(window_spec.rowsBetween(Window.currentRow, Window.currentRow + n - 1))) \
            .withColumn(f"walks_last_{n}", f.avg("walks").over(window_spec.rowsBetween(Window.currentRow, Window.currentRow + n - 1))) \
        
        return df_avg

    # Apply the function for the last 1, 3, 7, and 15 games
    df_last_1 = calculate_averages(df_filtered, 1)
    df_last_3 = calculate_averages(df_last_1, 3)
    df_last_7 = calculate_averages(df_last_3, 7)
    df_last_15 = calculate_averages(df_last_7, 15)

    # Select the required columns for display
    df_result = df_last_15.select(
        'team_name', 
        'season',
        'officialdate', 
        'id',
        'runs', 
        'strikeouts',
        'baseonballs',
        'walks',
        'game_batting_avg',
        'runs_last_1', 
        'batting_avg_last_1', 
        'strikeouts_last_1', 
        'walks_last_1', 
        'runs_last_3', 
        'batting_avg_last_3', 
        'strikeouts_last_3', 
        'walks_last_3', 
        'runs_last_7', 
        'batting_avg_last_7', 
        'strikeouts_last_7', 
        'walks_last_7', 
        'runs_last_15', 
        'batting_avg_last_15',
        'strikeouts_last_15', 
        'walks_last_15'
    )

    # # Show the results
    df_current_batting = df_result.where(f.expr(f"officialdate < '{game_date}'")) \
        .withColumn("games_back", f.expr("rank() over(partition by team_name, season order by id desc)")) \
            .where(f.expr("games_back = 1")).drop('games_back')

    for stattype in ['batting_avg','runs','strikeouts','walks']:
        for timeframe in ['1','3','7','15']:
            df_current_batting = df_current_batting \
                .withColumn(f"rank_{stattype}_last_{timeframe}", f.expr(f"rank() over(partition by season order by {stattype}_last_{timeframe} desc)"))
                
    return df_current_batting.drop('runs').drop('strikeouts').drop('baseonballs').drop('walks').drop('game_batting_avg')

def get_pitcher_performance(df_player_pitching, game_date):
    df_filtered = df_player_pitching \
        .withColumn("game_date", f.to_date("officialdate")) \
        .withColumn("gamesback", f.expr("rank() over(partition by player_name, season order by officialdate desc)"))

    # Function to calculate average runs and strikeouts over the last N appearances
    def calculate_averages(df, n):
        window_spec = Window.partitionBy("player_name","season").orderBy(f.col("officialdate").desc()).rowsBetween(Window.unboundedPreceding, Window.currentRow)
        
        df_avg = df \
            .withColumn(f"strikeouts_last_{n}", f.avg("strikeouts").over(window_spec.rowsBetween(Window.currentRow, Window.currentRow + n - 1))) \
            .withColumn(f"runs_last_{n}", f.avg("runs").over(window_spec.rowsBetween(Window.currentRow, Window.currentRow + n - 1)))
        
        return df_avg

    # Apply the function for the last 1, 3, and 7 appearances
    df_last_1 = calculate_averages(df_filtered, 1)
    df_last_3 = calculate_averages(df_last_1, 3)
    df_last_7 = calculate_averages(df_last_3, 7)

    # Select the required columns for display
    df_result = df_last_7.select(
        'player_name',
        'season',
        'officialdate',
        'team_name',
        'strikeouts',
        'runs',
        'strikeouts_last_1',
        'runs_last_1',
        'strikeouts_last_3',
        'runs_last_3',
        'strikeouts_last_7',
        'runs_last_7'
    )

    # Filter for current performance up to the given game date
    df_current_pitching = df_result.where(f.expr(f"officialdate < '{game_date}'")) \
        .withColumn("games_back", f.expr("rank() over(partition by player_name, season order by officialdate desc)")) \
        .where(f.expr("games_back = 1")).drop('games_back')

    # Ranking the performance metrics
    for stattype in ['strikeouts', 'runs']:
        for timeframe in ['1', '3', '7']:
            df_current_pitching = df_current_pitching \
                .withColumn(f"rank_{stattype}_last_{timeframe}", f.expr(f"rank() over(order by {stattype}_last_{timeframe} desc)"))
                
    return df_current_pitching.drop('strikeouts').drop('runs')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest MLB Game info for a provided date range")
    parser.add_argument("--date", type=str, help="date to process")
    parser.add_argument("--output_bucket", type=str, help="a GCS Bucket")
    parser.add_argument("--output_destination", type=str, help="a location within GCS bucket where output is stored")
    parser.add_argument("--write_mode", type=str, help="overwrite or append, as used in spark.write.*")
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder \
        .appName("Ingest MLB Game info for a provided date range") \
        .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

    mlb_season_dates = {
        2021: {
            "first_game": "2021-04-01",
            "last_game": "2021-10-03"
        },
        2022: {
            "first_game": "2022-04-07",
            "last_game": "2022-10-05"
        },
        2023: {
            "first_game": "2023-03-30",
            "last_game": "2023-10-01"
        },
        2024: {
            "first_game": "2024-03-28",
            "last_game": "2024-09-29"
        }
    }
    # march 20 to sep 29 2019

    # # Define the start and end dates
    # start_date = '2024-04-08'
    # end_date = '2024-07-25'
    season = args.date.split('-')[0]

    df_individual_pitching = spark.read.format('csv').load('gs://mlb_api_extracts/mlb_api_output/individual_pitching.csv', header = True, inferSchema = True)
    df_mlb_games = spark.read.format('csv').load('gs://mlb_api_extracts/mlb_api_output/games_history.csv', header = True, inferSchema = True)
    df_team_batting = spark.read.format('csv').load('gs://mlb_api_extracts/mlb_api_output/team_batting.csv', header = True, inferSchema = True)
    df_team_pitching = spark.read.format('csv').load('gs://mlb_api_extracts/mlb_api_output/team_pitching.csv', header = True, inferSchema = True)
    df_individual_batting = spark.read.format('csv').load('gs://mlb_api_extracts/mlb_api_output/individual_batting.csv', header = True, inferSchema = True)
    df_todays_games = spark.read.format("csv").load("gs://mlb_api_extracts/mlb_api_output/scheduled.csv", header = True, inferSchema = True).where(f.expr(f"officialdate = '{args.date}'"))

    df_current_batting = get_current_batting_ranks(df_team_batting,game_date = args.date).where(f.expr("season = 2024"))
    df_current_pitching = get_pitcher_performance(df_individual_pitching.where(f.expr("detailedstate = 'Final'")), game_date=args.date).where(f.expr("season = 2024"))

    df_current_batting_home = prefix_columns('home_', df_current_batting)
    df_current_batting_away = prefix_columns('away_', df_current_batting)

    df_current_pitching_home = prefix_columns('home_sp_', df_current_pitching)
    df_current_pitching_away = prefix_columns('away_sp_', df_current_pitching)

    df_model_input_daily =  \
        df_todays_games \
        .join(df_current_batting_home, df_todays_games.HomeTeamName == df_current_batting_home.home_team_name, "left")\
        .join(df_current_batting_away, df_todays_games.AwayTeamName == df_current_batting_away.away_team_name, "left") \
        .join(df_current_pitching_home, df_todays_games.HomeProbPitcherFullName == df_current_pitching_home.home_sp_player_name, "left")\
        .join(df_current_pitching_away, df_todays_games.AwayProbPitcherFullName == df_current_pitching_away.away_sp_player_name, "left")

    df_model_input_daily[df_model_input_daily.columns[2:]] \
        .withColumn('batch', f.expr('current_timestamp()')) \
        .write.format("parquet").save(f"gs://{args.output_bucket}/{args.output_destination}" \
            , mode = args.write_mode)

    spark.stop()