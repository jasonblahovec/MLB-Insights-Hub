import argparse
import pyspark

from pyspark.sql.types import *
import pyspark.sql.functions as f

import pandas as pd

def get_baseball_data_fields():
    """
    Returns a list of baseball data fields with documentation on each column's meaning.
    
    Columns:
    - GamePK: Primary key of the game.
    - GameId: Unique identifier for the game.
    - GameNumber: The sequence number of the game in a series (e.g., Game 1, Game 2).
    - VenueName: Name of the venue where the game is played.
    - Season: The season year.
    - OfficialDate: Official date of the game.
    - home_officialdate: Date of last game for the home team.
    - away_officialdate: Date of last game for the away team.
    - DayNight: Indicator if the game is a day or night game.
    
    Team Performance:
    - AwayRuns: Total runs scored by the away team.
    - HomeRuns: Total runs scored by the home team.
    
    Team Information:
    - AwayTeamId: Unique identifier for the away team.
    - AwayTeamShortName: Short name of the away team.
    - AwayProbPitcherId: Unique identifier for the away team's probable pitcher.
    - AwayProbPitcherFullName: Full name of the away team's probable pitcher.
    - AwayWinningPercentage: Winning percentage of the away team.
    - AwayWins: Total wins by the away team.
    - AwayLosses: Total losses by the away team.
    - HomeTeamId: Unique identifier for the home team.
    - HomeTeamShortName: Short name of the home team.
    - HomeProbPitcherId: Unique identifier for the home team's probable pitcher.
    - HomeProbPitcherFullName: Full name of the home team's probable pitcher.
    - HomeWinningPercentage: Winning percentage of the home team.
    - HomeWins: Total wins by the home team.
    - HomeLosses: Total losses by the home team.
    
    Recent Team Performance (Home):
    - home_runs_last_1: Home team runs in the last 1 game.
    - home_batting_avg_last_1: Home team batting average in the last 1 game.
    - home_strikeouts_last_1: Home team strikeouts in the last 1 game.
    - home_walks_last_1: Home team walks in the last 1 game.
    - home_runs_last_3: Home team runs in the last 3 games.
    - home_batting_avg_last_3: Home team batting average in the last 3 games.
    - home_strikeouts_last_3: Home team strikeouts in the last 3 games.
    - home_walks_last_3: Home team walks in the last 3 games.
    - home_runs_last_7: Home team runs in the last 7 games.
    - home_batting_avg_last_7: Home team batting average in the last 7 games.
    - home_strikeouts_last_7: Home team strikeouts in the last 7 games.
    - home_walks_last_7: Home team walks in the last 7 games.
    - home_runs_last_15: Home team runs in the last 15 games.
    - home_batting_avg_last_15: Home team batting average in the last 15 games.
    - home_strikeouts_last_15: Home team strikeouts in the last 15 games.
    - home_walks_last_15: Home team walks in the last 15 games.
    
    Home Team Rankings:
    - home_rank_runs_last_1: Home team rank in runs over the last 1 game.
    - home_rank_batting_avg_last_1: Home team rank in batting average over the last 1 game.
    - home_rank_strikeouts_last_1: Home team rank in strikeouts over the last 1 game.
    - home_rank_walks_last_1: Home team rank in walks over the last 1 game.
    - home_rank_runs_last_3: Home team rank in runs over the last 3 games.
    - home_rank_batting_avg_last_3: Home team rank in batting average over the last 3 games.
    - home_rank_strikeouts_last_3: Home team rank in strikeouts over the last 3 games.
    - home_rank_walks_last_3: Home team rank in walks over the last 3 games.
    - home_rank_runs_last_7: Home team rank in runs over the last 7 games.
    - home_rank_batting_avg_last_7: Home team rank in batting average over the last 7 games.
    - home_rank_strikeouts_last_7: Home team rank in strikeouts over the last 7 games.
    - home_rank_walks_last_7: Home team rank in walks over the last 7 games.
    - home_rank_runs_last_15: Home team rank in runs over the last 15 games.
    - home_rank_batting_avg_last_15: Home team rank in batting average over the last 15 games.
    - home_rank_strikeouts_last_15: Home team rank in strikeouts over the last 15 games.
    - home_rank_walks_last_15: Home team rank in walks over the last 15 games.
    
    Home Starting Pitcher Performance:
    - home_sp_player_name: Name of the home team's starting pitcher.
    - home_sp_officialdate: Official date for the home team's starting pitcher's last appearance.
    - home_sp_strikeouts_last_1: Home team's starting pitcher strikeouts in the last 1 game.
    - home_sp_strikeouts_last_3: Home team's starting pitcher strikeouts in the last 3 games.
    - home_sp_strikeouts_last_7: Home team's starting pitcher strikeouts in the last 7 games.
    - home_sp_strikeouts_last_15: Home team's starting pitcher strikeouts in the last 15 games.
    - home_sp_runs_last_1: Home team's starting pitcher runs allowed in the last 1 game.
    - home_sp_runs_last_3: Home team's starting pitcher runs allowed in the last 3 games.
    - home_sp_runs_last_7: Home team's starting pitcher runs allowed in the last 7 games.
    - home_sp_runs_last_15: Home team's starting pitcher runs allowed in the last 15 games.
    
    Recent Team Performance (Away):
    - away_runs_last_1: Away team runs in the last 1 game.
    - away_batting_avg_last_1: Away team batting average in the last 1 game.
    - away_strikeouts_last_1: Away team strikeouts in the last 1 game.
    - away_walks_last_1: Away team walks in the last 1 game.
    - away_runs_last_3: Away team runs in the last 3 games.
    - away_batting_avg_last_3: Away team batting average in the last 3 games.
    - away_strikeouts_last_3: Away team strikeouts in the last 3 games.
    - away_walks_last_3: Away team walks in the last 3 games.
    - away_runs_last_7: Away team runs in the last 7 games.
    - away_batting_avg_last_7: Away team batting average in the last 7 games.
    - away_strikeouts_last_7: Away team strikeouts in the last 7 games.
    - away_walks_last_7: Away team walks in the last 7 games.
    - away_runs_last_15: Away team runs in the last 15 games.
    - away_batting_avg_last_15: Away team batting average in the last 15 games.
    - away_strikeouts_last_15: Away team strikeouts in the last 15 games.
    - away_walks_last_15: Away team walks in the last 15 games.
    
    Away Team Rankings:
    - away_rank_runs_last_1: Away team rank in runs over the last 1 game.
    - away_rank_batting_avg_last_1: Away team rank in batting average over the last 1 game.
    - away_rank_strikeouts_last_1: Away team rank in strikeouts over the last 1 game.
    - away_rank_walks_last_1: Away team rank in walks over the last 1 game.
    - away_rank_runs_last_3: Away team rank in runs over the last 3 games.
    - away_rank_batting_avg_last_3: Away team rank in batting average over the last 3 games.
    - away_rank_strikeouts_last_3: Away team rank in strikeouts over the last 3 games.
    - away_rank_walks_last_3: Away team rank in walks over the last 3 games.
    - away_rank_runs_last_7: Away team rank in runs over the last 7 games.
    - away_rank_batting_avg_last_7: Away team rank in batting average over the last 7 games.
    - away_rank_strikeouts_last_7: Away team rank in strikeouts over the last 7 games.
    - away_rank_walks_last_7: Away team rank in walks over the last 7 games.
    - away_rank_runs_last_15: Away team rank in runs over the last 15 games.
    - away_rank_batting_avg_last_15: Away team rank in batting average over the last 15 games.
    - away_rank_strikeouts_last_15: Away team rank in strikeouts over the last 15 games.
    - away_rank_walks_last_15: Away team rank in walks over the last 15 games.
    
    Away Starting Pitcher Performance:
    - away_sp_player_name: Name of the away team's starting pitcher.
    - away_sp_officialdate: Official date for the away team's starting pitcher's last appearance.
    - away_sp_strikeouts_last_1: Away team's starting pitcher strikeouts in the last 1 game.
    - away_sp_strikeouts_last_3: Away team's starting pitcher strikeouts in the last 3 games.
    - away_sp_strikeouts_last_7: Away team's starting pitcher strikeouts in the last 7 games.
    - away_sp_strikeouts_last_15: Away team's starting pitcher strikeouts in the last 15 games.
    - away_sp_runs_last_1: Away team's starting pitcher runs allowed in the last 1 game.
    - away_sp_runs_last_3: Away team's starting pitcher runs allowed in the last 3 games.
    - away_sp_runs_last_7: Away team's starting pitcher runs allowed in the last 7 games.
    - away_sp_runs_last_15: Away team's starting pitcher runs allowed in the last 15 games.
    """
    return [
        "GamePK", "GameId", "GameNumber", "VenueName", "Season", "OfficialDate", 
        "home_officialdate", "away_officialdate", "DayNight", "AwayRuns", "HomeRuns", 
        "AwayTeamId", "AwayTeamShortName", "AwayProbPitcherId", "AwayProbPitcherFullName", 
        "AwayWinningPercentage", "AwayWins", "AwayLosses", "HomeTeamId", 
        "HomeTeamShortName", "HomeProbPitcherId", "HomeProbPitcherFullName", 
        "HomeWinningPercentage", "HomeWins", "HomeLosses", 
        
        "home_runs_last_1", 
        "home_batting_avg_last_1", "home_strikeouts_last_1", "home_walks_last_1", 
        "home_runs_last_3", "home_batting_avg_last_3", "home_strikeouts_last_3", 
        "home_walks_last_3", "home_runs_last_7", "home_batting_avg_last_7", 
        "home_strikeouts_last_7", "home_walks_last_7", "home_runs_last_15", 
        "home_batting_avg_last_15", "home_strikeouts_last_15", "home_walks_last_15", 
        "home_rank_runs_last_1", "home_rank_batting_avg_last_1", 
        "home_rank_strikeouts_last_1", "home_rank_walks_last_1", 
        "home_rank_runs_last_3", "home_rank_batting_avg_last_3", 
        "home_rank_strikeouts_last_3", "home_rank_walks_last_3", 
        "home_rank_runs_last_7", "home_rank_batting_avg_last_7", 
        "home_rank_strikeouts_last_7", "home_rank_walks_last_7", 
        "home_rank_runs_last_15", "home_rank_batting_avg_last_15", 
        "home_rank_strikeouts_last_15", "home_rank_walks_last_15", 

        "away_sp_player_name", "away_sp_officialdate", "away_sp_strikeouts_last_1", 
        "away_sp_strikeouts_last_3", "away_sp_strikeouts_last_7", "away_sp_runs_last_1", "away_sp_runs_last_3", 
        "away_sp_runs_last_7",

        "away_runs_last_1", 
        "away_batting_avg_last_1", "away_strikeouts_last_1", "away_walks_last_1", 
        "away_runs_last_3", "away_batting_avg_last_3", "away_strikeouts_last_3", 
        "away_walks_last_3", "away_runs_last_7", "away_batting_avg_last_7", 
        "away_strikeouts_last_7", "away_walks_last_7", "away_runs_last_15", 
        "away_batting_avg_last_15", "away_strikeouts_last_15", "away_walks_last_15", 
        "away_rank_runs_last_1", "away_rank_batting_avg_last_1", 
        "away_rank_strikeouts_last_1", "away_rank_walks_last_1", 
        "away_rank_runs_last_3", "away_rank_batting_avg_last_3", 
        "away_rank_strikeouts_last_3", "away_rank_walks_last_3", 
        "away_rank_runs_last_7", "away_rank_batting_avg_last_7", 
        "away_rank_strikeouts_last_7", "away_rank_walks_last_7", 
        "away_rank_runs_last_15", "away_rank_batting_avg_last_15", 
        "away_rank_strikeouts_last_15", "away_rank_walks_last_15",

        "home_sp_player_name", "home_sp_officialdate", "home_sp_strikeouts_last_1", 
        "home_sp_strikeouts_last_3", "home_sp_strikeouts_last_7", "home_sp_runs_last_1", "home_sp_runs_last_3", 
        "home_sp_runs_last_7", 


    ]

def get_home_vs_away_models_v0(reporting_table):
    """
    Accepts the reporting table and outputs a table of ML features for XGBoost models 
    to predict AwayRuns and HomeRuns separately.

    Args:
        reporting_table (DataFrame): Input Spark DataFrame with raw data.

    Returns:
        (DataFrame, DataFrame): Two DataFrames for AwayRuns and HomeRuns models.
    """
    
    # Ensure all necessary columns are selected
    columns = get_baseball_data_fields()
    df = reporting_table.select(columns).na.drop()
    
    # Generate additional features if needed (e.g., ratios, interactions, etc.)
    # Example: Home team win/loss ratio, away team win/loss ratio
    df = df.withColumn("home_win_loss_ratio", f.col("HomeWins") / (f.col("HomeLosses") + 1))
    df = df.withColumn("away_win_loss_ratio", f.col("AwayWins") / (f.col("AwayLosses") + 1))
    df = df.withColumn("home_to_away_wins", f.col("HomeWins") / (f.col("AwayWins") + 1))
    df = df.withColumn("home_to_away_losses", f.col("HomeLosses") / (f.col("AwayLosses") + 1))
    df = df.withColumn("home_sp_days_rest", f.datediff(df["OfficialDate"], df["home_sp_officialdate"]))
    df = df.withColumn("away_sp_days_rest", f.datediff(df["OfficialDate"], df["away_sp_officialdate"]))
    df = df.withColumn("game_month", f.expr("month(OfficialDate)"))
    df = df.withColumn("home_days_rest", f.datediff(df["OfficialDate"], df["home_officialdate"]))
    df = df.withColumn("away_days_rest", f.datediff(df["OfficialDate"], df["away_officialdate"]))
    
    id_link = ["GamePK","season"]

    features_both = [
        # 'AwayWinningPercentage'
        # ,'HomeWinningPercentage'
        # ,'home_to_away_wins'
        # ,'home_to_away_losses'
        'home_win_loss_ratio'
        ,'away_win_loss_ratio'
        ,'GameNumber'
        # # ,'VenueName'
        ,'game_month'
        ,'home_days_rest'
        ,'away_days_rest'
        # ,'dayNight'

    ]

    # Select features and target variable for AwayRuns model
    features_away = df.select(id_link+features_both+[
        "away_runs_last_1"
        , "away_batting_avg_last_1"
        , "away_strikeouts_last_1"
        , "away_walks_last_1"
        , "away_runs_last_3"
        , "away_batting_avg_last_3"
        , "away_strikeouts_last_3"
        , "away_walks_last_3"
        , "away_runs_last_7"
        , "away_batting_avg_last_7"
        , "away_strikeouts_last_7"
        , "away_walks_last_7"
        , "away_runs_last_15"
        , "away_batting_avg_last_15"
        , "away_strikeouts_last_15"
        , "away_walks_last_15"
        , "home_sp_strikeouts_last_1"
        , "home_sp_strikeouts_last_3"
        , "home_sp_strikeouts_last_7"
        , "home_sp_runs_last_1"
        , "home_sp_runs_last_3"
        , "home_sp_runs_last_7"
        , "home_sp_days_rest"
    ])
    target_away = df.select("AwayRuns")
    
    # Select features and target variable for HomeRuns model
    features_home = df.select(id_link+features_both+[
        "home_runs_last_1"
        , "home_batting_avg_last_1"
        , "home_strikeouts_last_1"
        , "home_walks_last_1"
        , "home_runs_last_3"
        , "home_batting_avg_last_3"
        , "home_strikeouts_last_3"
        , "home_walks_last_3"
        , "home_runs_last_7"
        , "home_batting_avg_last_7"
        , "home_strikeouts_last_7"
        , "home_walks_last_7"
        , "home_runs_last_15"
        , "home_batting_avg_last_15"
        , "home_strikeouts_last_15"
        , "home_walks_last_15"
        , "away_sp_strikeouts_last_1"
        , "away_sp_strikeouts_last_3"
        , "away_sp_strikeouts_last_7"
        , "away_sp_runs_last_1"
        , "away_sp_runs_last_3"
        , "away_sp_runs_last_7"
        , "away_sp_days_rest"
    ])
    target_home = df.select("HomeRuns")
    
    # Convert Spark DataFrames to Pandas DataFrames for XGBoost
    features_away_pd = features_away.toPandas()
    target_away_pd = target_away.toPandas()
    features_home_pd = features_home.toPandas()
    target_home_pd = target_home.toPandas()
    
    # Merge features and targets for final DataFrames
    away_data = pd.concat([features_away_pd, target_away_pd], axis=1)
    home_data = pd.concat([features_home_pd, target_home_pd], axis=1)
    
    return away_data, home_data

def get_combined_model_data(reporting_table):
    """
    Accepts the reporting table and outputs a DataFrame of ML features for an XGBoost model
    to predict whether the home team wins.

    Args:
        reporting_table (DataFrame): Input Spark DataFrame with raw data.

    Returns:
        DataFrame: DataFrame containing features and the target variable for the combined model.
    """
    
    # Ensure all necessary columns are selected
    columns = get_baseball_data_fields()
    df = reporting_table.select(columns).na.drop()
    
    # Generate additional features (e.g., ratios, interactions, etc.)
    df = df.withColumn("home_win_loss_ratio", f.col("HomeWins") / (f.col("HomeLosses") + 1))
    df = df.withColumn("away_win_loss_ratio", f.col("AwayWins") / (f.col("AwayLosses") + 1))
    df = df.withColumn("home_to_away_wins", f.col("HomeWins") / (f.col("AwayWins") + 1))
    df = df.withColumn("home_to_away_losses", f.col("HomeLosses") / (f.col("AwayLosses") + 1))
    df = df.withColumn("home_sp_days_rest", f.datediff(df["OfficialDate"], df["home_sp_officialdate"]))
    df = df.withColumn("away_sp_days_rest", f.datediff(df["OfficialDate"], df["away_sp_officialdate"]))
    df = df.withColumn("game_month", f.expr("month(OfficialDate)"))
    df = df.withColumn("home_days_rest", f.datediff(df["OfficialDate"], df["home_officialdate"]))
    df = df.withColumn("away_days_rest", f.datediff(df["OfficialDate"], df["away_officialdate"]))
    
    # Create the HomeTeamWin flag
    df = df.withColumn("HomeTeamWin", f.when(f.col("HomeRuns") > f.col("AwayRuns"), 1).otherwise(0))
    
    id_link = ["GamePK", "season"]

    features_combined = [
        # 'AwayWinningPercentage',
        'HomeWinningPercentage',
        'home_to_away_wins',
        # 'home_to_away_losses',
        'home_win_loss_ratio',
        # 'away_win_loss_ratio',
        'GameNumber',
        'game_month',
        'home_days_rest',
        'away_days_rest',
        "home_runs_last_1",
        "home_batting_avg_last_1",
        "home_strikeouts_last_1",
        "home_walks_last_1",
        "home_runs_last_3",
        "home_batting_avg_last_3",
        "home_strikeouts_last_3",
        "home_walks_last_3",
        "home_runs_last_7",
        "home_batting_avg_last_7",
        "home_strikeouts_last_7",
        "home_walks_last_7",
        "home_runs_last_15",
        "home_batting_avg_last_15",
        "home_strikeouts_last_15",
        "home_walks_last_15",
        "away_runs_last_1",
        "away_batting_avg_last_1",
        "away_strikeouts_last_1",
        "away_walks_last_1",
        "away_runs_last_3",
        "away_batting_avg_last_3",
        "away_strikeouts_last_3",
        "away_walks_last_3",
        "away_runs_last_7",
        "away_batting_avg_last_7",
        "away_strikeouts_last_7",
        "away_walks_last_7",
        "away_runs_last_15",
        "away_batting_avg_last_15",
        "away_strikeouts_last_15",
        "away_walks_last_15",
        "home_sp_strikeouts_last_1",
        "home_sp_strikeouts_last_3",
        "home_sp_strikeouts_last_7",
        "home_sp_runs_last_1",
        "home_sp_runs_last_3",
        "home_sp_runs_last_7",
        "home_sp_days_rest",
        "away_sp_strikeouts_last_1",
        "away_sp_strikeouts_last_3",
        "away_sp_strikeouts_last_7",
        "away_sp_runs_last_1",
        "away_sp_runs_last_3",
        "away_sp_runs_last_7",
        "away_sp_days_rest",
    ]
    
    # Select features and target variable for the combined model
    combined_features = df.select(id_link + features_combined)
    target_combined = df.select("HomeTeamWin")
    
    # Convert Spark DataFrame to Pandas DataFrame for XGBoost
    combined_features_pd = combined_features.toPandas()
    target_combined_pd = target_combined.toPandas()
    
    # Merge features and target for final DataFrame
    combined_data = pd.concat([combined_features_pd, target_combined_pd], axis=1)
    
    return combined_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MLB Xgboost inputs")
    parser.add_argument("--input_bucket", type=str, help="a GCS Bucket")
    parser.add_argument("--historical_path", type=str, help="path to input data - history")
    parser.add_argument("--scheduled_path", type=str, help="path to input data - scheduled")
    parser.add_argument("--output_bucket", type=str, help="a GCS Bucket")
    parser.add_argument("--output_destination", type=str, help="a location within GCS bucket where output is stored")
    parser.add_argument("--write_mode", type=str, help="overwrite or append, as used in spark.write.*")
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder \
        .appName("Prepare MLB XGBoost Model Inputs") \
        .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")


    df_team_batting_vs_starting_pitching = spark.read.format("parquet").load(f'gs://{args.input_bucket}/{args.historical_path}')
    df_scheduled = spark.read.format("parquet").load(f'gs://{args.input_bucket}/{args.scheduled_path}')

    reporting_table = df_team_batting_vs_starting_pitching[get_baseball_data_fields()].na.drop()

    # away_data, home_data = get_home_vs_away_models_v0(reporting_table)
    combined_model_data = get_combined_model_data(reporting_table)
    combined_model_data_scheduled = get_combined_model_data(df_scheduled)

    spark.createDataFrame(combined_model_data).write.format("parquet") \
        .save(f"gs://{args.output_bucket}/{args.output_destination}/model_home_away_train", mode = args.write_mode)
    spark.createDataFrame(combined_model_data_scheduled).write.format("parquet") \
        .save(f"gs://{args.output_bucket}/{args.output_destination}/model_home_away_predict", mode = args.write_mode)
    # spark.createDataFrame(home_data).write.format("parquet") \
    #     .save(f"gs://{args.output_bucket}/{args.output_destination}/model_home", mode = args.write_mode)
    # spark.createDataFrame(away_data).write.format("parquet") \
    #     .save(f"gs://{args.output_bucket}/{args.output_destination}/model_away", mode = args.write_mode)


    spark.stop()