import argparse
import pyspark
from pyspark.sql.types import *
import pyspark.sql.functions as f
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest MLB Game info for a provided date range")
    parser.add_argument("--begin_date", type=str, help="First day in date range to process")
    parser.add_argument("--end_date", type=str, help="Last day in date range to process")
    parser.add_argument("--output_bucket", type=str, help="a GCS Bucket")
    parser.add_argument("--output_destination", type=str, help="a location within GCS bucket where output is stored")
    parser.add_argument("--write_mode", type=str, help="overwrite or append, as used in spark.write.*")
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder \
        .appName("Prepare features from MLB Game info for games in a provided date range") \
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

    spark.sparkContext.addPyFile("gs://dataproc-init-01/source/feature_engineering.py")
    from feature_engineering import prefix_columns, get_current_batting_ranks, get_pitcher_performance
    season = args.begin_date.split('-')[0]

    # Create a date range
    date_range = pd.date_range(start=args.begin_date, end=args.end_date)
    formatted_dates = [x.strftime('%Y-%m-%d') for x in date_range]

    df_individual_pitching = spark.read.format('csv').load('gs://mlb_api_extracts/mlb_api_output/individual_pitching.csv', header = True, inferSchema = True)
    df_mlb_games = spark.read.format('csv').load('gs://mlb_api_extracts/mlb_api_output/games_history.csv', header = True, inferSchema = True)
    df_team_batting = spark.read.format('csv').load('gs://mlb_api_extracts/mlb_api_output/team_batting.csv', header = True, inferSchema = True)
    df_team_pitching = spark.read.format('csv').load('gs://mlb_api_extracts/mlb_api_output/team_pitching.csv', header = True, inferSchema = True)
    df_individual_batting = spark.read.format('csv').load('gs://mlb_api_extracts/mlb_api_output/individual_batting.csv', header = True, inferSchema = True)

    print(df_mlb_games.count(), df_team_batting.count(), df_individual_pitching.count())

    output = {}
    # Loop over the dates
    for date in formatted_dates:
        print(date)
        df_todays_games = df_mlb_games.where(f.expr(f"season = {season}")).where(f.expr(f"officialdate = '{date}' and DetailedState  ='Final'"))

        df_current_batting = get_current_batting_ranks(df_team_batting.where(f.expr(f"season = {season}")),game_date = date)
        df_current_pitching = get_pitcher_performance(df_individual_pitching.where(f.expr(f"season = {season}")).where(f.expr("detailedstate = 'Final'")), game_date=date)

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

        print(df_model_input_daily.count())
        output[date] = df_model_input_daily

    union_df = None

    for date, df in output.items():
        if union_df is None:
            union_df = df
        else:
            union_df = union_df.union(df)

    output_df = union_df[union_df.columns[2:]]
    output_df.write.format("parquet").save(f"gs://{args.output_bucket}/{args.output_destination}", mode = args.write_mode)

    spark.stop()