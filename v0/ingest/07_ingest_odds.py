import argparse
import pyspark
import requests
from pyspark.sql.types import *
import pyspark.sql.functions as f
import pandas as pd
import numpy as np
import pandas as pd
from google.cloud import storage
from datetime import datetime, timedelta

def display_odds_data(odds_data):
    """Convert odds data to a pandas DataFrame and display it."""
    df = pd.DataFrame(odds_data)
    if not df.empty:
        return df
    else:
        return "No data available."

def fetch_historical_odds(api_key, sport, regions, markets, odds_format, date):
    """Fetch historical odds for a specific sport from The Odds API."""
    url = f'https://api.the-odds-api.com/v4/historical/sports/{sport}/odds'
    params = {
        'apiKey': api_key,
        'regions': regions,
        'markets': markets,
        'oddsFormat': odds_format,
        'date': date,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json(), response.headers  # Return the odds data and headers for quota info
    else:
        raise Exception(f'Failed to get historical odds: status_code {response.status_code}, response body {response.text}')

def fetch_historical_events(api_key, sport, date):
    """Fetch historical events for a specific sport from The Odds API."""
    url = f'https://api.the-odds-api.com/v4/historical/sports/{sport}/events'
    params = {
        'apiKey': api_key,
        'date': date,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json(), response.headers  # Return the events data and headers for quota info
    else:
        raise Exception(f'Failed to get historical events: status_code {response.status_code}, response body {response.text}')

def fetch_historical_event_odds(api_key, sport, event_id, regions, markets, date_format, odds_format, date):
    """
    Fetch historical odds for a specific sport from The Odds API.

    Parameters:
    - api_key (str): Your API key for The Odds API.
    - sport (str): The sport for which to fetch odds.
    - event_id (str): The event ID for the specific game or match.
    - regions (str): The regions from which to fetch odds.
    - markets (str): The markets to fetch odds for.
    - date_format (str): The format of the date.
    - odds_format (str): The format of the odds.
    - date (str): The date for which to fetch historical odds.

    Returns:
    - dict: The odds data.
    - dict: The response headers (including quota information).
    """
    url = f'https://api.the-odds-api.com/v4/historical/sports/{sport}/events/{event_id}/odds'
    params = {
        'apiKey': api_key,
        'regions': regions,
        'markets': markets,
        'dateFormat': date_format,
        'oddsFormat': odds_format,
        'date': date,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json(), response.headers  # Return the odds data and headers for quota info
    else:
        raise Exception(f'Failed to get historical odds: status_code {response.status_code}, response body {response.text}')

def parse_mlb_odds(df, column='data'):
    def parse_dict(data_str):
        if isinstance(data_str, str):
            try:
                data_dict = ast.literal_eval(data_str.replace('=', ':'))
                return data_dict
            except (SyntaxError, ValueError):
                return None
        return data_str

    df['parsed_data'] = df[column].apply(parse_dict)
    
    df['sport_key'] = df['parsed_data'].apply(lambda x: x.get('sport_key') if x else None)
    df['sport_title'] = df['parsed_data'].apply(lambda x: x.get('sport_title') if x else None)
    df['away_team'] = df['parsed_data'].apply(lambda x: x.get('away_team') if x else None)
    df['home_team'] = df['parsed_data'].apply(lambda x: x.get('home_team') if x else None)
    df['id'] = df['parsed_data'].apply(lambda x: x.get('id') if x else None)
    df['commence_time'] = df['parsed_data'].apply(lambda x: x.get('commence_time') if x else None)
    
    def extract_bookmakers(data_dict):
        if not data_dict:
            return []
        return data_dict.get('bookmakers', [])
    
    df['bookmakers'] = df['parsed_data'].apply(extract_bookmakers)
    df.drop(columns=['parsed_data'], inplace=True)
    
    return df

def parse_and_unpivot_bookmakers(df, column='bookmakers'):
    def parse_markets(data_str):
        if isinstance(data_str, str):
            try:
                data_dict = ast.literal_eval(data_str.replace('=', ':'))
                return data_dict
            except (SyntaxError, ValueError):
                return None
        return data_str
    
    records = []
    
    for idx, row in df.iterrows():
        bookmakers = row[column]
        if isinstance(bookmakers, list):
            for bookmaker in bookmakers:
                markets = parse_markets(bookmaker.get('markets'))
                if isinstance(markets, list):
                    for market in markets:
                        outcomes = market.get('outcomes', [])
                        for outcome in outcomes:
                            record = {
                                'sport_key': row['sport_key'],
                                'sport_title': row['sport_title'],
                                'away_team': row['away_team'],
                                'home_team': row['home_team'],
                                'id': row['id'],
                                'commence_time': row['commence_time'],
                                'bookmaker_title': bookmaker.get('title'),
                                'bookmaker_key': bookmaker.get('key'),
                                'bookmaker_last_update': bookmaker.get('last_update'),
                                'market_key': market.get('key'),
                                'market_last_update': market.get('last_update'),
                                'outcome_name': outcome.get('name'),
                                'outcome_price': outcome.get('price')
                            }
                            records.append(record)
    
    unpivoted_df = pd.DataFrame(records)
    return unpivoted_df

def get_odds_history_df(api_key, sport, regions, markets, odds_format, game_dates):
    odds_hist_dfs = []
    for date in game_dates:
        odds_data, headers = fetch_historical_odds(api_key=api_key, sport=sport, regions=regions, markets=markets, odds_format=odds_format, date=date)
        odds_df = display_odds_data(odds_data)
        odds_hist_dfs.append(odds_df)

    odds_df_hist = pd.concat(odds_hist_dfs, axis=0, ignore_index=True)
    odds_df_hist.reset_index(drop=True, inplace=True)

    df = parse_mlb_odds(odds_df_hist)
    unpivoted_df = parse_and_unpivot_bookmakers(df)
    
    return spark.createDataFrame(unpivoted_df)

def generate_game_dates(mlb_season_dates):
    all_game_dates = {}
    
    for year, dates in mlb_season_dates.items():
        start_date = datetime.strptime(dates["first_game"], "%Y-%m-%d")
        end_date = datetime.strptime(dates["last_game"], "%Y-%m-%d")
        
        game_dates = []
        current_date = start_date
        while current_date <= end_date:
            game_dates.append(current_date.strftime("%Y-%m-%dT10:00:00Z"))
            current_date += timedelta(days=1)
        
        all_game_dates[year] = game_dates
    
    return all_game_dates

def pivot_betting_data_by_book(df, bookmaker_keys):
    # Filter the dataframe to include only specified bookmaker keys
    filtered_df = df[df['bookmaker_key'].isin(bookmaker_keys)]

    # Pivot the dataframe to have one game/outcome per row
    pivot_df = filtered_df.pivot_table(
        index=['id', 'commence_time', 'away_team', 'home_team','outcome_name'],
        columns='bookmaker_key',
        values='outcome_price',
        aggfunc='first'
    ).reset_index()

    # Flatten the column hierarchy after pivoting
    pivot_df.columns.name = None

    return pivot_df

def get_mlb_season_dates():
    return {
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MLB Xgboost inputs")
    parser.add_argument("--first_game_date", type=str, help="")
    parser.add_argument("--last_game_date", type=str, help="")
    parser.add_argument("--output_bucket", type=str, help="a GCS Bucket")
    parser.add_argument("--output_destination", type=str, help="a location within GCS bucket where output is stored")
    parser.add_argument("--write_mode", type=str, help="overwrite or append, as used in spark.write.*")
    args = parser.parse_args()


    API_KEY = '80072a3664e0dbedd5bbd62c4797b579'#'4deab10a7b5f882011003a8bd936a341'
    SPORT = 'baseball_mlb'
    REGIONS = 'us'  # Can be 'uk', 'us', 'eu', 'au', or combinations thereof
    MARKETS = 'h2h'#,spreads,totals'  # Can be 'h2h', 'spreads', 'totals', etc.
    ODDS_FORMAT = 'american'  # 'decimal' or 'american'
    DATE_FORMAT = 'iso'  # 'iso' or 'unix'


    # mlb_season_dates = get_mlb_season_dates()
    dates = {"custom":{
        "first_game": args.first_game_date,
        "last_game": args.last_game_date
    }}

    spark = pyspark.sql.SparkSession.builder \
        .appName("") \
        .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

    all_game_dates = generate_game_dates(dates)
    print(all_game_dates)
    df_odds_history = get_odds_history_df( \
        API_KEY, SPORT, REGIONS, MARKETS, \
            ODDS_FORMAT, all_game_dates['custom'])

    bookmaker_keys = ['mybookieag', 'betmgm','draftkings','betrivers','fanduel','bet365']
    odds_p = df_odds_history.toPandas()
    odds_p_p = pivot_betting_data_by_book(odds_p, bookmaker_keys)
    away_p_p = odds_p_p[odds_p_p.outcome_name==odds_p_p.away_team]
    home_p_p = odds_p_p[odds_p_p.outcome_name==odds_p_p.home_team]

    combined_odds_df = spark.createDataFrame(pd.merge(
        away_p_p,
        home_p_p,
        on=['id', 'commence_time', 'away_team', 'home_team'],
        suffixes=('_away', '_home')
    ))

    combined_odds_df.write.format("parquet").save(f"gs://{args.output_bucket}/{args.output_destination}/odds_history", mode = args.write_mode)

    spark.stop()