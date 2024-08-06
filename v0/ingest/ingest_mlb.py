import pandas as pd
import argparse
# import pyspark

# from pyspark.sql.types import *
import mlbstatsapi
# import pandas as pd


def append_game_stats(game_cols, stats):
    stats['gamepk'] = game_cols[0]
    stats['id'] = game_cols[1]
    stats['gamenumber'] = game_cols[2]
    stats['season'] = game_cols[3]
    stats['officialdate'] = game_cols[4]
    stats['daynight'] = game_cols[5]
    stats['time'] = game_cols[6]
    stats['ampm'] = game_cols[7]
    stats['detailedstate'] = game_cols[8]
    return stats

def parse_pitching_teamstats(data, game_cols):
    parsed_data = []
    for team in data:
        team_id = team[0]
        team_name = team[1]
        stats = team[2]
        stats['team_id'] = team_id
        stats['team_name'] = team_name
        stats = append_game_stats(game_cols, stats)
        
        parsed_data.append(stats)
    
    df = pd.DataFrame(parsed_data)
    return df

def parse_input_pitcher(data, game_cols):
    parsed_data = []
    for player in data:
        player_id = player[0]
        player_name = player[1]
        stats = player[2]
        stats['player_id'] = player_id
        stats['player_name'] = player_name
        stats = append_game_stats(game_cols, stats)
        parsed_data.append(stats)
    
    df = pd.DataFrame(parsed_data)
    return df

def parse_batting_teamstats(data, game_cols):
    parsed_data = []
    for team in data:
        team_id = team[0]
        team_name = team[1]
        stats = team[2]
        stats['team_id'] = team_id
        stats['team_name'] = team_name
        stats = append_game_stats(game_cols, stats)
        
        parsed_data.append(stats)
    
    df = pd.DataFrame(parsed_data)
    return df

def parse_input_batter(data, game_cols):
    parsed_data = []
    for player in data:
        player_id = player[0]
        player_name = player[1]
        stats = player[2]
        stats['player_id'] = player_id
        stats['player_name'] = player_name
        stats = append_game_stats(game_cols, stats)
        parsed_data.append(stats)
    
    df = pd.DataFrame(parsed_data)
    return df


class MLBIngestHistory():
    
    def __init__(self, begin_date = '2024-03-25', end_date = '2024-06-28'):
        self.begin_date = begin_date
        self.end_date = end_date
        
        mlb = mlbstatsapi.Mlb()
        schedule = mlb.get_schedule(start_date = self.begin_date, end_date = self.end_date)
        self.games = {}
        self.failed_game_pks = []
        for date in schedule.dates:
            print(date)
            for game in date.games:
                try:
                    self.games[game.gamepk] = mlb.get_game(game_id = game.gamepk)
                except:
                    self.failed_game_pks.append(game.gamepk)
                    
        print(f"failed: {self.failed_game_pks}")
        
        self.df_batting_player, self.df_batting_team = self.prepare_batting()
        self.df_pitching_player, self.df_pitching_team = self.prepare_pitching()
        self.df_game_data = self.prepare_games()
        
    def prepare_batting(self):
        dfs_team = []
        dfs_batter = []
        for i,gamepk in enumerate(list(self.games.keys())):

            game = self.games[gamepk]
            game_cols = [game.gamepk
                ,game.gamedata.game.id
                ,game.gamedata.game.gamenumber
                ,game.gamedata.game.season
                ,game.gamedata.datetime.officialdate
                ,game.gamedata.datetime.daynight
                ,game.gamedata.datetime.time
                ,game.gamedata.datetime.ampm
                ,game.gamedata.status.detailedstate]
            list_team_batting = [[ \
                game.livedata.boxscore.teams.home.team.id, \
                game.livedata.boxscore.teams.home.team.name, \
                game.livedata.boxscore.teams.home.teamstats['batting']], \
                [game.livedata.boxscore.teams.away.team.id, \
                game.livedata.boxscore.teams.away.team.name, \
                game.livedata.boxscore.teams.away.teamstats['batting'], \
            ]]

            df_team = parse_batting_teamstats(list_team_batting, game_cols)

            home_data = (game.livedata.boxscore.teams.home.players)
            away_data = (game.livedata.boxscore.teams.away.players)

            home_batters = [home_data[x] for x in home_data]
            away_batters = [away_data[x] for x in away_data]

            home_out = [[x.person.id, x.person.fullname,x.stats['batting']] for x in home_batters if x.stats['batting']!= {}]
            away_out = [[x.person.id, x.person.fullname,x.stats['batting']] for x in away_batters if x.stats['batting']!= {}]

            df_home_batter = parse_input_batter(home_out, game_cols)
            df_away_batter = parse_input_batter(away_out, game_cols)

            df_home_batter['team_type'] = 'home'
            df_home_batter['team_name'] = game.livedata.boxscore.teams.home.team.name
            df_away_batter['team_type'] = 'away'
            df_away_batter['team_name'] = game.livedata.boxscore.teams.away.team.name

            dfs_team = dfs_team+[df_team]
            dfs_batter = dfs_batter+[df_home_batter]+[df_away_batter]
            
        return pd.concat(dfs_batter), pd.concat(dfs_team)
        
    def prepare_pitching(self):
        dfs_team_p = []
        dfs_pitcher = []
        for gamepk in list(self.games.keys()):
            game = self.games[gamepk]
            game_cols = [game.gamepk
                ,game.gamedata.game.id
                ,game.gamedata.game.gamenumber
                ,game.gamedata.game.season
                ,game.gamedata.datetime.officialdate
                ,game.gamedata.datetime.daynight
                ,game.gamedata.datetime.time
                ,game.gamedata.datetime.ampm
                ,game.gamedata.status.detailedstate]

            list_team_pitching = [[ \
                game.livedata.boxscore.teams.home.team.id, \
                game.livedata.boxscore.teams.home.team.name, \
                game.livedata.boxscore.teams.home.teamstats['pitching']], \
                [game.livedata.boxscore.teams.away.team.id, \
                game.livedata.boxscore.teams.away.team.name, \
                game.livedata.boxscore.teams.away.teamstats['pitching'], \
            ]]

            df_team_p = parse_pitching_teamstats(list_team_pitching, game_cols)


            home_data = (game.livedata.boxscore.teams.home.players)
            away_data = (game.livedata.boxscore.teams.away.players)
            list_home_pitchers = [home_data[x] for x in home_data if home_data[x].position.code=='1']
            list_away_pitchers = [away_data[x] for x in away_data if away_data[x].position.code=='1']

            home_out = [[x.person.id, x.person.fullname,x.stats['pitching']] for x in list_home_pitchers if x.stats['pitching']!= {}]
            away_out = [[x.person.id, x.person.fullname,x.stats['pitching']] for x in list_away_pitchers if x.stats['pitching']!= {}]

            df_home_pitcher = parse_input_pitcher(home_out, game_cols)
            df_away_pitcher = parse_input_pitcher(away_out, game_cols)   

            df_home_pitcher['team_type'] = 'home'
            df_home_pitcher['team_name'] = game.livedata.boxscore.teams.home.team.name
            df_away_pitcher['team_type'] = 'away'
            df_away_pitcher['team_name'] = game.livedata.boxscore.teams.away.team.name

            dfs_team_p = dfs_team_p+[df_team_p]
            dfs_pitcher = dfs_pitcher+[df_home_pitcher]+[df_away_pitcher]
            
        return pd.concat(dfs_pitcher), pd.concat(dfs_team_p)
        
    def prepare_games(self):
        game_data = []
        for i,gamepk in enumerate(list(self.games.keys())):

            game = self.games[gamepk]
            away_prob_pitcher = game.gamedata.probablepitchers.away
            home_prob_pitcher = game.gamedata.probablepitchers.home
            if away_prob_pitcher != {}:
                away_prob_pitcher_id = away_prob_pitcher.id
                away_prob_pitcher_fullname = away_prob_pitcher.fullname
            else:
                away_prob_pitcher_id = -999
                away_prob_pitcher_fullname = 'unannounced'   
            if home_prob_pitcher != {}:
                home_prob_pitcher_id = home_prob_pitcher.id
                home_prob_pitcher_fullname = home_prob_pitcher.fullname
            else:
                home_prob_pitcher_id = -999
                home_prob_pitcher_fullname = 'unannounced'  

            game_cols = [game.gamepk
                ,game.gamedata.game.id
                ,game.gamedata.game.gamenumber
                ,game.gamedata.venue.name
                ,game.gamedata.venue.location.city
                ,game.gamedata.venue.location.state
                ,game.gamedata.game.season
                ,game.gamedata.datetime.officialdate
                ,game.gamedata.datetime.daynight
                ,game.gamedata.datetime.time
                ,game.gamedata.datetime.ampm
                ,game.gamedata.teams.away.id
                ,game.gamedata.teams.away.name
                ,game.gamedata.teams.away.league.name
                ,game.gamedata.teams.away.shortname
                ,away_prob_pitcher_id
                ,away_prob_pitcher_fullname
                ,game.gamedata.teams.away.record.wins
                ,game.gamedata.teams.away.record.losses
                ,game.gamedata.teams.away.record.winningpercentage
                ,game.gamedata.teams.home.id
                ,game.gamedata.teams.home.name
                ,game.gamedata.teams.home.league.name
                ,game.gamedata.teams.home.shortname
                ,home_prob_pitcher_id
                ,home_prob_pitcher_fullname
                ,game.gamedata.teams.home.record.wins
                ,game.gamedata.teams.home.record.losses
                ,game.gamedata.teams.home.record.winningpercentage
                ,game.gamedata.status.detailedstate
                ,self.games[game.gamepk].livedata.boxscore.teams.home.teamstats['batting']['runs']
                ,self.games[game.gamepk].livedata.boxscore.teams.away.teamstats['batting']['runs']]
            game_data = game_data + [game_cols]


        # Define column headers
        columns = ['GamePK', 'GameID', 'GameNumber', 'VenueName', 'VenueCity', 'VenueState', 'Season', 'OfficialDate',
                   'DayNight', 'Time', 'AMPM', 'AwayTeamID', 'AwayTeamName', 'AwayLeagueName', 'AwayTeamShortName',
                   'AwayProbPitcherID', 'AwayProbPitcherFullName', 'AwayWins', 'AwayLosses', 'AwayWinningPercentage',
                   'HomeTeamID', 'HomeTeamName', 'HomeLeagueName', 'HomeTeamShortName', 'HomeProbPitcherID',
                   'HomeProbPitcherFullName', 'HomeWins', 'HomeLosses', 'HomeWinningPercentage', 'DetailedState',
                   'HomeRuns', 'AwayRuns']

        return pd.DataFrame(game_data, columns=columns)
    
class MLBIngestScheduled():
    
    def __init__(self, date = '2024-06-29'):
        self.date = date
        
        mlb = mlbstatsapi.Mlb()
        schedule = mlb.get_schedule(start_date = self.date, end_date = self.date)
        self.games = {}
        self.failed_game_pks = []
        for date in schedule.dates:
            for game in date.games:
                try:
                    self.games[game.gamepk] = mlb.get_game(game_id = game.gamepk)
                except:
                    self.failed_game_pks.append(game.gamepk)
                    
        print(f"failed: {self.failed_game_pks}")
        
        self.df_game_data = self.prepare_games()
        
    def prepare_games(self):
        game_data = []
        for i,gamepk in enumerate(list(self.games.keys())):

            game = self.games[gamepk]
            away_prob_pitcher = game.gamedata.probablepitchers.away
            home_prob_pitcher = game.gamedata.probablepitchers.home
            if away_prob_pitcher != {}:
                away_prob_pitcher_id = away_prob_pitcher.id
                away_prob_pitcher_fullname = away_prob_pitcher.fullname
            else:
                away_prob_pitcher_id = -999
                away_prob_pitcher_fullname = 'unannounced'   
            if home_prob_pitcher != {}:
                home_prob_pitcher_id = home_prob_pitcher.id
                home_prob_pitcher_fullname = home_prob_pitcher.fullname
            else:
                home_prob_pitcher_id = -999
                home_prob_pitcher_fullname = 'unannounced'  

            game_cols = [game.gamepk
                ,game.gamedata.game.id
                ,game.gamedata.game.gamenumber
                ,game.gamedata.venue.name
                ,game.gamedata.venue.location.city
                ,game.gamedata.venue.location.state
                ,game.gamedata.game.season
                ,game.gamedata.datetime.officialdate
                ,game.gamedata.datetime.daynight
                ,game.gamedata.datetime.time
                ,game.gamedata.datetime.ampm
                ,game.gamedata.teams.away.id
                ,game.gamedata.teams.away.name
                ,game.gamedata.teams.away.league.name
                ,game.gamedata.teams.away.shortname
                ,away_prob_pitcher_id
                ,away_prob_pitcher_fullname
                ,game.gamedata.teams.away.record.wins
                ,game.gamedata.teams.away.record.losses
                ,game.gamedata.teams.away.record.winningpercentage
                ,game.gamedata.teams.home.id
                ,game.gamedata.teams.home.name
                ,game.gamedata.teams.home.league.name
                ,game.gamedata.teams.home.shortname
                ,home_prob_pitcher_id
                ,home_prob_pitcher_fullname
                ,game.gamedata.teams.home.record.wins
                ,game.gamedata.teams.home.record.losses
                ,game.gamedata.teams.home.record.winningpercentage
                ,game.gamedata.status.detailedstate
                ,self.games[game.gamepk].livedata.boxscore.teams.home.teamstats['batting']['runs']
                ,self.games[game.gamepk].livedata.boxscore.teams.away.teamstats['batting']['runs']]
            game_data = game_data + [game_cols]


        # Define column headers
        columns = ['GamePK', 'GameID', 'GameNumber', 'VenueName', 'VenueCity', 'VenueState', 'Season', 'OfficialDate',
                   'DayNight', 'Time', 'AMPM', 'AwayTeamID', 'AwayTeamName', 'AwayLeagueName', 'AwayTeamShortName',
                   'AwayProbPitcherID', 'AwayProbPitcherFullName', 'AwayWins', 'AwayLosses', 'AwayWinningPercentage',
                   'HomeTeamID', 'HomeTeamName', 'HomeLeagueName', 'HomeTeamShortName', 'HomeProbPitcherID',
                   'HomeProbPitcherFullName', 'HomeWins', 'HomeLosses', 'HomeWinningPercentage', 'DetailedState',
                   'HomeRuns', 'AwayRuns']

        return pd.DataFrame(game_data, columns=columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest MLB Game info for a provided date range")
    parser.add_argument("--begin_date", type=int, help="First day in date range to process")
    parser.add_argument("--end_date", type=int, help="Last day in date range to process")
    parser.add_argument("--scheduled_date", type=int, help="Date from which to pull scheduled games")
    parser.add_argument("--output_bucket", type=str, help="a GCS Bucket")
    parser.add_argument("--output_destination", type=str, help="a location within GCS bucket where output is stored")
    # parser.add_argument("--write_mode", type=str, help="overwrite or append, as used in spark.write.*")
    args = parser.parse_args()

    # spark = pyspark.sql.SparkSession.builder \
    #     .appName("Ingest MLB Game info for a provided date range") \
    #     .getOrCreate()
    # spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

    bucket_name = args.output_bucket
    output_destination = args.output_destination
    # write_mode = args.write_mode


    mlb = MLBIngestHistory(begin_date = args.begin_date, end_date = args.end_date)

    mlb.df_batting_player.to_csv(f"gs://{bucket_name}/{output_destination}/individual_batting/mlb_individual_batting_history_{mlb.begin_date.replace('-','')}_{mlb.end_date.replace('-','')}.csv")
    mlb.df_batting_team.to_csv(f"gs://{bucket_name}/{output_destination}/team_batting/mlb_team_batting_history_{mlb.begin_date.replace('-','')}_{mlb.end_date.replace('-','')}.csv")
    mlb.df_pitching_player.to_csv(f"gs://{bucket_name}/{output_destination}/individual_pitching/mlb_individual_pitching_history_{mlb.begin_date.replace('-','')}_{mlb.end_date.replace('-','')}.csv")
    mlb.df_pitching_team.to_csv(f"gs://{bucket_name}/{output_destination}/team_pitching/mlb_team_pitching_history_{mlb.begin_date.replace('-','')}_{mlb.end_date.replace('-','')}.csv")
    mlb.df_game_data.to_csv(f"gs://{bucket_name}/{output_destination}/games_history/mlb_games_history_{mlb.begin_date.replace('-','')}_{mlb.end_date.replace('-','')}.csv")

    mlbs = MLBIngestScheduled(date = args.scheduled_date)
    mlbs.df_game_data.to_csv(f"gs://{bucket_name}/{output_destination}/scheduled/mlb_games_scheduled_{mlbs.date.replace('-','')}.csv")

    # spark.stop()