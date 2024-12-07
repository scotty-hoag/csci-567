import pandas as pd
import shutil
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import makedata

from sklearn.model_selection import train_test_split
from itertools import combinations, product
from collections import Counter

INPUT_COLS_MATCHDATA = [
    "blueTeamTag",
    "redTeamTag",
    "bResult",
    "rResult", #For experimental purposes only. Remove after final implementation
    "blueTop",
    "blueTopChamp",
    "blueJungle",
    "blueJungleChamp",
    "blueMiddle",
    "blueMiddleChamp",
    "blueADC",
    "blueADCChamp",
    "blueSupport",
    "blueSupportChamp",
    "redTop",
    "redTopChamp",
    "redJungle",
    "redJungleChamp",
    "redMiddle",
    "redMiddleChamp",
    "redADC",
    "redADCChamp",
    "redSupport",
    "redSupportChamp"
]

#Map the 'Name' and 'Win Rate' columns from champion_data.csv to the corresponding fields in the matchinfo.csv columns
#as noted in the values fields below. The key represents the label of the engineered field as mentioned in the research
#paper.
WIN_RATE_CHAMPION_REPLACE_COLS = {
    'blueTopChamp': 'btPlayerChampion', 
    'blueJungleChamp': 'bjPlayerChampion', 
    'blueMiddleChamp': 'bmPlayerChampion', 
    'blueADCChamp': 'baPlayerChampion', 
    'blueSupportChamp': 'bsPlayerChampion',
    'redTopChamp': 'rtPlayerChampion', 
    'redJungleChamp': 'rjPlayerChampion', 
    'redMiddleChamp': 'rmPlayerChampion', 
    'redADCChamp': 'raPlayerChampion', 
    'redSupportChamp': 'rsPlayerChampion'
}

#Map the 'Name' and 'Win Rate' columns from player_data.csv to the corresponding fields in the matchinfo.csv columns
#as noted in the values fields below.
WIN_RATE_PLAYER_ROLE_REPLACE_COLS = {
    "blueTop": "btPlayerRole",
    "blueJungle": "bjPlayerRole",
    "blueMiddle": "bmPlayerRole",
    "blueADC": "baPlayerRole",
    "blueSupport": "bsPlayerRole",
    "redTop": "rtPlayerRole",
    "redJungle": "rjPlayerRole",
    "redMiddle": "rmPlayerRole",
    "redADC": "raPlayerRole",
    "redSupport": "rsPlayerRole"
}

DROPCOLS = [
    'blueTeamTag', 
    'redTeamTag', 
    'blueTop', 
    'blueTopChamp',
    'blueJungle', 
    'blueJungleChamp', 
    'blueMiddle', 
    'blueMiddleChamp',
    'blueADC', 
    'blueADCChamp', 
    'blueSupport', 
    'blueSupportChamp', 
    'redTop',
    'redTopChamp', 
    'redJungle', 
    'redJungleChamp', 
    'redMiddle',
    'redMiddleChamp', 
    'redADC', 
    'redADCChamp', 
    'redSupport',
    'redSupportChamp'
]
      
TEAM_COLOR = ["Blue", "Red"]
TEAM_ROLE = ["Top", "Jungle", "Middle", "ADC", "Support"]

def load_data_from_csv(bIsTrainingSet, bIsGeneratedSet=True, bGenerateOutputFile=False, bIncludeChampionRole_Feature=False):
    """
        Loads data from the feature CSV files and morphs the relevant fields into a dataframe for training use.

        Input:
            bIsTrainingSet - bool: If True, imports the data from the training folder into a dataframe. If False, use the 
                test folder as the import source.
            bGenerateOutputFile - bool: If True, generates a csv file representing the contents of the training data. Generated 
                CSV files will be located in the /feature_data directory named 'featureInput' or 'featureInput_zScoreNormalized'.
                If bIncludeChampionRole_Feature is set to False, then the files will be appended with '_noChampionRole'.
            bIncludeChampionRole_Feature - bool: If True, include the championRole features described in feature 5 of the research
                paper. Note that this feature is not used by the research paper as part of the training data.

        Return data:
            dataframe - A pandas dataframe whose columns are described in Table II (pg 178) of the Research Paper, in addition
                to the label column: bResult, which equals 1 if the blue team has won the match.

    """
    #Note: We assume that the folder structure will be consistent throughout the development process.
    dir_feature_data = "feature_data"
    
    if bIsGeneratedSet:
        setType = "temp_train" if bIsTrainingSet else "temp_test"
    else:
        setType = "train" if bIsTrainingSet else "test"

    file_match_data = "match_info.csv"
    file_data_champion = "champion_data.csv"
    file_champion_vs_data = "champion_vs_data.csv"
    file_player_data = "player_data.csv"
    file_player_vs_data = "player_vs_data.csv"
    file_player_wins_with_each_champion_data = "player_wins_with_each_champion_data.csv"
    file_team_data = "team_data.csv"
    file_match_vs_coop = "match_vs_coop_data.csv"

    match_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', f"{dir_feature_data}\{setType}\{file_match_data}")
    dir_data_champion = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', f"{dir_feature_data}\{setType}\{file_data_champion}")
    dir_player_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', f"{dir_feature_data}\{setType}\{file_player_data}")
    dir_player_wins_with_each_champion_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 
                                                           f"{dir_feature_data}\{setType}\{file_player_wins_with_each_champion_data}")
    dir_player_vs_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', f"{dir_feature_data}\{setType}\{file_player_vs_data}")
    dir_match_vs_coop = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', f"{dir_feature_data}\{setType}\{file_match_vs_coop}")
    dir_team_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', f"{dir_feature_data}\{setType}\{file_team_data}")
    
    #Load the initial data, then start replacing the fields with the engineered data.

    lt_input_cols = INPUT_COLS_MATCHDATA.copy()

    if "rResult" in INPUT_COLS_MATCHDATA:
        lt_input_cols.remove("rResult")

    match_data = pd.read_csv(match_data_path, usecols=lt_input_cols)
    champion_data = pd.read_csv(dir_data_champion)
    match_vs_coop_data = pd.read_csv(dir_match_vs_coop)
    player_data = pd.read_csv(dir_player_data)
    player_wr_champion_data = pd.read_csv(dir_player_wins_with_each_champion_data)
    player_vs_data = pd.read_csv(dir_player_vs_data)
    team_data = pd.read_csv(dir_team_data, index_col=False)

    #Remove extra white space before processing
    player_data.columns = player_data.columns.str.strip()
    champion_data.columns = champion_data.columns.str.strip()

    #Fill in 0s for NaN fields
    team_data = team_data.fillna(0.5)

    modified_match_df = match_data

    #Throw out the entries that don't have player data. The rows that do not have an entry for 'blueTop' also do not have
    #entries for the other roles. In addition, do the same for entries missing a 'blueTeamTag'
    # modified_match_df = modified_match_df.dropna(subset=['blueTeamTag', 'blueTop'], ignore_index=True) 

    #Implementation of Feature 1: playerRole
    for teamColor in TEAM_COLOR:
        for role in TEAM_ROLE:
            #Append 'Plays' if the role is 'Top'
            playLabel = "Top Plays" if role == "Top" else role
            
            #Commented-out code is the older implementation, for reference
            # player_data[f"wr{teamColor}{role}"] = (player_data[f"{teamColor} {role} Wins"] / player_data[f"{teamColor} {playLabel}"]).fillna(0.5)
            player_data[f"wr{teamColor}{role}"] = ((player_data[f"Blue {role} Wins"] + player_data[f"Red {role} Wins"]) 
                                                   / (player_data[f"Blue {playLabel}"] + player_data[f"Red {playLabel}"])).fillna(0.5)
                        
            player_data_map = player_data.set_index('Name')[f"wr{teamColor}{role}"].to_dict()
            
            labelName = WIN_RATE_PLAYER_ROLE_REPLACE_COLS[f"{teamColor.lower()}{role}"]
            modified_match_df[labelName] = modified_match_df[f"{teamColor.lower()}{role}"].map(player_data_map)
    
    #Implementation of Feature 2: playerChampion        
    player_wins_df = player_wr_champion_data.melt(
        id_vars=['Player Name'],
        var_name='Champion',
        value_name='WinRate'
    )

    for field, newField in WIN_RATE_CHAMPION_REPLACE_COLS.items():
        playerField = field.replace("Champ", "")
        role_merge = modified_match_df.merge(
            player_wins_df,
            left_on=[playerField, field],
            right_on=['Player Name', 'Champion'],
            how='left'            
        )

        modified_match_df[newField] = role_merge['WinRate'].fillna(0.5)
        
    #Implementation of Feature 3: coopPlayer_blue/red
    modified_match_df['bCoopPlayer'] = match_vs_coop_data['sumCoopBluePlayers']
    modified_match_df['rCoopPlayer'] = match_vs_coop_data['sumCoopRedPlayers']
        
    # #Implementation of Feature 4: vsPlayer
    modified_match_df['vsPlayer'] = match_vs_coop_data['sumVsBluePlayers']

    #Implementation of Feature 5: championRole
    # if bIncludeChampionRole_Feature:
    #     for teamColor in TEAM_COLOR:
    #         for role in TEAM_ROLE:
    #             #Append 'Plays' if the role is 'Top' to handle the column discrepancy.
    #             playLabel = "Top Plays" if role == "Top" else role

    #             # champion_data[f"wr{teamColor}{role}"] = (champion_data[f"{role} Wins"] / champion_data[playLabel]).fillna(0)
    #             champion_data[f"wr{teamColor}{role}"] = ((champion_data[f"Blue {role} Wins"] + champion_data[f"Red {role} Wins"]) 
    #                                                / (champion_data[f"Blue {playLabel}"] + champion_data[f"Red {playLabel}"])).fillna(0.5)
                
    #             champion_data_map = champion_data.set_index('Name')[f"wr{teamColor}{role}"].to_dict()

    #             # labelName = WIN_RATE_CHAMPION_REPLACE_COLS[f"{teamColor.lower()}{role}Champ"]
    #             labelName = f"{teamColor[0].lower()}{role[0].lower()}ChampionRole"
    #             modified_match_df[labelName] = modified_match_df[f"{teamColor.lower()}{role}Champ"].map(champion_data_map)
                
    #Implementation of Feature 6
    modified_match_df['bCoopChampion'] = match_vs_coop_data['sumCoopBlueChampions'].fillna(0.5)
    modified_match_df['rCoopChampion'] = match_vs_coop_data['sumCoopRedChampions'].fillna(0.5)
        
    #Implementation of Feature 7
    modified_match_df['vsChampion'] = match_vs_coop_data['sumVsBlueChampions'].fillna(0.5)
        
    #Implementation of Feature 8
    team_data['teamTag'] = team_data['teamTag'].astype(str)

    #Concept with respect to blueTeam only
    team_data_blue_map = team_data.set_index('teamTag')['Win Ratio Blue'].to_dict()
    modified_match_df['bTeamColor'] = modified_match_df['blueTeamTag'].map(team_data_blue_map)

    team_data_red_map = team_data.set_index('teamTag')['Win Ratio Red'].to_dict()
    modified_match_df['rTeamColor'] = modified_match_df['redTeamTag'].map(team_data_red_map)  

    #Drop all unnecessary cols from modified_match_df.
    modified_match_df.drop(columns=DROPCOLS, inplace=True)

    if bGenerateOutputFile:
         outputFileName = "featureInput{}.csv".format("" if bIncludeChampionRole_Feature else "_noChampionRole")
         dir_file_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', f"{dir_feature_data}\{setType}\{outputFileName}")
         modified_match_df.to_csv(dir_file_output, index=False)
         print("Wrote data file: {}".format(dir_file_output))

    return modified_match_df

def load_matchdata_into_df(dirMatchData):
    match_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', f"feature_data\\{dirMatchData}\\match_info.csv")
    match_data = pd.read_csv(match_data_path, usecols=INPUT_COLS_MATCHDATA)

    #Extract the label from the match_data dataframe
    full_df = match_data.drop(columns='bResult')
    y_data_full_df = match_data['bResult']

    x_train, x_test, y_train, y_test = train_test_split(full_df, y_data_full_df, test_size=0.1, stratify=y_data_full_df, random_state=42)

    x_train_player_data = generate_playerData_df(x_train)
    x_test_player_data = generate_playerData_df(x_test)

    # x_train_player_vs_data = generate_playerData_df(x_train)

    x_train_player_champion_winRate = generate_player_champion_winRate(x_train, y_train)
    x_test_player_champion_winRate = generate_player_champion_winRate(x_test, y_test)

    dt_train_player_coop_winRate = generate_player_coop_df(x_train, y_train)
    dt_test_player_coop_winRate = generate_player_coop_df(x_test, y_test)

    dt_train_player_vs = generate_player_vs_df(x_train, y_train)
    dt_test_player_vs = generate_player_vs_df(x_test, y_test)

    dt_train_playerChampion_vs = generate_player_with_champion_wr_df(x_train, y_train)
    dt_test_playerChampion_vs = generate_player_with_champion_wr_df(x_test, y_test)

    dt_train_champion_vs = generate_champion_vs_df(x_train, y_train)
    dt_test_champion_vs = generate_champion_vs_df(x_test, y_test)

    df_train_blue_team_wr, df_train_red_team_wr = generate_team_wr(x_train, y_train)
    df_test_blue_team_wr, df_test_red_team_wr = generate_team_wr(x_test, y_test)

    if 'rResult' in x_train.columns:
        x_train.drop(columns='rResult', inplace=True)

    if 'rResult' in x_test.columns:        
        x_test.drop(columns='rResult', inplace=True)
    
    x_train = process_feature1(x_train, x_train_player_data)
    x_test = process_feature1(x_test, x_test_player_data)

    x_train = process_feature2(x_train, x_train_player_champion_winRate)
    x_test = process_feature2(x_test, x_test_player_champion_winRate)

    x_train = process_feature3(x_train, dt_train_player_coop_winRate)
    x_test = process_feature3(x_test, dt_test_player_coop_winRate)

    x_train = process_feature4(x_train, dt_train_player_vs)
    x_test = process_feature4(x_test, dt_test_player_vs)

    # x_train['btPlayerChampion'] = process_feature2_test(x_train, x_train_player_champion_winRate, 'blueTop', 'blueTopChamp')
    
    x_train = process_feature6(x_train, dt_train_playerChampion_vs)
    x_test = process_feature6(x_test, dt_test_playerChampion_vs)

    x_train = process_feature7(x_train, dt_train_champion_vs)
    x_test = process_feature7(x_test, dt_test_champion_vs)

    x_train = process_feature8(x_train, df_train_blue_team_wr, df_train_red_team_wr)
    x_test = process_feature8(x_test, df_test_blue_team_wr, df_test_red_team_wr)

    x_train.drop(columns=DROPCOLS, inplace=True)
    x_test.drop(columns=DROPCOLS, inplace=True)

    x_combined = pd.concat([x_train, x_test])
    
    return x_train, x_test, y_train, y_test, x_combined, y_data_full_df

def generate_playerData_df(df_split_dataset):
    role_columns = []
    for team in TEAM_COLOR:
        for role in TEAM_ROLE:
            role_columns.append(f"{team.lower()}{role}")

    players_df = df_split_dataset.melt(value_vars=role_columns, value_name='Player', var_name='Role') 
    player_counts_vectorized = players_df['Player'].value_counts().reset_index()
    player_counts_vectorized.columns = ['Player', 'Plays']

    #Total wins
    blue_wins = df_split_dataset[df_split_dataset['rResult'] == 0][role_columns[:5]].melt(value_name='Player').value_counts().reset_index(name='Win_Count').fillna(0)
    red_wins = df_split_dataset[df_split_dataset['rResult'] == 1][role_columns[5:]].melt(value_name='Player').value_counts().reset_index(name='Win_Count').fillna(0)
    win_counts_vectorized = pd.concat([blue_wins, red_wins]).groupby('Player').sum().reset_index()
    win_counts_vectorized.drop(columns='variable', inplace=True)
    player_counts_vectorized["Win_Count"] = win_counts_vectorized["Win_Count"]
    player_counts_vectorized["Win_Count"] = player_counts_vectorized["Win_Count"].fillna(0)

    #Create player_data dataframe.
    for role in TEAM_ROLE:
        num_plays = df_split_dataset[[f"blue{role}", f"red{role}"]].melt(value_name='Player').value_counts().reset_index(name=f"{role}_Plays")
        num_plays = num_plays.groupby('Player').sum().reset_index()
        num_plays.drop(columns='variable', inplace=True)
        
        blue_wins = df_split_dataset[df_split_dataset['rResult'] == 0][[f"blue{role}"]].melt(value_name='Player').value_counts().reset_index(name=f"{role}_Win_Count").fillna(0)
        red_wins = df_split_dataset[df_split_dataset['rResult'] == 1][[f"red{role}"]].melt(value_name='Player').value_counts().reset_index(name=f"{role}_Win_Count").fillna(0)
        
        win_count_vectorized = pd.concat([blue_wins, red_wins]).groupby('Player').sum().reset_index()
        win_count_vectorized.drop(columns='variable', inplace=True)
        
        merge_df = num_plays.merge(win_count_vectorized, on='Player', how='left').fillna(0)
        merge_df[f"{role}_Win_Ratio"] = ((merge_df[f"{role}_Win_Count"]) / merge_df[f"{role}_Plays"].astype(float)).fillna(0.5)

        player_counts_vectorized = player_counts_vectorized.merge(merge_df, on='Player', how='left')
        player_counts_vectorized[f"{role}_Win_Count"] = player_counts_vectorized[f"{role}_Win_Count"].fillna(0)
        player_counts_vectorized[f"{role}_Plays"] = player_counts_vectorized[f"{role}_Plays"].fillna(0)
        player_counts_vectorized[f"{role}_Win_Ratio"] = player_counts_vectorized[f"{role}_Win_Ratio"].fillna(0.5)

    return player_counts_vectorized

def generate_player_champion_winRate(df_split_dataset, df_training_labels):
    """
        Generates a support dataframe that contains a player's win rate for specific
        champions. This is in support of implementing feature 2 from the research paper.
    """
    modified_match_data = df_split_dataset.copy()
    modified_match_data['match_id'] = range(1, len(modified_match_data) + 1)
    modified_match_data['bResult'] = df_training_labels
    
    blue_team = modified_match_data.melt(
        id_vars=['match_id', 'bResult'],
        value_vars=['blueTop', 'blueJungle', 'blueMiddle', 'blueADC', 'blueSupport', ],
        var_name='position',
        value_name='player'
    )

    blue_team['champion'] = modified_match_data.melt(
        id_vars=['match_id'],
        value_vars=['blueTopChamp', 'blueJungleChamp', 'blueMiddleChamp', 'blueADCChamp', 'blueSupportChamp'],
        value_name='champion'
    )['champion']

    blue_team['team'] = 'blue'

    red_team = modified_match_data.melt(
        id_vars=['match_id', 'bResult'],
        value_vars=['redTop', 'redJungle', 'redMiddle', 'redADC', 'redSupport'],
        var_name='position',
        value_name='player'
    )
    red_team['champion'] = modified_match_data.melt(
        id_vars=['match_id'],
        value_vars=['redTopChamp', 'redJungleChamp', 'redMiddleChamp', 'redADCChamp', 'redSupportChamp'],
        value_name='champion'
    )['champion']

    red_team['team'] = 'red'

    # Combine blue and red team data
    combined_df = pd.concat([blue_team, red_team])

    # Calculate number of times a player has used a certain champion
    usage_count = combined_df.groupby(['player', 'champion']).size().reset_index(name='usage_count')

    # Calculate number of times a player has won with a certain champion
    combined_df['win'] = (combined_df['team'] == 'blue') & (combined_df['bResult'] == 1) | \
                        (combined_df['team'] == 'red') & (combined_df['bResult'] == 0)

    win_count = combined_df[combined_df['win']].groupby(['player', 'champion']).size().reset_index(name='win_count')

    # Merge usage and win counts
    result = pd.merge(usage_count, win_count, how='left', on=['player', 'champion'])
    result['win_count'] = result['win_count'].fillna(0)
    result['win_ratio'] = result['win_count'] / result['usage_count']

    return result 

def generate_player_coop_df(df_split_dataset, df_training_labels):
    modified_match_data = df_split_dataset.copy()
    modified_match_data['match_id'] = range(1, len(modified_match_data) + 1)
    modified_match_data['bResult'] = df_training_labels

    # Dictionary to track total pair counts and win counts
    pair_counts = Counter()
    pair_wins = Counter()

    # Iterate through each row in the DataFrame
    for _, row in modified_match_data.iterrows():
        # Blue team players and result
        blue_team = [row['blueTop'], row['blueMiddle'], row['blueJungle'], 
                     row['blueMiddle'] , row['blueADC'], row['blueSupport']]
        blue_win = row['bResult'] == 1
        
        # Red team players
        red_team = [row['redTop'], row['redMiddle'], row['redJungle'], 
                    row['redMiddle'], row['redADC'], row['redSupport']]
        
        # Count blue team pairs
        for player1, player2 in combinations(sorted(set(blue_team)), 2):
            pair = (player1, player2)
            pair_counts[pair] += 1
            if blue_win:
                pair_wins[pair] += 1
        
        # Count red team pairs
        for player1, player2 in combinations(sorted(set(red_team)), 2):
            pair = (player1, player2)
            pair_counts[pair] += 1
            if not blue_win:
                pair_wins[pair] += 1

    # Create a DataFrame from the pair counts and win counts
    pair_df = pd.DataFrame(
        {
            'pair': list(pair_counts.keys()),
            'games_played': list(pair_counts.values()),
            'num_wins': [pair_wins[pair] for pair in pair_counts.keys()]
        }
    )

    pair_df['win_ratio'] = pair_df['num_wins'] / pair_df['games_played']

    #Vectorize the dataframe for fast lookup
    pair_dt = pair_df.set_index('pair')['win_ratio'].to_dict()

    return pair_dt

def generate_player_vs_df(df_split_dataset, df_training_labels):
    modified_match_data = df_split_dataset.copy()
    modified_match_data['match_id'] = range(1, len(modified_match_data) + 1)
    modified_match_data['bResult'] = df_training_labels

    # Extract blue and red players into separate DataFrames
    blue_players_df = modified_match_data[['blueTop', 'blueMiddle', 'blueJungle', 'blueADC', 'blueSupport']]
    red_players_df = modified_match_data[['redTop', 'redMiddle', 'redJungle', 'redADC', 'redSupport']]

    # Flatten the blue and red players DataFrames into lists of tuples containing player pairs
    blue_players = blue_players_df.values.tolist()
    red_players = red_players_df.values.tolist()

    # Create a list of all matchups by pairing blue and red players
    matchups = []
    num_wins = []

    for blue, red, blue_win in zip(blue_players, red_players, modified_match_data['bResult']):
        for blue_player, red_player in product(blue, red):
            matchups.append((blue_player, red_player))

            # If blue team won, increment win for blue player, otherwise for red player
            if blue_win == 1:
                num_wins.append((blue_player, red_player, 1, 0))
            else:
                num_wins.append((blue_player, red_player, 0, 1))

    # Create a DataFrame from the matchups and count occurrences
    matchups_df = pd.DataFrame(matchups, columns=['player', 'opponent'])
    matchups_count_df = matchups_df.groupby(['player', 'opponent']).size().reset_index(name='numPlayed')

    # Create a DataFrame from the num_wins and calculate the number of wins
    num_wins_df = pd.DataFrame(num_wins, columns=['player', 'opponent', 'playerWins', 'opponentWins'])
    num_wins_summary = num_wins_df.groupby(['player', 'opponent']).sum().reset_index()
    num_wins_summary['numWins'] = num_wins_summary['playerWins']

    # Merge the matchup counts and win counts DataFrames
    player_vs_df = pd.merge(matchups_count_df, num_wins_summary[['player', 'opponent', 'numWins']], 
                        on=['player', 'opponent'], how='left')

    # Fill NaN values in numWins with 0
    player_vs_df['numWins'] = player_vs_df['numWins'].fillna(0).astype(int)
    player_vs_df['win_ratio'] = player_vs_df['numWins'] / player_vs_df['numPlayed']

    dt_player_vs = player_vs_df.set_index(['player', 'opponent'])['win_ratio'].to_dict()

    return dt_player_vs

def generate_player_with_champion_wr_df(df_split_dataset, df_training_labels):
    #Implementation of feature 6: coopChampion-blue/red
    modified_match_data = df_split_dataset.copy()
    modified_match_data['match_id'] = range(1, len(modified_match_data) + 1)
    modified_match_data['bResult'] = df_training_labels

    blue_team = pd.DataFrame()
    red_team = pd.DataFrame()

    blue_team = modified_match_data.melt(
        id_vars=['match_id', 'bResult'],
        value_vars=['blueTop', 'blueJungle', 'blueMiddle', 'blueADC', 'blueSupport', ],
        var_name='position',
        value_name='player'
    )

    blue_team['champion'] = modified_match_data.melt(
        id_vars=['match_id'],
        value_vars=['blueTopChamp', 'blueJungleChamp', 'blueMiddleChamp', 'blueADCChamp', 'blueSupportChamp'],
        value_name='champion'
    )['champion']

    blue_team['team'] = 'blue'

    red_team = modified_match_data.melt(
        id_vars=['match_id', 'bResult'],
        value_vars=['redTop', 'redJungle', 'redMiddle', 'redADC', 'redSupport'],
        var_name='position',
        value_name='player'
    )
    red_team['champion'] = modified_match_data.melt(
        id_vars=['match_id'],
        value_vars=['redTopChamp', 'redJungleChamp', 'redMiddleChamp', 'redADCChamp', 'redSupportChamp'],
        value_name='champion'
    )['champion']

    red_team['team'] = 'red'

    # Combine blue and red team data
    combined_df = pd.concat([blue_team, red_team])

    # Calculate number of times a player has won with a certain champion
    combined_df['win'] = (combined_df['team'] == 'blue') & (combined_df['bResult'] == 1) | \
                        (combined_df['team'] == 'red') & (combined_df['bResult'] == 0)

    # Calculate number of times a player has used a certain champion
    combined_df['position'] = combined_df['position'].str.replace('red|blue', '', regex=True)

    win_count = combined_df[combined_df['win']].groupby(['champion', 'position']).size().reset_index(name='win_count')
    usage_count = combined_df.groupby(['champion', 'position']).size().reset_index(name='usage_count')

    # Merge usage and win counts
    df_champion_wr = pd.merge(usage_count, win_count, how='left', on=['champion', 'position'])
    df_champion_wr['win_count'] = df_champion_wr['win_count'].fillna(0)
    df_champion_wr['win_ratio'] = df_champion_wr['win_count'] / df_champion_wr['usage_count']
         
    return df_champion_wr

def generate_champion_vs_df(df_split_dataset, df_training_labels):
    modified_match_data = df_split_dataset.copy()
    modified_match_data['match_id'] = range(1, len(modified_match_data) + 1)
    modified_match_data['bResult'] = df_training_labels

    blue_champs_df = modified_match_data[['blueTopChamp', 'blueMiddleChamp', 'blueJungleChamp', 'blueADCChamp', 'blueSupportChamp']]
    red_champs_df =  modified_match_data[['redTopChamp', 'redMiddleChamp', 'redJungleChamp', 'redADCChamp', 'redSupportChamp']]

    blue_champs = blue_champs_df.values.tolist()
    red_champs = red_champs_df.values.tolist()

    matchups = []
    num_wins = []

    for blue, red, blue_win in zip(blue_champs, red_champs, modified_match_data['bResult']):
        for blue_player, red_player in product(blue, red):
            matchups.append((blue_player, red_player))

            # If blue team won, increment win for blue player, otherwise for red player
            if blue_win == 1:
                num_wins.append((blue_player, red_player, 1, 0))
            else:
                num_wins.append((blue_player, red_player, 0, 1))

    # Create a DataFrame from the matchups and count occurrences
    matchups_df = pd.DataFrame(matchups, columns=['champion', 'opponent'])
    matchups_count_df = matchups_df.groupby(['champion', 'opponent']).size().reset_index(name='numPlayed')

    # Create a DataFrame from the num_wins and calculate the number of wins
    num_wins_df = pd.DataFrame(num_wins, columns=['champion', 'opponent', 'championWins', 'opponentWins'])
    num_wins_summary = num_wins_df.groupby(['champion', 'opponent']).sum().reset_index()
    num_wins_summary['numWins'] = num_wins_summary['championWins']

    # Merge the matchup counts and win counts DataFrames
    champion_vs_df = pd.merge(matchups_count_df, num_wins_summary[['champion', 'opponent', 'numWins']], 
                        on=['champion', 'opponent'], how='left')

    # Fill NaN values in numWins with 0
    champion_vs_df['numWins'] = champion_vs_df['numWins'].fillna(0).astype(int)
    champion_vs_df['win_ratio'] = champion_vs_df['numWins'] / champion_vs_df['numPlayed']

    dt_champion_vs = champion_vs_df.set_index(['champion', 'opponent'])['win_ratio'].to_dict()

    return dt_champion_vs

def generate_team_wr(x_dataset, df_training_labels):
    modified_match_data = x_dataset.copy()
    modified_match_data['bResult'] = df_training_labels

    modified_match_data['bNumWins'] = modified_match_data['bResult'].apply(lambda x: 1 if x == 1 else 0)
    modified_match_data['rNumWins'] = modified_match_data['bResult'].apply(lambda x: 1 if x == 0 else 0)

    blue_team_df = modified_match_data.groupby('blueTeamTag').agg({
        'bResult': 'size',           # Total count of occurrences for each blue team.
        'bNumWins': 'sum'           # Sum of 'numWins' to count where 'bResult' equals 1
    }).reset_index()

    red_team_df = modified_match_data.groupby('redTeamTag').agg({
        'bResult': 'size',           # Total count of occurrences for each blue team.
        'rNumWins': 'sum'           # Sum of 'numWins' to count where 'bResult' equals 1
    }).reset_index()

    blue_team_df.columns = ['blueTeamTag', 'matchesPlayed', 'numWins']
    red_team_df.columns = ['redTeamTag', 'matchesPlayed', 'numWins']

    blue_team_df['win_ratio'] = blue_team_df['numWins'] / blue_team_df['matchesPlayed']
    red_team_df['win_ratio'] = red_team_df['numWins'] / red_team_df['matchesPlayed']

    blue_team_dt = blue_team_df.set_index(['blueTeamTag'])['win_ratio'].to_dict()
    red_team_dt = red_team_df.set_index(['redTeamTag'])['win_ratio'].to_dict()

    return blue_team_dt, red_team_dt

def process_feature1(x_dataset, df_player_data):
    for teamColor in TEAM_COLOR:
        for role in TEAM_ROLE:            
            player_data_map = df_player_data.set_index('Player')[f"{role}_Win_Ratio"].to_dict()
            
            labelName = WIN_RATE_PLAYER_ROLE_REPLACE_COLS[f"{teamColor.lower()}{role}"]
            x_dataset[labelName] = x_dataset[f"{teamColor.lower()}{role}"].map(player_data_map)

    return x_dataset

def process_feature2(x_dataset, df_player_champion_data):
    merged_df = x_dataset.copy()

    for teamColor in TEAM_COLOR:
        for role in TEAM_ROLE:
            combinedLabel = f"{teamColor.lower()}{role}"
            
            merged_df = merged_df.merge(df_player_champion_data, how='left', 
                                        left_on=[combinedLabel, f"{combinedLabel}Champ"], 
                                        right_on=['player', 'champion'])

            labelName = WIN_RATE_CHAMPION_REPLACE_COLS[f"{teamColor.lower()}{role}Champ"]            
            merged_df.drop(columns=['player', 'champion', 'usage_count', 'win_count'], inplace=True)
            merged_df.rename(columns={'win_ratio' : labelName}, inplace=True)

    return merged_df

def process_feature3(x_dataset, dt_pair_data):
    modified_match_data = x_dataset.copy()
    modified_match_data['match_id'] = range(1, len(modified_match_data) + 1)

    # Apply the function to compute the sum of win ratios for blue and red teams
    modified_match_data['bCoopPlayer'] = modified_match_data.apply(
        lambda row: compute_sum_pair_win_ratios([row['blueTop'], row['blueJungle'], row['blueMiddle'] , 
                                                 row['blueADC'], row['blueSupport']], 
                                                 dt_pair_data), 
                                                 axis=1)
    modified_match_data['rCoopPlayer'] = modified_match_data.apply(
        lambda row: compute_sum_pair_win_ratios([row['redTop'], row['redJungle'], row['redMiddle'],
                                                 row['redADC'], row['redSupport']], 
                                                 dt_pair_data), 
                                                 axis=1)

    return modified_match_data

def process_feature4(x_dataset, dt_player_vs):
    modified_match_data = x_dataset.copy()
    win_ratios_df = modified_match_data.apply(compute_sum_vs_redTeam, axis=1, win_dict=dt_player_vs)
    modified_match_data['vsPlayer'] = win_ratios_df

    return modified_match_data

def process_feature5(x_dataset):
    pass


def process_feature6(x_dataset, df_champion_wr):
     #Convert the matches DataFrame to a long format
    df_long = pd.melt(x_dataset.reset_index(), id_vars=['index'], 
                    value_vars=['blueTopChamp', 'blueJungleChamp', 'blueMiddleChamp', 'blueADCChamp', 'blueSupportChamp',
                                'redTopChamp', 'redJungleChamp', 'redMiddleChamp', 'redADCChamp', 'redSupportChamp'],
                    var_name='team_role', value_name='champion')

    df_long['position'] = df_long['team_role'].str.replace('red|blue', '', regex=True).replace('Champ', '', regex=True)
    df_merged = df_long.merge(df_champion_wr, how='left', on=['champion', 'position'])
    df_result = df_merged.pivot(index='index', columns='team_role', values='win_ratio')

    player_combos = list(combinations(TEAM_ROLE, 2))
    output_df = x_dataset.copy()

    for teamColor in ['red', 'blue']:
        team_initial = 'b' if teamColor == 'blue' else 'r'
        outputLabel = f"{team_initial}CoopChampion"
        output_df[outputLabel] = 0.0

        for pair in player_combos:
            combinedLabel_1 = f"{teamColor}{pair[0]}Champ"
            combinedLabel_2 = f"{teamColor}{pair[1]}Champ"
            
            output_df[outputLabel] += df_result[combinedLabel_1] + df_result[combinedLabel_2]

    return output_df

def process_feature7(x_dataset, dt_champion_pair_data):
    modified_match_data = x_dataset.copy()
    modified_match_data['match_id'] = range(1, len(modified_match_data) + 1)

    # Apply the function to compute the sum of win ratios for blue and red teams
    modified_match_data['vsChampion'] = modified_match_data.apply(
        lambda row: compute_sum_pair_win_ratios([row['blueTopChamp'], row['blueJungleChamp'], row['blueMiddleChamp'] , 
                                                 row['blueADCChamp'], row['blueSupportChamp']], 
                                                 dt_champion_pair_data), 
                                                 axis=1)

    return modified_match_data

def process_feature8(x_dataset, dt_blue_team_wr, dt_red_team_wr):
    modified_match_data = x_dataset.copy()

    modified_match_data['bTeamColor'] = modified_match_data['blueTeamTag'].map(dt_blue_team_wr)
    modified_match_data['rTeamColor'] = modified_match_data['redTeamTag'].map(dt_red_team_wr)

    return modified_match_data

# Function to compute the sum of pair win ratios for a given list of players
def compute_sum_pair_win_ratios(players, pair_dt):
    pairs = list(combinations(sorted(set(players)), 2))
    return sum(pair_dt.get(tuple(sorted(set(pair))), 0) for pair in pairs)

def compute_sum_vs_redTeam(row, win_dict):
    """
        Returns the total value of all the blue players' win ratios against
        all members of the red team. Used in support of implementing feature 4.        
    """
    blue_roles = ['blueTop', 'blueJungle', 'blueMiddle', 'blueADC', 'blueSupport']
    red_roles = ['redTop', 'redJungle', 'redMiddle', 'redADC', 'redSupport']    
    total = 0.0

    for blue_role in blue_roles:
        blue_player = row[blue_role]

        for red_role in red_roles:
            red_player = row[red_role]

            # Retrieve the win_ratio from win_dict, handle missing data with None or a default value
            win_ratio = win_dict.get((blue_player, red_player))
            
            total += win_ratio
                    
    return total

def generate_temp_csv_data():
    tempTrain_dirName = "temp_train"
    tempTest_dirName = "temp_test"

    if os.path.exists(f"feature_data\\{tempTrain_dirName}"):
        shutil.rmtree(f"feature_data\\{tempTrain_dirName}")
    
    os.mkdir(f"feature_data\\{tempTrain_dirName}")

    if os.path.exists(f"feature_data\\{tempTest_dirName}"):
        shutil.rmtree(f"feature_data\\{tempTest_dirName}")
    
    os.mkdir(f"feature_data\\{tempTest_dirName}")

    makedata.make_fixed_split_match_data(feature_folder_train=tempTrain_dirName, feature_folder_test=tempTest_dirName, percent_for_training=0.90)

    for folder in [tempTrain_dirName, tempTest_dirName]:
        makedata.parse_match_info(folder, feature_folder_original_data="original")
        makedata.WritePlayerData(folder)
        makedata.WriteChampionData(folder)
        makedata.WritePlayerWinsWithEachChampionData(folder)
        makedata.WriteTeamData(folder)
        makedata.WritePlayerVsData(folder)
        makedata.WriteChampionVsData(folder)
        makedata.WriteMatchVSAndCoopData(folder)

    pass

#Define main function to enable running file independently from other components.
if __name__ == "__main__":

    x_train, x_test, y_train, y_test, x_combined, y_data_full_df = load_matchdata_into_df("original")

    # generate_temp_csv_data()

    # makedata.make_fixed_split_match_data(feature_folder_train="temp_train", feature_folder_test="temp_test")
    pass

    # bArgIsTrainingSet = False
    # bArgGenerateOutputFile = False
    # bArgIncludeChampionRole_Feature = False

    # import sys
    # args = sys.argv[1:]
    # for i in range(len(args)):
    #     if (args[i] == "-train"):
    #         bArgIsTrainingSet = True
    #     elif (args[i] == "-test"):
    #         bArgIsTrainingSet = False
    #     elif (args[i] == "-o" or args[i] == "-O"):
    #         bArgGenerateOutputFile = True
    #     elif (args[i] == "-c" or args[i] == "-C"):
    #         bArgIncludeChampionRole_Feature = True
    
    # load_data_from_csv( \
    #     bIsTrainingSet=bArgIsTrainingSet, \
    #     bGenerateOutputFile=bArgGenerateOutputFile, \
    #     bIncludeChampionRole_Feature=bArgIncludeChampionRole_Feature)
