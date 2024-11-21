import pandas as pd
import shutil
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import makedata

from sklearn.model_selection import train_test_split

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

    x_train_player_vs_data = generate_playerData_df(x_train)

    if 'rResult' in x_train.columns:
        x_train.drop(columns='rResult', inplace=True)

    if 'rResult' in x_test.columns:        
        x_test.drop(columns='rResult', inplace=True)
    
    x_train = process_feature1(x_train, x_train_player_data)
    x_test = process_feature1(x_test, x_test_player_data)
    
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

def generate_player_vs_df(df_split_dataset):
    df_team_1 = df_split_dataset
    pass

def process_feature1(x_dataset, df_player_data):
    for teamColor in TEAM_COLOR:
        for role in TEAM_ROLE:            
            player_data_map = df_player_data.set_index('Player')[f"{role}_Win_Ratio"].to_dict()
            
            labelName = WIN_RATE_PLAYER_ROLE_REPLACE_COLS[f"{teamColor.lower()}{role}"]
            x_dataset[labelName] = x_dataset[f"{teamColor.lower()}{role}"].map(player_data_map)

    return x_dataset

def generate_temp_csv_data(train_test_split=0.9):
    tempTrain_dirName = "temp_train"
    tempTest_dirName = "temp_test"

    if os.path.exists(f"feature_data\\{tempTrain_dirName}"):
        shutil.rmtree(f"feature_data\\{tempTrain_dirName}")
    
    os.mkdir(f"feature_data\\{tempTrain_dirName}")

    if os.path.exists(f"feature_data\\{tempTest_dirName}"):
        shutil.rmtree(f"feature_data\\{tempTest_dirName}")
    
    os.mkdir(f"feature_data\\{tempTest_dirName}")

    makedata.make_fixed_split_match_data(feature_folder_train=tempTrain_dirName, feature_folder_test=tempTest_dirName, train_test_split=0.9)

    for folder in [tempTrain_dirName, tempTest_dirName]:
        makedata.parse_match_info(folder, feature_folder_original_data="original")
        makedata.WritePlayerData(folder)
        makedata.WriteChampionData(folder)
        makedata.WritePlayerWinsWithEachChampionData(folder)
        makedata.WriteTeamData(folder)
        makedata.WritePlayerVsData(folder)
        makedata.WriteChampionVsData(folder)
        makedata.WriteMatchVSAndCoopData(folder)

#Define main function to enable running file independently from other components.
if __name__ == "__main__":
    args = sys.argv[1:]
    print(len(args))

    #Run the commands below if script is run via VSCode and not cmdline.
    if len(args) == 0:
        generate_temp_csv_data()

    else:
        bArgIsTrainingSet = False
        bArgGenerateOutputFile = False
        bArgIncludeChampionRole_Feature = False
        
        for i in range(len(args)):
            if (args[i] == "-train"):
                bArgIsTrainingSet = True
            elif (args[i] == "-test"):
                bArgIsTrainingSet = False
            elif (args[i] == "-o" or args[i] == "-O"):
                bArgGenerateOutputFile = True
            elif (args[i] == "-c" or args[i] == "-C"):
                bArgIncludeChampionRole_Feature = True
        
        load_data_from_csv( \
            bIsTrainingSet=bArgIsTrainingSet, \
            bGenerateOutputFile=bArgGenerateOutputFile, \
            bIncludeChampionRole_Feature=bArgIncludeChampionRole_Feature)
