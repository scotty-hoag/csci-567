import numpy as np
import pandas as pd
import os

INPUT_COLS_MATCHDATA = [
    "blueTeamTag",
    "redTeamTag",
    "bResult",
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
    "redMiddle": "rnPlayerRole",
    "redADC": "raPlayerRole",
    "redSupport": "rsPlayerRole"
}

TEAM_COLOR = ["Blue", "Red"]
TEAM_ROLE = ["Top", "Jungle", "Middle", "ADC", "Support"]

def load_data():
    """
        Return data:

    """
    #Note: We assume that the folder structure will be consistent throughout the development process.
    dir_feature_data = "feature_data"

    file_match_data = "matchinfo.csv"
    file_data_champion = "champion_data.csv"
    file_champion_vs_data = "champion_vs_data.csv"
    file_player_data = "player_data.csv"
    file_player_vs_data = "player_vs_data.csv"
    file_player_wins_with_each_champion_data = "player_wins_with_each_champion_data.csv"
    file_team_data = "team_data.csv"

    match_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', f"lol_data\{file_match_data}")
    dir_data_champion = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', f"feature_data\{file_data_champion}")
    dir_player_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', f"feature_data\{file_player_data}")
    dir_player_wins_with_each_champion_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 
                                                           f"feature_data\{file_player_wins_with_each_champion_data}")
    dir_player_vs_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', f"feature_data\{file_player_vs_data}")
    
    #Load the initial data, then start replacing the fields with the engineered data.
    match_data = pd.read_csv(match_data_path, usecols=INPUT_COLS_MATCHDATA)
    champion_data = pd.read_csv(dir_data_champion)
    player_data = pd.read_csv(dir_player_data)
    player_wr_champion_data = pd.read_csv(dir_player_wins_with_each_champion_data)
    player_vs_data = pd.read_csv(dir_player_vs_data)

    #Remove extra white space before processing
    player_data.columns = player_data.columns.str.strip()
        
    modified_match_df = match_data

    #Implementation of Feature 1: playerRole

    #Compute the win ratios for each role and add the field to the dataframe.
    for teamColor in TEAM_COLOR:
        for role in TEAM_ROLE:
            player_data[f"wr{teamColor}{role}"] = (player_data[f"{teamColor} {role} Wins"] / player_data[f"{teamColor} {role}"]).fillna(0)
            player_data_map = player_data.set_index('Name')[f"wr{teamColor}{role}"].to_dict()
            modified_match_df[f"wr{teamColor}{role}"] = modified_match_df[f"{teamColor.lower()}{role}"].map(player_data_map).fillna(0)
    
    #Implementation of Feature 2: playerChampion
    player_wins_df = player_wr_champion_data.melt(
        id_vars=['Player Name'],
        var_name='Champion',
        value_name='WinRate'
    )

    for field, newField in WIN_RATE_PLAYER_ROLE_REPLACE_COLS.items():
        role_merge = modified_match_df.merge(
            player_wins_df,
            left_on=[field, f"{field}Champ"],
            right_on=['Player Name', 'Champion'],
            how='left'            
        )

        modified_match_df[newField] = role_merge['WinRate']

    #Drop columns at end of processing.
    # modified_match_df.drop(WIN_RATE_PLAYER_ROLE_REPLACE_COLS.keys(), axis=1, inplace=True)    

    #Implementation of Feature 3
    #TODO: Implement after csv file has been generated.
        
    #Implementation of Feature 4: vsPlayer
    player_winRatio_vs_df = player_vs_data.melt(
        id_vars=['Name'],
        var_name='Opponent',
        value_name='WinRate'
    )

    #Example implementation    
    match_long_df = match_data.melt(
        id_vars=['blueTop'], 
        value_vars=['redTop','redJungle','redMiddle'],                                     
        var_name='role', 
        value_name='opponent'
    )
    
    # for role in WIN_RATE_PLAYER_ROLE_REPLACE_COLS:
    #     blue_player = match_data[role]
        
    #     opponent_merge = modified_match_df.merge(
    #         player_vs_data,
    #         left_on=[role],
    #         right_on=['Name']
    #     )

    merged_df = match_long_df.merge(
        player_winRatio_vs_df,
        left_on=['blueTop', 'opponent'],
        right_on=['Name', 'Opponent'],
        how='left'
    )

    strTest = "test"    


    #Implementation of Feature 5
        
    #Implementation of Feature 6
        
    #Implementation of Feature 7
        
    #Implementation of Feature 8
                
#Define main function to enable running file independently from other components.
if __name__ == "__main__":
    load_data()