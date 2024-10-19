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
    "btPlayerChampion": "blueTopChamp",
    "bjPlayerChampion": "blueJungleChamp",
    "bmPlayerChampion": "blueMiddleChamp",
    "baPlayerChampion": "blueADCChamp",    
    "bsPlayerChampion": "blueSupportChamp",    
    "rtPlayerChampion": "redTopChamp",    
    "rjPlayerChampion": "redJungleChamp",    
    "rmPlayerChampion": "redMiddleChamp",
    "raPlayerChampion": "redADCChamp",    
    "rsPlayerChampion": "redSupportChamp"
}

#Map the 'Name' and 'Win Rate' columns from player_data.csv to the corresponding fields in the matchinfo.csv columns
#as noted in the values fields below.
WIN_RATE_PLAYER_ROLE_REPLACE_COLS = {
    "btPlayerRole": "blueTop",
    "bjPlayerRole": "blueJungle",    
    "bmPlayerRole": "blueMiddle",    
    "baPlayerRole": "blueADC",
    "bsPlayerRole": "blueSupport",    
    "rtPlayerRole": "redTop",    
    "rjPlayerRole": "redJungle",    
    "rnPlayerRole": "redMiddle",    
    "raPlayerRole": "redADC",    
    "rsPlayerRole": "redSupport",    
}

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
    
    #Load the initial data, then start replacing the fields with the engineered data.
    match_data = pd.read_csv(match_data_path, usecols=INPUT_COLS_MATCHDATA)
    champion_data = pd.read_csv(dir_data_champion)
    player_data = pd.read_csv(dir_player_data)
    

    #TODO: Replace fields in match_data with engineered fields, using the names used in the research paper.
    modified_match_df = match_data

    for key, value in WIN_RATE_CHAMPION_REPLACE_COLS.items():
        modified_match_df = modified_match_df.merge(champion_data[['Name', 'Win Ratio']], left_on=value, right_on='Name')
        modified_match_df.rename(columns={'Win Ratio': key}, inplace=True)

        #The Name field is automatically merged into the original dataframe, so we need to drop it, along with the old column mapping
        #since we don't need that either.
        modified_match_df.drop(['Name', value], axis=1, inplace=True) 
    
    for key, value in WIN_RATE_PLAYER_ROLE_REPLACE_COLS.items():
        pass


#Define main function to enable running file independently from other components.
if __name__ == "__main__":
    load_data()