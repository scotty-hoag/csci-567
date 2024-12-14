import numpy as np
import pandas as pd
import os

import time

IMPORT_COLS = [
    'gameid',
    'league',
    'participantid',    
    'side',
    'position',
    'playername',
    'teamname',
    'champion',
    'result'
]

def import_dataset_csv(filePath):
    df_import = pd.read_csv(filePath, usecols=IMPORT_COLS)
    rows = []

    for gameid, group in df_import.groupby('gameid'):
        row = {
            "id": gameid,
            "blueTeamTag": group.iloc[10]['teamname'],
            "redTeamTag": group.iloc[11]['teamname'],
            "blueTop": group.iloc[0]['playername'],
            "blueTopChamp": group.iloc[0]['champion'],
            "blueJungle": group.iloc[1]['playername'],
            "blueJungleChamp": group.iloc[1]['champion'],
            "blueMiddle": group.iloc[2]['playername'],
            "blueMiddleChamp": group.iloc[2]['champion'],
            "blueADC": group.iloc[3]['playername'],
            "blueADCChamp": group.iloc[3]['champion'],
            "blueSupport": group.iloc[4]['playername'],
            "blueSupportChamp": group.iloc[4]['champion'],
            "redTop": group.iloc[5]['playername'],
            "redTopChamp": group.iloc[5]['champion'],
            "redJungle": group.iloc[6]['playername'],
            "redJungleChamp": group.iloc[6]['champion'],
            "redMiddle": group.iloc[7]['playername'],
            "redMiddleChamp": group.iloc[7]['champion'],
            "redADC": group.iloc[8]['playername'],
            "redADCChamp": group.iloc[8]['champion'],
            "redSupport": group.iloc[9]['playername'],
            "redSupportChamp": group.iloc[9]['champion'],
            'rResult': group.iloc[11]['result'],
            'bResult': group.iloc[10]['result']
        }

        rows.append(row)

    new_df = pd.DataFrame(rows)

    fileDir, file_name = os.path.split(filePath)
    converted_fullPath = f"{fileDir}\\{file_name.split('.')[0]}_converted.csv"

    if not os.path.isfile(converted_fullPath):        
        new_df.to_csv(converted_fullPath)

    
def combine_datasets(directory):
    """

    """
    ltEntries = [
        "match_info_2019.csv",
        "match_info_2020.csv",
        "match_info_2021.csv",
        "match_info_2022.csv",
        "match_info_2023.csv",
        "match_info_2024.csv"
    ]

    base_df = pd.DataFrame()
    dataFrames = []

    for item in ltEntries:
        currentDf = pd.read_csv(f"{directory}\\{item}")        
        dataFrames.append(currentDf)

    base_df = pd.concat(dataFrames, ignore_index=True)
    base_df.drop(columns='Unnamed: 0', inplace=True)

    if not os.path.isfile(f"{directory}\\match_info_combined.csv"):        
        base_df.to_csv(f"{directory}\\match_info_combined.csv")


if __name__ == "__main__":
    lt_files = [
        "D:\\Github Repos\\csci-567\\feature_data\\new\\2019_LoL_esports_match_data_from_OraclesElixir.csv",
        "D:\\Github Repos\\csci-567\\feature_data\\new\\2020_LoL_esports_match_data_from_OraclesElixir.csv",
        "D:\\Github Repos\\csci-567\\feature_data\\new\\2021_LoL_esports_match_data_from_OraclesElixir.csv",
        "D:\\Github Repos\\csci-567\\feature_data\\new\\2022_LoL_esports_match_data_from_OraclesElixir.csv",
        "D:\\Github Repos\\csci-567\\feature_data\\new\\2023_LoL_esports_match_data_from_OraclesElixir.csv",
        "D:\\Github Repos\\csci-567\\feature_data\\new\\2024_LoL_esports_match_data_from_OraclesElixir.csv"
    ]

    combine_datasets("D:\\Github Repos\\csci-567\\feature_data\\new")

    # startTime = time.time()

    # for item in lt_files:
    #     import_dataset_csv(item)

    # endTime = time.time()

    # elapsed_time = round(endTime - startTime, 3)
    # print(f"Execution time: {elapsed_time}") 