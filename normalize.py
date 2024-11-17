import pandas as pd
import random

def zscore_normalize(filepath_input, filepath_output):
    df = pd.read_csv(filepath_input)

    df_z_scaled = df.copy()
    for column in df_z_scaled.columns:
        if column == 'bResult':
            continue
        df_z_scaled[column] = (df_z_scaled[column] - df_z_scaled[column].mean()) / df_z_scaled[column].std() 

    df_z_scaled.to_csv(filepath_output, index=False)
    return

if __name__ == "__main__":
    import sys
    train = False
    test = False
    include_champion_role_features = False

    filename_with_champion_role = "featureInput"
    filename_no_champion_role = "featureInput_noChampionRole"
    filename_extension = "csv"
    filename_output_marker = "ZSCORED"
    filename = filename_no_champion_role

    # Handle command line arguments.
    args = sys.argv[1:]
    for i in range(len(args)):
        if (args[i] == "-train"):
            train = True
        elif (args[i] == "-test"):
            test = True
        elif (args[i] == "-c" or args[i] == "-C"):
            filename = filename_with_champion_role
        elif (args[i] == "-seed"):
            i += 1
            seed = args[i]
            random.seed(seed)
    
    if (not train and not test):
        print("ERROR: Must train or test. Niether selected.")
        exit(1)
    
    if (train and test):
        print("ERROR: Can not both train and test.")
        exit(1)
    
    feature_folder = "feature_data"
    feature_subfolder = ""
    if (train):
        feature_subfolder = "train"
        print("Normalizing feature data: Training Set.")
    if (test):
        feature_subfolder = "test"
        print("Normalizing feature data: Test Set.")
    filepath_input = "{}/{}/{}.{}".format(feature_folder, feature_subfolder, filename, filename_extension)
    filepath_output = "{}/{}/{}_{}.{}".format(feature_folder, feature_subfolder, filename, filename_output_marker, filename_extension)

    zscore_normalize(filepath_input=filepath_input, filepath_output=filepath_output)
