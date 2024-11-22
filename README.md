# csci-567
Repository for the USC CSCI-567 semester group project.

Features defined in the paper are calculated and placed in the following places:


1. playerRole_i,j: player_data.csv, "Win Ratio"

2. playerChampion_i,j: player_wins_with_each_champion_data.csv

3. coopPlayer_blue: match_vs_coop_data.csv, "sumCoopBluePlayers"
   coopPlayer_red: match_vs_coop_data.csv, "sumCoopRedPlayers"

4. vsPlayer: match_vs_coop_data.csv, "sumVsBluePlayers", "sumVsRedPlayers"
    You should only need the sumVsBluePlayers stat.
    Individual summmed player ratios under the roll name + Vs.
    Individual player win ratios in player_vs_data.csv.

5. championRole_i,j: champion_data.csv, "Win Ratio"

6. coopChampion_blue: match_vs_coop_data.csv, "sumCoopBlueChampions"
   coopChampion_red: match_vs_coop_data.csv, "sumCoopRedChampions"

7. vsChampion: match_vs_coop_data.csv, "sumVsBlueChampions", "sumVsRedChampions"
    You should only need the sumVsBlueChampions stat.
    Individual summmed champion ratios under the roll name + Vs.
    Individual champion win ratios in champion_vs_data.csv.

8. teamColor_j: team_data.csv, Win Ratio Blue, Win Ratio Red


To re-process data into training and testing sets, run the "makedata.bat" script. Or, manually run the following commands to generate data with the dataset split into approx 90% training data and 10% testing data.

python3 makedata.py -makeog
python3 makedata.py -makeinfo 0.90
python3 makedata.py -train
python3 makedata.py -test


Example output of makedata.bat:


C:\...\csci-567>makedata.bat

Made new SPLIT feature training and test data.
Num lines: 7620, Num Lines Valid: 7582, Num Lines Removed: 38, Lines Training: 6817 (0.899), Lines Test: 765 (0.101)
Creating new feature data: Training Set.
Done.
Creating new feature data: Test Set.
Done.


Neural Network

To get the neural net model from code/neuralnetwork.py, call the get_lol_nnet_model method.

Parameters:
- train_model: Bool. If true, load the training and testing data and train the model before returning it. If false, return the model without training. 
- in_training_type: Enum selector for hyperparameter set to use. Default is to use the hyperparameters discovered from tuning.
- in_random_seed: Random seed to use for training. Default is 42. 
- in_training_data_path: Relative path to the training data.
- in_testing_data_path: Relative path to the testing data.
- print_debug: If true, print the training status of the model.fit method. If false, do not print any of the training output.