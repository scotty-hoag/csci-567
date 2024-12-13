import pandas as pd

data = pd.read_csv('../../lol_data/matchinfo.csv')

teams = set(list(data['blueTeamTag'].unique()) + list(data['redTeamTag'].unique()))
len(teams)

cleaned_df = data[(data['blueTeamTag'].isnull() == False) & (data['redTeamTag'].isnull() == False)]

test = cleaned_df.sample(frac=0.1, random_state=3)
test.to_csv('test.csv')
train = cleaned_df.drop(test.index)
train.to_csv('train.csv')

from sklearn.impute import KNNImputer

blue_cols = ['factorized_blueTop', 'factorized_blueJungle', 'factorized_blueMiddle', 'factorized_blueADC',
             'factorized_blueSupport']
red_cols = ['factorized_redTop', 'factorized_redJungle', 'factorized_redMiddle', 'factorized_redADC', 'factorized_redSupport']
blue_champ_cols = ['factorized_blueTopChamp', 'factorized_blueJungleChamp', 'factorized_blueMiddleChamp', 'factorized_blueADCChamp',
                   'factorized_blueSupportChamp']
red_champ_cols = ['factorized_redTopChamp', 'factorized_redJungleChamp', 'factorized_redMiddleChamp', 'factorized_redADCChamp',
                  'factorized_redSupportChamp']

new_cols = 'bResult, btPlayerRole, bjPlayerRole, bmPlayerRole, baPlayerRole, bsPlayerRole, rtPlayerRole, rjPlayerRole, \
rmPlayerRole, raPlayerRole, rsPlayerRole, btPlayerChampion, bjPlayerChampion, bmPlayerChampion, baPlayerChampion, \
bsPlayerChampion, rtPlayerChampion, rjPlayerChampion, rmPlayerChampion, raPlayerChampion, rsPlayerChampion, \
bCoopPlayer, rCoopPlayer, bCoopChampion, rCoopChampion, vsPlayer, vsChampion, bTeamColor, rTeamColor'.split(', ')

roles = ['top', 'jungle', 'middle', 'adc', 'support']
player_roles = {roles[x]: [blue_cols[x], red_cols[x]] for x in range(len(roles))}
champ_roles = {roles[x]: [blue_champ_cols[x], red_champ_cols[x]] for x in range(len(roles))}

def switch(b_win):
    if b_win == 1:
        return 0
    elif b_win == 0:
        return 1
    else:
        "fatal error. b_win must be either 0 or 1."
        exit(1)

def createPlayerRole(col_name, col_alt, player, b_win, df):
    matches = len(df[df[col_name] == player])
    matches += len(df[df[col_alt] == player])
    wins = len(df[(df[col_name] == player) & (df['bResult'] == b_win)])
    wins += len(df[(df[col_alt] == player) & (df['bResult'] == switch(b_win))])
    if matches == 0:
        return 0.5
    else:
        return wins / matches

def createPlayerChampion(player, champ, df):
    df_blue = df[(df[blue_cols[0]] == player) | (df[blue_cols[1]] == player) | \
                 (df[blue_cols[2]] == player) | (df[blue_cols[3]] == player) | \
                 (df[blue_cols[4]] == player)]
    df_red = df[(df[red_cols[0]] == player) | (df[red_cols[1]] == player) | \
                (df[red_cols[2]] == player) | (df[red_cols[3]] == player) | \
                (df[red_cols[4]] == player)]
    filtered_blue = df_blue[(df_blue[blue_champ_cols[0]] == champ) | (df_blue[blue_champ_cols[1]] == champ) | \
                            (df_blue[blue_champ_cols[2]] == champ) | (df_blue[blue_champ_cols[3]] == champ) | \
                            (df_blue[blue_champ_cols[4]] == champ)]
    filtered_red = df_red[(df_red[red_champ_cols[0]] == champ) | (df_red[red_champ_cols[1]] == champ) | \
                          (df_red[red_champ_cols[2]] == champ) | (df_red[red_champ_cols[3]] == champ) | \
                          (df_red[red_champ_cols[4]] == champ)]

    matches = len(filtered_blue)
    matches += len(filtered_red)

    wins = len(filtered_blue[filtered_blue['bResult'] == 1])
    wins += len(filtered_red[filtered_red['bResult'] == 0])

    if matches == 0:
        return 0.5
    else:
        return wins / matches

def createCoopPlayers(players, b_cols, r_cols, df):
    total = 0
    for i in range(len(players)):
        sub_total = 0
        df_blue = df[(df[b_cols[0]] == players[i]) | (df[b_cols[1]] == players[i]) | \
                     (df[b_cols[2]] == players[i]) | (df[b_cols[3]] == players[i]) | \
                     (df[b_cols[4]] == players[i])]
        df_red = df[(df[r_cols[0]] == players[i]) | (df[r_cols[1]] == players[i]) | \
                    (df[r_cols[2]] == players[i]) | (df[r_cols[3]] == players[i]) | \
                    (df[r_cols[4]] == players[i])]
        for j in range(len(players)):
            if i != j:
                filtered_blue = df_blue[(df_blue[b_cols[0]] == players[j]) | (df_blue[b_cols[1]] == players[j]) | \
                                        (df_blue[b_cols[2]] == players[j]) | (df_blue[b_cols[3]] == players[j]) | \
                                        (df_blue[b_cols[4]] == players[j])]

                filtered_red = df_red[(df_red[r_cols[0]] == players[j]) | (df_red[r_cols[1]] == players[j]) | \
                                      (df_red[r_cols[2]] == players[j]) | (df_red[r_cols[3]] == players[j]) | \
                                      (df_red[r_cols[4]] == players[j])]

                matches = len(filtered_blue)
                matches += len(filtered_red)

                wins = len(filtered_blue[filtered_blue['bResult'] == 1])
                wins += len(filtered_red[filtered_red['bResult'] == 0])

                ratio = 0
                if matches == 0:
                    ratio += 0.5
                else:
                    ratio += wins / matches
                sub_total += ratio
        total += sub_total
    return total

def createVSPlayer(blue_players, red_players, b_cols, r_cols, df):
    total = 0
    for bp in blue_players:
        sub_total = 0
        df_blue = df[(df[b_cols[0]] == bp) | (df[b_cols[1]] == bp) | \
                     (df[b_cols[2]] == bp) | (df[b_cols[3]] == bp) | \
                     (df[b_cols[4]] == bp)]
        df_red = df[(df[r_cols[0]] == bp) | (df[r_cols[1]] == bp) | \
                    (df[r_cols[2]] == bp) | (df[r_cols[3]] == bp) | \
                    (df[r_cols[4]] == bp)]
        for rp in red_players:
            filtered_blue = df_blue[(df_blue[r_cols[0]] == rp) | (df_blue[r_cols[1]] == rp) | \
                                    (df_blue[r_cols[2]] == rp) | (df_blue[r_cols[3]] == rp) | \
                                    (df_blue[r_cols[4]] == rp)]
            filtered_red = df_red[(df_red[b_cols[0]] == rp) | (df_red[b_cols[1]] == rp) | \
                                  (df_red[b_cols[2]] == rp) | (df_red[b_cols[3]] == rp) | \
                                  (df_red[b_cols[4]] == rp)]

            matches = len(filtered_blue)
            matches += len(filtered_red)

            wins = len(filtered_blue[filtered_blue['bResult'] == 1])
            wins += len(filtered_red[filtered_red['bResult'] == 0])

            ratio = 0
            if matches == 0:
                ratio += 0.5
            else:
                ratio += wins / matches
            sub_total += ratio
        total += sub_total
    return total

def createTeamColor(team_tag, color, df):
    filtered_df = None
    result = 0
    if color == 'blue':
        filtered_df = df[df['factorized_blueTeamTag'] == team_tag]
        result += 1
    elif color == 'red':
        filtered_df = df[df['factorized_redTeamTag'] == team_tag]
    else:
        print('error: please pass either red or blue for the team color.')
        exit(1)
    matches = len(filtered_df)
    wins = len(filtered_df[filtered_df['bResult'] == result])

    if matches == 0:
        return 0.5
    else:
        return wins / matches

def createFinalDF(df):
    final_df = pd.DataFrame(data=[[0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0]], columns=new_cols)
    for index, row in df.iterrows():
        # btPlayerRole
        bt_player = row['factorized_blueTop']
        bt_player_role = createPlayerRole(player_roles['top'][0], player_roles['top'][1], bt_player, 1, df)
        # bjPlayerRole
        bj_player = row['factorized_blueJungle']
        bj_player_role = createPlayerRole(player_roles['jungle'][0], player_roles['jungle'][1], bj_player, 1, df)
        # bmPlayerRole
        bm_player = row['factorized_blueMiddle']
        bm_player_role = createPlayerRole(player_roles['middle'][0], player_roles['middle'][1], bm_player, 1, df)
        # baPlayerRole
        ba_player = row['factorized_blueADC']
        ba_player_role = createPlayerRole(player_roles['adc'][0], player_roles['adc'][1], ba_player, 1, df)
        # bsPlayerRole
        bs_player = row['factorized_blueSupport']
        bs_player_role = createPlayerRole(player_roles['support'][0], player_roles['support'][1], bs_player, 1, df)
        # rtPlayerRole
        rt_player = row['factorized_redTop']
        rt_player_role = createPlayerRole(player_roles['top'][1], player_roles['top'][0], rt_player, 0, df)
        # rjPlayerRole
        rj_player = row['factorized_redJungle']
        rj_player_role = createPlayerRole(player_roles['jungle'][1], player_roles['jungle'][0], rj_player, 0, df)
        # rmPlayerRole
        rm_player = row['factorized_redMiddle']
        rm_player_role = createPlayerRole(player_roles['middle'][1], player_roles['middle'][0], rm_player, 0, df)
        # raPlayerRole
        ra_player = row['factorized_redADC']
        ra_player_role = createPlayerRole(player_roles['adc'][1], player_roles['adc'][0], ra_player, 0, df)
        # rsPlayerRole
        rs_player = row['factorized_redSupport']
        rs_player_role = createPlayerRole(player_roles['support'][1], player_roles['support'][0], rs_player, 0, df)
        # btPlayerChampion
        bt_champ = row['factorized_blueTopChamp']
        bt_player_champion = createPlayerChampion(bt_player, bt_champ, df)
        # bjPlayerChampion
        bj_champ = row['factorized_blueJungleChamp']
        bj_player_champion = createPlayerChampion(bj_player, bj_champ, df)
        # bmPlayerChampion
        bm_champ = row['factorized_blueMiddleChamp']
        bm_player_champion = createPlayerChampion(bm_player, bm_champ, df)
        # baPlayerChampion
        ba_champ = row['factorized_blueADCChamp']
        ba_player_champion = createPlayerChampion(ba_player, ba_champ, df)
        # bsPlayerChampion
        bs_champ = row['factorized_blueSupportChamp']
        bs_player_champion = createPlayerChampion(bs_player, bs_champ, df)
        # rtPlayerChampion
        rt_champ = row['factorized_redTopChamp']
        rt_player_champion = createPlayerChampion(rt_player, rt_champ, df)
        # rjPlayerChampion
        rj_champ = row['factorized_redJungleChamp']
        rj_player_champion = createPlayerChampion(rj_player, rj_champ, df)
        # rmPlayerChampion
        rm_champ = row['factorized_redMiddleChamp']
        rm_player_champion = createPlayerChampion(rm_player, rm_champ, df)
        # raPlayerChampion
        ra_champ = row['factorized_redADCChamp']
        ra_player_champion = createPlayerChampion(ra_player, ra_champ, df)
        # rsPlayerChampion
        rs_champ = row['factorized_redSupportChamp']
        rs_player_champion = createPlayerChampion(rs_player, rs_champ, df)
        # bCoopPlayer
        blue_players = [bt_player, bj_player, bm_player, ba_player, bs_player]
        b_coop_player = createCoopPlayers(blue_players, blue_cols, red_cols, df)
        # rCoopPlayer
        red_players = [rt_player, rj_player, rm_player, ra_player, rs_player]
        r_coop_player = createCoopPlayers(red_players, blue_cols, red_cols, df)
        # bCoopChampion
        blue_champs = [bt_champ, bj_champ, bm_champ, ba_champ, bs_champ]
        b_coop_champion = createCoopPlayers(blue_champs, blue_champ_cols, red_champ_cols, df)
        # rCoopChampion
        red_champs = [rt_champ, rj_champ, rm_champ, ra_champ, rs_champ]
        r_coop_champion = createCoopPlayers(red_champs, blue_champ_cols, red_champ_cols, df)
        # vsPlayer
        vs_player = createVSPlayer(blue_players, red_players, blue_cols, red_cols, df)
        # vsChampion
        vs_champion = createVSPlayer(blue_champs, red_champs, blue_champ_cols, red_champ_cols, df)
        # bTeamColor
        # b_team_color = createTeamColor(blue_players + blue_champs, blue_cols + blue_champ_cols, 1)
        b_team_tag = row['factorized_blueTeamTag']
        # b_team = [bt_player, bj_player, bm_player, ba_player, bs_player, bt_champ, bj_champ, bm_champ, ba_champ,
        #           bs_champ]
        b_team_color = createTeamColor(b_team_tag, 'blue', df)
        # rTeamColor
        # r_team_color = createTeamColor(red_players + red_champs, red_cols + red_champ_cols, 0)
        r_team_tag = row['factorized_redTeamTag']
        # r_team = [rt_player, rj_player, rm_player, ra_player, rs_player, rt_champ, rj_champ, rm_champ, ra_champ,
        #           rs_champ]
        r_team_color = createTeamColor(r_team_tag, 'red', df)

        new_col_values = [row['bResult'], bt_player_role, bj_player_role, bm_player_role, ba_player_role,
                          bs_player_role, rt_player_role, rj_player_role, rm_player_role, ra_player_role,
                          rs_player_role, bt_player_champion, bj_player_champion, bm_player_champion,
                          ba_player_champion, bs_player_champion, rt_player_champion, rj_player_champion,
                          rm_player_champion, ra_player_champion, rs_player_champion, b_coop_player, r_coop_player,
                          b_coop_champion, r_coop_champion, vs_player, vs_champion, b_team_color, r_team_color]
        new_row = {}
        for new_col in range(len(new_col_values)):
            new_row[new_cols[new_col]] = new_col_values[new_col]
        print(new_row)
        final_df = final_df._append(new_row, ignore_index=True)
    return final_df

# performs preprocessing steps
# input: pandas dataframe, filename for the name of the csv file that the processed dataframe should be saved to
def processDF(df, filename):
    imputer = KNNImputer(n_neighbors=5)
    factorized_cols = {}
    df = df.drop(['League', 'Year', 'Season', 'Type', 'Address', 'gamelength'], axis=1)
    need_factorizing = list(df.columns)
    need_factorizing.pop(1)
    need_factorizing.pop(1)
    imputation_index = {}
    for col in need_factorizing:
        factorization = pd.factorize(df[col])
        factorized_cols[col] = factorization[0]
        imputation_index[col] = list(factorization[1])
    factorized_df = df[['bResult', 'rResult']].copy(deep=True)
    for col in need_factorizing:
        col_name = 'factorized_' + col
        factorized_df[col_name] = factorized_cols[col]
    imputed_df = imputer.fit_transform(factorized_df)
    prepped_df = pd.DataFrame(data=imputed_df, columns=list(factorized_df.columns))
    col_types = {}
    for col in list(prepped_df.columns):
        col_types[col] = 'object'
    prepped_df = prepped_df.astype(col_types)
    for col in list(prepped_df.columns):
        col_name = col[11:]
        if col_name in imputation_index.keys():
            for index in range(len(imputation_index[col_name])):
                replacement_val = imputation_index[col_name][index]
                prepped_df.loc[prepped_df[col] == index, col] = replacement_val
    final_df = createFinalDF(prepped_df)
    final_df = final_df.drop(0)
    scaled_df = final_df.copy(deep=True)
    for column in scaled_df.columns:
        if column != 'bResult':
            scaled_df[column] = (scaled_df[column] - scaled_df[column].mean()) / scaled_df[column].std()
    scaled_df.to_csv(filename)

processDF(train, 'train_update.csv')
processDF(test, 'test_update.csv')