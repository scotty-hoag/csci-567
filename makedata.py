import io
import csv

# [player_name]
glob_numMatches = {}
glob_numMatchesWon = {}

# [roll_name][player_name]
glob_numMatchesInRole = {}
glob_numMatchesInRoleWon = {}

# [champion_name][player_name]
glob_numChampionPlaysPerPlayer = {}
glob_numChampionPlaysPerPlayerWins = {}

# (set of all player names)
glob_playerNamesSet = set()
glob_playerNames = []
# (set of all champion names)
glob_championNamesSet = set()
glob_championNames = []
# (set of all team names)
glob_teamNamesSet = set()
glob_teamNames = []


glob_rolesBlue =  ['blueTop',        'blueJungle',       'blueMiddle',       'blueADC',      'blueSupport']
glob_rolesRed  =  ['redTop',         'redJungle',        'redMiddle',        'redADC',       'redSupport' ]
glob_champsBlue = ['blueTopChamp',   'blueJungleChamp',  'blueMiddleChamp',  'blueADCChamp', 'blueSupportChamp']
glob_champsRed  = ['redTopChamp',    'redJungleChamp',   'redMiddleChamp',   'redADCChamp',  'redSupportChamp']
glob_teamTagsBlue = ['blueTeamTag']
glob_teamTagsRed  = ['redTeamTag']

glob_rolesAll  = glob_rolesBlue  + glob_rolesRed
glob_champsAll = glob_champsBlue + glob_champsRed
glob_teamTagsAll = glob_teamTagsBlue + glob_teamTagsRed


class Player:
    name = ""

    gamesPlayed = 0
    gamesWon = 0
    
    rolesPlayed = {}
    rolesWon = {}

    championsPlayed = {}
    championsWon = {}

    vsPlayerPlayed = {}
    vsPlayerWon = {}

    def __init__(self):
        self.rolesPlayed = {}
        self.rolesWon = {}

        self.championsPlayed = {}
        self.championsWon = {}

        self.vsPlayerPlayed = {}
        self.vsPlayerWon = {}
glob_players = {}

class Champion:
    name = ""

    gamesPlayed = 0
    gamesWon = 0

    rolesPlayed = {}
    rolesWon = {}
    
    vsChampionPlayed = {}
    vsChampionWon = {}

    def __init__(self):
        self.rolesPlayed = {}
        self.rolesWon = {}

        self.vsChampionPlayed = {}
        self.vsChampionWon = {}
glob_champions = {}

class Team:
    name = ""

    gamesPlayed = 0
    gamesWon = 0
    
    gamesPlayedBlue = 0
    gamesWonBlue = 0

    gamesPlayedRed = 0
    gamesWonRed = 0
    
    def __init__(self):
        pass
glob_teams = {}


def IsRowValid(row):
    # For now, skip rows that don't have complete information.
    bValid = not (row["blueTeamTag"] == "" or row["redTeamTag"] == "")
    return bValid

# Make fixed match data.
# Remove invalid data rows.
# Attach row indicies.
with open('lol_data/matchinfo.csv', newline='', encoding='utf-8') as csvFile:
    reader = csv.DictReader(csvFile)
    rowNumber = 0
    outputString = ""
    for row in reader:
        # For now, skip rows that don't have complete information.
        if (rowNumber == 0):
            outputString += "index,"
            for key in list(row.keys()):
                outputString += "{},".format(key)
            outputString += "\n"

        if not IsRowValid(row):
            continue

        outputString += "{},".format(rowNumber)
        rowNumber += 1
        for value in list(row.values()):
            outputString += "{},".format(value)
        outputString += "\n"

    file = open("feature_data/match_info.csv", 'w+')
    file.write(outputString)
    file.close()




with open('feature_data/match_info.csv', newline='', encoding='utf-8') as csvFile:
    reader = csv.DictReader(csvFile)
    for row in reader:
        # For now, skip rows that don't have complete information.
        if not IsRowValid(row):
            continue

        # Player, champion, and team names are added to SETs, enforcing uniqueness.
        # After we fill the sets, make sorted lists.
        for role in glob_rolesAll:
            glob_playerNamesSet.add(row[role])
        glob_playerNames = sorted(glob_playerNamesSet)
        for champion in glob_champsAll:
            glob_championNamesSet.add(row[champion])
        glob_championNames = sorted(glob_championNamesSet)
        for teamTag in glob_teamTagsAll:
            glob_teamNamesSet.add(row[teamTag])
        glob_teamNames = sorted(glob_teamNamesSet)
        

    for role in glob_rolesAll:
        glob_numMatchesInRole[role] = {}
        glob_numMatchesInRoleWon[role] = {}

    # Record the player role and champion win stats.
    for playerName in glob_playerNames:
        player = Player()
        player.name = playerName
        glob_players[playerName] = player

        # Init the number of plays/wins 
        for role in glob_rolesAll:
            player.rolesPlayed[role] = 0
            player.rolesWon[role] = 0

        # Init the number of champion plays and wins.
        for champ in glob_championNames:
            player.championsPlayed[champ] = 0
            player.championsWon[champ] = 0
        
        # Init the number of vs player plays and wins.
        for vsPlayerName in glob_playerNames:
            player.vsPlayerPlayed[vsPlayerName] = 0
            player.vsPlayerWon[vsPlayerName] = 0
    
    # Record the champion role win stats.
    for championName in glob_championNames:
        champion = Champion()
        champion.name = championName
        glob_champions[championName] = champion

        # Init the number of plays/wins 
        for role in glob_rolesAll:
            champion.rolesPlayed[role] = 0
            champion.rolesWon[role] = 0

        # Init the number of vs champion plays and wins.
        for vsChampionName in glob_championNames:
            champion.vsChampionPlayed[vsChampionName] = 0
            champion.vsChampionWon[vsChampionName] = 0

    # Record the team stats.
    for teamName in glob_teamNames:
        team = Team()
        team.name = teamName
        glob_teams[teamName] = team


with open('feature_data/match_info.csv', newline='', encoding='utf-8') as csvFile:
    reader = csv.DictReader(csvFile)
    for row in reader:
        # For now, skip rows that don't have complete information.
        if not IsRowValid(row):
            continue

        # Find all of the wins for each player in each role.
        winBlue = int(row['bResult'])
        winRed = int(row['rResult'])
        winningTeamRoles = []
        losingTeamRoles = []
        winningTeamChamps = []
        losingTeamChamps = []
        
        teamNameBlue = row[glob_teamTagsBlue[0]]
        teamNameRed = row[glob_teamTagsRed[0]]

        teamBlue = glob_teams[teamNameBlue]
        teamRed = glob_teams[teamNameRed]

        teamBlue.gamesPlayed += 1
        teamRed.gamesPlayed += 1

        if winBlue == 1:
            winningTeamRoles = glob_rolesBlue
            losingTeamRoles = glob_rolesRed

            winningTeamChamps = glob_champsBlue
            losingTeamChamps = glob_champsRed

            teamBlue.gamesPlayedBlue += 1
            teamBlue.gamesWonBlue += 1
            teamBlue.gamesWon += 1

            teamRed.gamesPlayedRed += 1
        else:
            winningTeamRoles = glob_rolesRed
            losingTeamRoles = glob_rolesBlue

            winningTeamChamps = glob_champsRed
            losingTeamChamps = glob_champsBlue

            teamRed.gamesPlayedRed += 1
            teamRed.gamesWonRed += 1
            teamRed.gamesWon += 1

            teamBlue.gamesPlayedBlue += 1
        
        for roleIndex in range(len(losingTeamRoles)):
            role = losingTeamRoles[roleIndex]
            playerName = row[role]
            player = glob_players[playerName]

            championRole = losingTeamChamps[roleIndex]
            championName = row[championRole]
            champion = glob_champions[championName]

            # Inc matches played by the player/champion.
            player.gamesPlayed += 1
            champion.gamesPlayed += 1

            # Inc matches played by the player/champion in the specific role.
            player.rolesPlayed[role] += 1
            champion.rolesPlayed[role] += 1

            # Inc matches played by the player with the specific Champion.
            player.championsPlayed[championName] += 1

            # For each member of the other team, record +1 vs play.
            for vsRole in winningTeamRoles:
                vsPlayerName = row[vsRole]
                player.vsPlayerPlayed[vsPlayerName] += 1
            for vsChampionRole in winningTeamChamps:
                vsChampionName = row[vsChampionRole]
                champion.vsChampionPlayed[vsChampionName] += 1

        for roleIndex in range(len(winningTeamRoles)):
            role = winningTeamRoles[roleIndex]
            playerName = row[role]
            player = glob_players[playerName]

            championRole = winningTeamChamps[roleIndex]
            championName = row[championRole]
            champion = glob_champions[championName]
            
            # Inc matches played and won by the player/champion.
            player.gamesPlayed += 1
            player.gamesWon += 1
            champion.gamesPlayed += 1
            champion.gamesWon += 1

            # Inc matches played and won by the player/champion in the specific role.
            player.rolesPlayed[role] += 1
            player.rolesWon[role] += 1
            champion.rolesPlayed[role] += 1
            champion.rolesWon[role] += 1

            # Inc matches played and won by the player with the specific Champion.
            player.championsPlayed[championName] += 1
            player.championsWon[championName] += 1
            
            # For each member of the other team, record +1 vs play and +1 vs win.
            for vsRole in losingTeamRoles:
                vsPlayerName = row[vsRole]
                player.vsPlayerPlayed[vsPlayerName] += 1
                player.vsPlayerWon[vsPlayerName] += 1
            for vsChampionRole in losingTeamChamps:
                vsChampionName = row[vsChampionRole]
                champion.vsChampionPlayed[vsChampionName] += 1
                champion.vsChampionWon[vsChampionName] += 1

def WritePlayerData():
    outputString = "Name," + \
        "Games Played,Games Won,Win Ratio," + \
        "Blue Top Plays,Red Top Plays,Top Plays," + \
        "Blue Top Wins,Red Top Wins,Top Wins," + \
        "Blue Jungle, Red Jungle, Jungle," + \
        "Blue Jungle Wins, Red Jungle Wins, Jungle Wins," + \
        "Blue Middle, Red Middle, Middle," + \
        "Blue Middle Wins, Red Middle Wins, Middle Wins," + \
        "Blue ADC, Red ADC, ADC," + \
        "Blue ADC Wins, Red ADC Wins, ADC Wins," + \
        "Blue Support, Red Support, Support," + \
        "Blue Support Wins, Red Support Wins, Support Wins," + \
        "\n"

    rBlue = glob_rolesBlue
    rReds = glob_rolesRed 
    for playerName, player in glob_players.items():
        if player == None or player.name == "":
            continue
        playerWinRatio = ((player.gamesWon/player.gamesPlayed) if (player.gamesPlayed > 0) else 0)
        outputString += "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format( \
            player.name \
            , player.gamesPlayed, player.gamesWon, playerWinRatio \
            , player.rolesPlayed[rBlue[0]], player.rolesPlayed[rReds[0]], player.rolesPlayed[rBlue[0]] + player.rolesPlayed[rReds[0]] \
            , player.rolesWon[   rBlue[0]], player.rolesWon[   rReds[0]], player.rolesWon   [rBlue[0]] + player.rolesWon[   rReds[0]] \
            , player.rolesPlayed[rBlue[1]], player.rolesPlayed[rReds[1]], player.rolesPlayed[rBlue[1]] + player.rolesPlayed[rReds[1]] \
            , player.rolesWon[   rBlue[1]], player.rolesWon[   rReds[1]], player.rolesWon   [rBlue[1]] + player.rolesWon[   rReds[1]] \
            , player.rolesPlayed[rBlue[2]], player.rolesPlayed[rReds[2]], player.rolesPlayed[rBlue[2]] + player.rolesPlayed[rReds[2]] \
            , player.rolesWon[   rBlue[2]], player.rolesWon[   rReds[2]], player.rolesWon   [rBlue[2]] + player.rolesWon[   rReds[2]] \
            , player.rolesPlayed[rBlue[3]], player.rolesPlayed[rReds[3]], player.rolesPlayed[rBlue[3]] + player.rolesPlayed[rReds[3]] \
            , player.rolesWon[   rBlue[3]], player.rolesWon[   rReds[3]], player.rolesWon   [rBlue[3]] + player.rolesWon[   rReds[3]] \
            , player.rolesPlayed[rBlue[4]], player.rolesPlayed[rReds[4]], player.rolesPlayed[rBlue[4]] + player.rolesPlayed[rReds[4]] \
            , player.rolesWon[   rBlue[4]], player.rolesWon[   rReds[4]], player.rolesWon   [rBlue[4]] + player.rolesWon[   rReds[4]] \
            )
    file = open("feature_data/player_data.csv", 'w+')
    file.write(outputString)
    file.close()

def WriteChampionData():
    outputString = "Name," + \
        "Games Played,Games Won,Win Ratio," + \
        "Blue Top Plays,Red Top Plays,Top Plays," + \
        "Blue Top Wins,Red Top Wins,Top Wins," + \
        "Blue Jungle, Red Jungle, Jungle," + \
        "Blue Jungle Wins, Red Jungle Wins, Jungle Wins," + \
        "Blue Middle, Red Middle, Middle," + \
        "Blue Middle Wins, Red Middle Wins, Middle Wins," + \
        "Blue ADC, Red ADC, ADC," + \
        "Blue ADC Wins, Red ADC Wins, ADC Wins," + \
        "Blue Support, Red Support, Support," + \
        "Blue Support Wins, Red Support Wins, Support Wins," + \
        "\n"

    rBlue = glob_rolesBlue
    rReds = glob_rolesRed 
    for championName, champion in glob_champions.items():
        if champion == None or champion.name == "":
            continue
        championWinRatio = ((champion.gamesWon/champion.gamesPlayed) if (champion.gamesPlayed > 0) else 0)
        outputString += "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format( \
            champion.name \
            , champion.gamesPlayed, champion.gamesWon, championWinRatio \
            , champion.rolesPlayed[rBlue[0]], champion.rolesPlayed[rReds[0]], champion.rolesPlayed[rBlue[0]] + champion.rolesPlayed[rReds[0]] \
            , champion.rolesWon[   rBlue[0]], champion.rolesWon[   rReds[0]], champion.rolesWon   [rBlue[0]] + champion.rolesWon[   rReds[0]] \
            , champion.rolesPlayed[rBlue[1]], champion.rolesPlayed[rReds[1]], champion.rolesPlayed[rBlue[1]] + champion.rolesPlayed[rReds[1]] \
            , champion.rolesWon[   rBlue[1]], champion.rolesWon[   rReds[1]], champion.rolesWon   [rBlue[1]] + champion.rolesWon[   rReds[1]] \
            , champion.rolesPlayed[rBlue[2]], champion.rolesPlayed[rReds[2]], champion.rolesPlayed[rBlue[2]] + champion.rolesPlayed[rReds[2]] \
            , champion.rolesWon[   rBlue[2]], champion.rolesWon[   rReds[2]], champion.rolesWon   [rBlue[2]] + champion.rolesWon[   rReds[2]] \
            , champion.rolesPlayed[rBlue[3]], champion.rolesPlayed[rReds[3]], champion.rolesPlayed[rBlue[3]] + champion.rolesPlayed[rReds[3]] \
            , champion.rolesWon[   rBlue[3]], champion.rolesWon[   rReds[3]], champion.rolesWon   [rBlue[3]] + champion.rolesWon[   rReds[3]] \
            , champion.rolesPlayed[rBlue[4]], champion.rolesPlayed[rReds[4]], champion.rolesPlayed[rBlue[4]] + champion.rolesPlayed[rReds[4]] \
            , champion.rolesWon[   rBlue[4]], champion.rolesWon[   rReds[4]], champion.rolesWon   [rBlue[4]] + champion.rolesWon[   rReds[4]] \
            )
    file = open("feature_data/champion_data.csv", 'w+')
    file.write(outputString)
    file.close()

def WritePlayerWinsWithEachChampionData():
    outputString = "Player Name"
    for championName, champion in glob_champions.items():
            if champion == None or champion.name == "":
                continue
            outputString += ",{}".format(championName)
    outputString += "\n"

    # For each player,
    #    For each champion,
    #        win ratio = games won / games played
    for playerName, player in glob_players.items():
        if player == None or player.name == "":
            continue
        
        outputString += playerName
        for championName, champion in glob_champions.items():
            if champion == None or champion.name == "":
                continue
            
            playerWinRatio = ((player.championsWon[championName]/player.championsPlayed[championName]) if (player.championsPlayed[championName] > 0) else 0)

            outputString += ",{}".format(playerWinRatio)
        outputString += "\n"

    file = open("feature_data/player_wins_with_each_champion_data.csv", 'w+')
    file.write(outputString)
    file.close()


def WriteTeamData():
    outputString = "teamTag" + \
        ",Games Played,Games Won,Win Ratio" + \
        ",Games Played Blue,Games Won Blue,Win Ratio Blue" + \
        ",Games Played Red,Games Won Red,Win Ratio Red" + \
        "\n"

    for teamName, team in glob_teams.items():
        if team == None or teamName == "":
            continue

        winRatio = ((team.gamesWon/team.gamesPlayed) if (team.gamesPlayed > 0) else 0)
        winRatioBlue = ((team.gamesWonBlue/team.gamesPlayedBlue) if (team.gamesPlayedBlue > 0) else 0)
        winRatioRed = ((team.gamesWonRed/team.gamesPlayedRed) if (team.gamesPlayedRed > 0) else 0)
        outputString += "{},{},{},{},{},{},{},{},{},{},\n".format(team.name \
            , team.gamesPlayed,     team.gamesWon,     winRatio     \
            , team.gamesPlayedBlue, team.gamesWonBlue, winRatioBlue \
            , team.gamesPlayedRed,  team.gamesWonRed,  winRatioRed  \
        )

    file = open("feature_data/team_data.csv", 'w+')
    file.write(outputString)
    file.close()


def WritePlayerVsData():
    outputString = "Name"
    for playerName, player in glob_players.items():
            if player == None or playerName == "":
                continue
            outputString += ",{}".format(playerName)
    outputString += "\n"

    # For each player,
    #    For each other player,
    #        win ratio = games won / games played
    for playerName, player in glob_players.items():
        if player == None or playerName == "":
            continue
        
        outputString += playerName
        for vsPlayerName, vsPlayer in glob_players.items():
            if vsPlayer == None or vsPlayerName == "":
                continue
            
            playerWinRatio = ((player.vsPlayerWon[vsPlayerName]/player.vsPlayerPlayed[vsPlayerName]) if (player.vsPlayerPlayed[vsPlayerName] > 0) else 0)

            outputString += ",{}".format(playerWinRatio)
        outputString += "\n"

    file = open("feature_data/player_vs_data.csv", 'w+')
    file.write(outputString)
    file.close()
    
def WriteChampionVsData():
    outputString = "Name"
    for championName, champion in glob_champions.items():
            if champion == None or championName == "":
                continue
            outputString += ",{}".format(championName)
    outputString += "\n"

    # For each champion,
    #    For each other champion,
    #        win ratio = games won / games played
    for championName, champion in glob_champions.items():
        if champion == None or championName == "":
            continue
        
        outputString += championName
        for vsChampionName, vsChampion in glob_champions.items():
            if vsChampion == None or vsChampionName == "":
                continue
            
            championWinRatio = ((champion.vsChampionWon[vsChampionName]/champion.vsChampionPlayed[vsChampionName]) if (champion.vsChampionPlayed[vsChampionName] > 0) else 0)

            outputString += ",{}".format(championWinRatio)
        outputString += "\n"

    file = open("feature_data/champion_vs_data.csv", 'w+')
    file.write(outputString)
    file.close()

def WriteMatchVSAndCoopData():
    csvFile = open('feature_data/match_info.csv', newline='', encoding='utf-8')
    reader = csv.DictReader(csvFile)
    
    outputString = "index,"
    for name in glob_rolesAll:
        outputString += "{},{},{},".format(name, name+"Vs", name+"Coop")
    for name in glob_champsAll:
        outputString += "{},{},{},".format(name, name+"Vs", name+"Coop")
    outputString += "\n"

    for row in reader:
        # For now, skip rows that don't have complete information.
        if not IsRowValid(row):
            continue

        outputString += "{},".format(row['index'])

        # Blue Players
        for rollName in glob_rolesBlue:
            playerName = row[rollName]
            player = glob_players[playerName]
            statVs = 0
            statCoop = 0

            for vsRollName in glob_rolesRed:
                vsPlayerName = row[vsRollName]

                statVs += ((player.vsPlayerWon[vsPlayerName]/player.vsPlayerPlayed[vsPlayerName]) if (player.vsPlayerPlayed[vsPlayerName] > 0) else 0)

            outputString += "{},{},{},".format(playerName, statVs, statCoop)
        
        # Red Players
        for rollName in glob_rolesRed:
            playerName = row[rollName]
            player = glob_players[playerName]
            statVs = 0
            statCoop = 0

            for vsRollName in glob_rolesBlue:
                vsPlayerName = row[vsRollName]

                statVs += ((player.vsPlayerWon[vsPlayerName]/player.vsPlayerPlayed[vsPlayerName]) if (player.vsPlayerPlayed[vsPlayerName] > 0) else 0)

            outputString += "{},{},{},".format(playerName, statVs, statCoop)

        # Blue Champions
        for rollName in glob_champsBlue:
            championName = row[rollName]
            champion = glob_champions[championName]
            statVs = 0
            statCoop = 0

            for vsRollName in glob_champsRed:
                vsChampionName = row[vsRollName]

                statVs += ((champion.vsChampionWon[vsChampionName]/champion.vsChampionPlayed[vsChampionName]) if (champion.vsChampionPlayed[vsChampionName] > 0) else 0)

            outputString += "{},{},{},".format(championName, statVs, statCoop)
        
        # Red Players
        for rollName in glob_champsRed:
            championName = row[rollName]
            champion = glob_champions[championName]
            statVs = 0
            statCoop = 0

            for vsRollName in glob_champsBlue:
                vsChampionName = row[vsRollName]

                statVs += ((champion.vsChampionWon[vsChampionName]/champion.vsChampionPlayed[vsChampionName]) if (champion.vsChampionPlayed[vsChampionName] > 0) else 0)

            outputString += "{},{},{},".format(championName, statVs, statCoop)
            
        outputString += "\n"

    file = open("feature_data/match_vs_coop_data.csv", 'w+')
    file.write(outputString)
    file.close()




            

#WritePlayerData()
#WriteChampionData()
#WritePlayerWinsWithEachChampionData()
#WriteTeamData()
#WritePlayerVsData()
#WriteChampionVsData()
WriteMatchVSAndCoopData()