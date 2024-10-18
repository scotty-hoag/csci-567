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
glob_playerNames = set()
# (set of all champion names)
glob_championNames = set()


glob_rolesBlue =  ['blueTop',        'blueJungle',       'blueMiddle',       'blueADC',      'blueSupport']
glob_rolesRed  =  ['redTop',         'redJungle',        'redMiddle',        'redADC',       'redSupport' ]
glob_champsBlue = ['blueTopChamp',   'blueJungleChamp',  'blueMiddleChamp',  'blueADCChamp', 'blueSupportChamp']
glob_champsRed  = ['redTopChamp',    'redJungleChamp',   'redMiddleChamp',   'redADCChamp',  'redSupportChamp']

glob_rolesAll  = glob_rolesBlue  + glob_rolesRed
glob_champsAll = glob_champsBlue + glob_champsRed


class Player:
    name = ""

    gamesPlayed = 0
    gamesWon = 0
    
    rolesPlayed = {}
    rolesWon = {}

    championsPlayed = {}
    championsWon = {}

    def __init__(self):
        self.rolesPlayed = {}
        self.rolesWon = {}

        self.championsPlayed = {}
        self.championsWon = {}
glob_players = {}

class Champion:
    name = ""

    gamesPlayed = 0
    gamesWon = 0

    rolesPlayed = {}
    rolesWon = {}

    def __init__(self):
        self.rolesPlayed = {}
        self.rolesWon = {}
glob_champions = {}

def IsRowValid(row):
    # For now, skip rows that don't have complete information.
    bValid = not (row["blueTeamTag"] == "" or row["redTeamTag"] == "")
    return bValid

with open('lol_data/matchinfo.csv', newline='', encoding='utf-8') as csvFile:
    reader = csv.DictReader(csvFile)
    for row in reader:
        # For now, skip rows that don't have complete information.
        if not IsRowValid(row):
            continue

        # Player and champion names are added to SETs, so enforces uniqueness.
        for role in glob_rolesAll:
            glob_playerNames.add(row[role])

        for champion in glob_champsAll:
            glob_championNames.add(row[champion])

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

        # Record the number of champion plays and wins.
        for champ in glob_championNames:
            player.championsPlayed[champ] = 0
            player.championsWon[champ] = 0
    
    # Record the champion role win stats.
    for championName in glob_championNames:
        champion = Champion()
        champion.name = championName
        glob_champions[championName] = champion

        # Init the number of plays/wins 
        for role in glob_rolesAll:
            champion.rolesPlayed[role] = 0
            champion.rolesWon[role] = 0

with open('lol_data/matchinfo.csv', newline='', encoding='utf-8') as csvFile:
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
        
        if winBlue == 1:
            winningTeamRoles = glob_rolesBlue
            losingTeamRoles = glob_rolesRed

            winningTeamChamps = glob_champsBlue
            losingTeamChamps = glob_champsRed
        else:
            winningTeamRoles = glob_rolesRed
            losingTeamRoles = glob_rolesBlue

            winningTeamChamps = glob_champsRed
            losingTeamChamps = glob_champsBlue
        
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

WritePlayerData()
WriteChampionData()