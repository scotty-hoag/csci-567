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

    rolesPlayed = {}
    rolesWon = {}
glob_champions = {}

def IsRowValid(row):
    # For now, skip rows that don't have complete information.
    bValid = not (row["blueTeamTag"] == "" or row["redTeamTag"] == "")
    return bValid

with open('lol_data/matchinfo.csv', newline='', encoding='utf-8') as csvFile:
    #csvReader = csv.reader(csvFile, delimiter=',', quotechar='|')
    #for row in reader:
    #    print(', '.join(row))
    
    #for role in glob_rolesRed:
    #    numMatchesInRole[role] = {}
    #    numMatchesInRoleWon[role] = {}
    #for role in rolesBlue:
    #    numMatchesInRole[role] = {}
    #    numMatchesInRoleWon[role] = {}

    reader = csv.DictReader(csvFile)
    for row in reader:
        # For now, skip rows that don't have complete information.
        if not IsRowValid(row):
            continue

        for role in glob_rolesAll:
            glob_playerNames.add(row[role])

        for champion in glob_champsAll:
            glob_championNames.add(row[champion])

    for role in glob_rolesAll:
        glob_numMatchesInRole[role] = {}
        glob_numMatchesInRoleWon[role] = {}

    for playerName in glob_playerNames:
        player = Player()
        player.name = playerName
        glob_players[playerName] = player

        # Init the number of plays/wins 
        for role in glob_rolesAll:
            player.rolesPlayed[role] = 0
            player.rolesWon[role] = 0

        # Record the number of champion plays and wins.
        for champ in glob_champsAll:
            player.championsPlayed[champ] = 0
            player.championsWon[champ] = 0

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
            champion = losingTeamChamps[roleIndex]
            playerName = row[role]
            player = glob_players[playerName]

            # Inc matches played by the player.
            player.gamesPlayed += 1

            # Inc matches played by the player in the specific role.
            player.rolesPlayed[role] += 1

            # Inc matches played by the player with the specific Champion.
            player.championsPlayed[champion] += 1

        for roleIndex in range(len(winningTeamRoles)):
            role = winningTeamRoles[roleIndex]
            champion = winningTeamChamps[roleIndex]
            playerName = row[role]
            player = glob_players[playerName]
            
            # Inc matches played and won by the player.
            player.gamesPlayed += 1
            player.gamesWon += 1

            # Inc matches played and won by the player in the specific role.
            player.rolesPlayed[role] += 1
            player.rolesWon[role] += 1

            # Inc matches played and won by the player with the specific Champion.
            player.championsPlayed[champion] += 1
            player.championsWon[champion] += 1

def WritePlayerData():
    outputString = "Player Name," + \
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
    pass

WritePlayerData()
WriteChampionData()