import pandas as pd
import numpy as np
import json
import random
import time
from datetime import datetime as dt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from scipy.stats import bootstrap
import matplotlib.pyplot as plt

FILE = r"C:\Users\gabeg\Documents\Uni Work\Strategic Leadership\merged_gw.csv"
WAIT_TIME = 1 #seconds
MY_ABBRV_TO_DATA = {"CHE":"Chelsea",
                    "EVE":"Everton",
                    "LIV":"Liverpool",
                    "WHU":"West Ham",
                    "AVL":"Aston Villa",
                    "FUL":"Fulham",
                    "BHA":"Brighton",
                    "WOV":"Wolves",
                    "MNC":"Man City",
                    "LCI":"Leicester",
                    "SOU":"Southampton",
                    "MNU":"Man Utd",
                    "TOT":"Spurs",
                    "BOU":"Bournemouth",
                    "BRE":"Brentford",
                    "ARS":"Arsenal",
                    "CRY":"Crystal Palace",
                    "NEW":"Newcastle",
                    "IPS":"Ipswich",
                    "NOT":"Nott'm Forest"}
DATA_TO_MY_ABBRV = {value:key for key, value in MY_ABBRV_TO_DATA.items()}
OPPONENT_TEAM_TO_NAME = {i+1:name for i, name in enumerate(sorted(list(DATA_TO_MY_ABBRV)))} 
GAMES_LOCATION = r"C:\Users\gabeg\Documents\Accurate xG\accuratexG\local"
TEAM_IDS_FILE = r"C:\Users\gabeg\Documents\Uni Work\Strategic Leadership\FPL Code\teamIds.csv"
TEAM_IDS = {}
with open(TEAM_IDS_FILE,"r") as f:
    for line in f.readlines():
        teamID, name = line.replace("\n","").split(",")
        TEAM_IDS[name] = teamID
ELO_FILE_LOCATIONS = r"C:\Users\gabeg\Documents\Accurate xG\OptaScrape\exampleEnv\ResultXG\teams\{teamID}\elo.json"
ML_MAX_GOALS_SCORED = 3
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
with open(r"C:\Users\gabeg\Documents\Uni Work\Strategic Leadership\FPL Code\fplNamesConversionReduced.json", "r") as f:
    obj = json.load(f)
    FPL_TO_OPTA_PLAYERS = {value[0]:key for key,value in obj.items()}
    OPTA_PLAYERS_TO_FPL = {value:key for key,value in FPL_TO_OPTA_PLAYERS.items()}

def fixFile():
    newData = []
    with open(FILE, "r", encoding="UTF-8") as f:
        for lineNum, line in enumerate(f.readlines()):
                if lineNum >= 14179: #after GW 21 (excluding GW 21), remove the manager points as they aren't used moving forward, and cause an error when loading the file due to an inconsistent number of columns 
                        lineData = line.split(", ")
                        lineData = lineData[:21] + lineData[28:]
                        newData.append(','.join(lineData))
                else:
                        newData.append(line)
    with open(FILE, "w", encoding="UTF-8") as f:
        f.writelines(newData)

def PoiBin(data):
    numberTrials = len(data)
    omega = 2 * np.pi / (numberTrials + 1)
    successProbabilities = np.array(data)
    chi = np.empty(numberTrials + 1, dtype=complex)
    chi[0] = 1
    halfNumberTrials = int(
        numberTrials / 2 + numberTrials % 2)
    # set first half of chis:
    idxArray = np.arange(1, halfNumberTrials + 1)
    expValue = np.exp(omega * idxArray * 1j)
    xy = 1 - successProbabilities + successProbabilities * expValue[:, np.newaxis]
    # sum over the principal values of the arguments of z:
    argz_sum = np.arctan2(xy.imag, xy.real).sum(axis=1)
    # get d value:
    exparg = np.log(np.abs(xy)).sum(axis=1)
    d_value = np.exp(exparg)
    # get chi values:
    chiCalc = d_value * np.exp(argz_sum * 1j)
    chi[1:halfNumberTrials + 1] = chiCalc
    # set second half of chis:
    chi[halfNumberTrials + 1:numberTrials + 1] = np.conjugate(
        chi[1:numberTrials - halfNumberTrials + 1] [::-1])
    chi /= numberTrials + 1
    xi = np.fft.fft(chi)
    xi += np.finfo(type(xi[0])).eps
    xi = {i:float(item.real) for i,item in enumerate(xi)}
    return xi

def readRxgFile(fileLocation):
    with open(fileLocation,"r",encoding="UTF-8") as targetFile:
        data = []
        for lineContents in targetFile.readlines():
            lineContents = lineContents.replace("\n", "")
            data.append(lineContents)
        return data
        
def calculateStatsFromLines(lineContents):
    teamXg = {0:[],1:[]}
    playerData = {}
    for line in lineContents:
        lineData = line.split(",")
        team = int(lineData[0])
        teamXg[team].append(float(lineData[1]))
        playerInfo = playerData.get(lineData[5], [])
        playerInfo.append(float(lineData[1]))
        playerData[lineData[5]] = playerInfo
    homeTeamXg = PoiBin(teamXg[0])
    awayTeamXg = PoiBin(teamXg[1])
    for player, data in playerData.items():
        dist = PoiBin(data)
        mlDist = {}
        for goals, p in dist.items():
            if goals <= ML_MAX_GOALS_SCORED:
                mlDist[goals] = p
            else:
                mlDist[ML_MAX_GOALS_SCORED] += p 
        for i in range(ML_MAX_GOALS_SCORED,0, -1):
            if mlDist.get(i) is None:
                mlDist[i] = 0
            else:
                break 
        playerData[player] = mlDist
    return {"homeXG":homeTeamXg,"awayXG":awayTeamXg, "playerData":playerData}

#fixFile()
def sanitiseDf():
    df = pd.read_csv(FILE)
    df = df[["name", "position", "team", "clean_sheets", "goals_conceded", "goals_scored", "minutes", "opponent_team", "own_goals", "penalties_missed", "penalties_saved", "red_cards", "team_a_score", "team_h_score", "total_points", "value", "was_home", "yellow_cards", "GW"]]
    df["matchFile"] = df.apply(lambda row: DATA_TO_MY_ABBRV[row["team"]] + DATA_TO_MY_ABBRV[OPPONENT_TEAM_TO_NAME[row["opponent_team"]]] if row["was_home"] else DATA_TO_MY_ABBRV[OPPONENT_TEAM_TO_NAME[row["opponent_team"]]] + DATA_TO_MY_ABBRV[row["team"]], axis=1)
    df["homeTeamID"] = df.apply(lambda row: TEAM_IDS[row["team"]],axis=1)
    df["awayTeamID"] = df.apply(lambda row: row["opponent_team"]-1,axis=1)
    gameStats = {}
    playerStats = {}

    for game in set(df["matchFile"]):
        stats = calculateStatsFromLines(readRxgFile(f"{GAMES_LOCATION}/{game}2425.txt"))
        gameStats[game] = stats
        playerData = stats["playerData"]
        for player, distribution in playerData.items():
            playerGames = playerStats.get(player,{})
            if not playerGames:
                playerStats[player] = playerGames
            playerGames[game] = distribution
            played = playerGames.get("gamesPlayed", [])
            played.append(game)
            playerGames["gamesPlayed"] = played

    
    r'''
    fplNames = set(df["name"])
    linkedNames = {}
    for optaName in playerStats:
        lastName = optaName.split(" ")[-1]
        potentialNames = list(filter(lambda fplName: lastName.lower() in fplName.lower(), fplNames))
        if not potentialNames:
            print(lastName, optaName)    
        linkedNames[optaName] = potentialNames

    print(linkedNames)

    with open(r"C:\Users\gabeg\Documents\Accurate xG\accuratexG\local\fplNamesConversionReduced.json", "r", encoding="UTF-8") as f:
        reducedLinkedNames = json.load(f)

    for reducedName, value in reducedLinkedNames.items():
        if len(value) != len(linkedNames[reducedName]):
            if len(playerStats[reducedName]["gamesPlayed"]) < 2:
                playerTeam = playerStats[reducedName]["gamesPlayed"]
            else:
                match1, match2 = playerStats[reducedName]["gamesPlayed"][0:2]
                if match1[:3] == match2[:3] or match1[:3] == match2[3:]: #specifically check the 1st 3 and last 3 to avoid acronyms appearing in the middle of the 6 letter match acronymn e.g. match1[:3] = ARS and match2 = xxARSx
                    playerTeam = match1[:3]
                else:
                    playerTeam = match1[3:]
            print(f"{playerTeam} Selected {reducedName}: {value} from {linkedNames[reducedName]}")'''
    print(len(playerStats))
    for i, row in df.iterrows():
        player = FPL_TO_OPTA_PLAYERS.get(row["name"],row["name"])
        game = row["matchFile"]
        playerGames = playerStats.get(player,{})
        if not playerGames.get(game):
            playerGames[game] = {i:0 if i != 0 else 1 for i in range(ML_MAX_GOALS_SCORED+1)} 
        playerStats[player] = playerGames
    print(len(playerStats))
    print(playerStats.keys())
    df["goals_scored_distribution"] = df.apply(lambda row: json.dumps(playerStats[FPL_TO_OPTA_PLAYERS.get(row["name"],row["name"])][row["matchFile"]]), axis=1)
    df.to_csv("sanitised_gws.csv",index=False)

def trainModelPG(X,y, graph=False):
    X = np.array(X)
    y = np.array(y)

    xTrain, xTest, yTrain, yTest = train_test_split(X,y,test_size=0.2,random_state=RANDOM_SEED)
    
    indexes = getTestSequence(X, xTest)

    model_PG = keras.Sequential([
        layers.Dense(400, activation='relu', input_shape=(X.shape[1],)),
        layers.Dense(400, activation='relu'),
        layers.Dense(400, activation='relu'),
        layers.Dense(400, activation='relu'),
        layers.Dense(400, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Ensure output is a probability
    ])

    model_PG.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Train the model
    model_PG.fit(xTrain, yTrain, epochs=10, batch_size=32)

    # Evaluate the model on the test set
    loss, accuracy = model_PG.evaluate(xTest, yTest)

    predictions = model_PG.predict(xTest)
    print(f"Out of {len(xTest)} pieces of data, the average predicted probability is {sum(predictions)/len(predictions)}. The max value is {max(predictions)} and the min value is {min(predictions)}")
    print(f"Model Player Goals was trained using {len(xTrain)} pieces of data, and tested on {len(xTest)} pieces of data. The results of the test were a test loss (MSE) of {loss} (RMSE: {loss**0.5}), and a test MAE of {accuracy}")
    '''if input("Would you like to save model?\n").lower() == "yes":
        model_PG.save_weights(f"fpl_wdl_model_pg.weights.h5")'''
    
    model_PG.load_weights(f"fpl_wdl_model_pg.weights.h5")

    MARGIN_OF_ERROR = 0.1
    predictions = model_PG.predict(xTest)
    if graph:
        count = 0
        boxWhiskersInput = []
        for i, prediction in enumerate(predictions):
            if abs(prediction[0] - yTest[i]) <= MARGIN_OF_ERROR:
                count += 1
            boxWhiskersInput.append(prediction[0] - yTest[i])
        print(count/len(predictions))

        plt.boxplot(boxWhiskersInput, vert=False)
        plt.title("Boxplot of 'Model Predicted Player Scores - Actual Player Scores Probability'")
        plt.xlabel("Probability difference between model prediction and actual player scores probabilities")
        plt.vlines([0],0.925,1.074, label="Target Value", colors=["r"])
        plt.legend()
        plt.show()
    
    return model_PG, indexes

def getTestSequence(unsplitArray: np.ndarray, testArray: np.ndarray):
    unsplitArray = unsplitArray.tolist()
    testArray = testArray.tolist()
    indexes = []
    for item in testArray:
        indexes.append(unsplitArray.index(item))
    return indexes

def trainModelCS(X,y):

    X = np.array(X)
    y = np.array(y)


    xTrain, xTest, yTrain, yTest = train_test_split(X,y,test_size=0.2,random_state=RANDOM_SEED)

    indexes = getTestSequence(X, xTest)

    model_CS = keras.Sequential([
        layers.Dense(400, activation='relu', input_shape=(X.shape[1],)),
        layers.Dense(400, activation='relu'),
        layers.Dense(400, activation='relu'),
        layers.Dense(400, activation='relu'),
        layers.Dense(400, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Ensure output is a probability
    ])


    model_CS.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Train the model
    model_CS.fit(xTrain, yTrain, epochs=10, batch_size=32)

    # Evaluate the model on the test set
    loss, accuracy = model_CS.evaluate(xTest, yTest)

    predictions = model_CS.predict(xTest)
    print(f"Model Clean Sheets was trained using {len(xTrain)} pieces of data, and tested on {len(xTest)} pieces of data. The results of the test were a test loss (MSE) of {loss} (RMSE: {loss**0.5}), and a test MAE of {accuracy}")
    '''if input("Would you like to save model?\n").lower() == "yes":
    model.save_weights(f"fpl_wdl_model_cs.weights.h5")'''
    MARGIN_OF_ERROR = 0.1
    model_CS.load_weights(f"fpl_wdl_model_cs.weights.h5")
    return model_CS, indexes
    predictions = model_CS.predict(xTest)
    count = 0
    boxWhiskersInput = []
    for i, prediction in enumerate(predictions):
        if abs(prediction[0] - yTest[i]) <= MARGIN_OF_ERROR:
            count += 1
        boxWhiskersInput.append(prediction[0] - yTest[i])
    print(count/len(predictions))

    plt.boxplot(boxWhiskersInput, vert=False)
    plt.title("Boxplot of 'Model Predicted Clean Sheet - Actual Clean Sheet Probability'")
    plt.xlabel("Probability difference between model prediction and actual clean sheet probabilities")
    plt.vlines([0],0.925,1.074, label="Target Value", colors=["r"])
    plt.legend()
    plt.show()

def buildInputsPG(playerNames):
    ROLLING_SIZE_PG = 11
    MAX_MINS_PER_GAME = 90

    X_PG = []
    y_PG = []

    #predicting goals scored
    #position (isGk, isDef, isMid, isFwd)
    #team elos diff
    #history of expected goals scored (0,1,2,3+) + team elos
    #percentage mins played
    #isHome

    MAX_BENCH_PLAYER_TRAINS = 0
    benchPlayers = 0
    count = 0
    globalIndexLookup = {}
    for player in playerNames:
        dfPlayer = df[df["name"] == player].sort_values(by=["GW"])
        dfPlayer["isHome"] = dfPlayer.apply(lambda row: DATA_TO_MY_ABBRV[row["team"]] == row["matchFile"][:3], axis=1)
        dfPlayer["minutes"] = dfPlayer.apply(lambda row: row["minutes"]/MAX_MINS_PER_GAME, axis=1)
        dfPlayer["eloHomeData"] = dfPlayer.apply(lambda row: getEloData(TEAM_IDS[MY_ABBRV_TO_DATA[row["matchFile"][:3]]]), axis=1)
        dfPlayer["eloAwayData"] = dfPlayer.apply(lambda row: getEloData(TEAM_IDS[MY_ABBRV_TO_DATA[row["matchFile"][3:]]]), axis=1)
        
        
        playerXInputs = []
        playerYInputs = []
        for i in range(ROLLING_SIZE_PG,len(dfPlayer)+1):
            currentSlice = dfPlayer[i-ROLLING_SIZE_PG:i].reset_index()

            obj = {"X":[dfPlayer["position"].iloc[0] == "GK", dfPlayer["position"].iloc[0] == "DEF", dfPlayer["position"].iloc[0] == "MID", dfPlayer["position"].iloc[0] == "FWD"],"y":None}

            for j, row in currentSlice.iterrows():
                obj["X"] += [row["isHome"], (row["eloAwayData"][row["GW"]-1] - row["eloHomeData"][row["GW"]-1])*-1**(row["isHome"])]
                if j + 1 < len(currentSlice):
                    obj["X"] += [row["minutes"], 1 - row["goals_scored_distribution"]["0"]]
                else:
                    obj["y"] = 1 - row["goals_scored_distribution"]["0"]
            
            benchPlayers += obj["y"] == 0
            if benchPlayers > MAX_BENCH_PLAYER_TRAINS and obj["y"] == 0:
                continue
            playerXInputs.append(obj["X"])
            playerYInputs.append(obj["y"])
            globalIndexLookup[count] = {"name":player,"match":row["matchFile"]}
            count += 1

        
        X_PG += playerXInputs
        y_PG += playerYInputs
    return X_PG, y_PG, globalIndexLookup

def buildInputsCS(gameStats):
    ROLLING_SIZE_CS = 11

    X_CS = []
    y_CS = []
    count = 0
    globalIndexLookup = {}
    for team in MY_ABBRV_TO_DATA:
        teamStats = list(filter(lambda obj: obj["home"] == team or obj["away"] == team,gameStats))
        teamID = TEAM_IDS[MY_ABBRV_TO_DATA[team]]
        eloData = getEloData(teamID)
        teamXInputs = []
        teamYInputs = []
        for rollValue in range(ROLLING_SIZE_CS,max(df["GW"])+1):
            games = teamStats[rollValue-ROLLING_SIZE_CS:rollValue]
            obj = {"X":[],"y":None}
            for i,game in enumerate(games):
                if game["home"] == team:
                    isHome = True
                    otherTeamID = TEAM_IDS[MY_ABBRV_TO_DATA[game["away"]]]
                else:
                    isHome = False
                    otherTeamID = TEAM_IDS[MY_ABBRV_TO_DATA[game["home"]]]
                otherTeamElo = getEloData(otherTeamID)
                data = [eloData[rollValue-ROLLING_SIZE_CS+i] - otherTeamElo[rollValue-ROLLING_SIZE_CS+i], isHome]
                cleanSheet = game["home_cleanSheet"] if isHome else game["away_cleanSheet"]
                if i+1 < len(games):
                    data.append(cleanSheet)
                else:
                    obj["y"] = cleanSheet
                obj["X"] += data
            teamXInputs.append(obj["X"])
            teamYInputs.append(obj["y"])
            globalIndexLookup[count] = f"{game["home"]}{game["away"]}_{team}"
            count += 1
        X_CS += teamXInputs
        y_CS += teamYInputs
    
    return X_CS, y_CS, globalIndexLookup

def getEloData(teamID):
    eloFileLocation = ELO_FILE_LOCATIONS.format(teamID=teamID)
    with open(eloFileLocation,"r") as f:
        obj = json.load(f)
        eloData = [eloObj["teamElo"] for eloObj in obj["matches"]]
    return eloData

def simulateSeason(df):
    matches = {}

    df = df[df["goals_scored_distribution"].str["0"] != 1]

    for i, row in df.iterrows():
        playerDistribution = row["goals_scored_distribution"]
        goalsScored = int(random.choices(list(playerDistribution.keys()), weights=list(playerDistribution.values()))[0])
        matchArray = matches.get(row["matchFile"], [])
        if goalsScored != 0:
            matchArray.append({"name":row["name"], "goalsScored":goalsScored, "team": DATA_TO_MY_ABBRV[row["team"]]})
        matches[row["matchFile"]] = matchArray
    return matches

def modelInterpret(modelType,value):
    CLEAN_SHEET_MIN_PREDICT = 0.45
    PLAYER_GOALS_MIN_PREDICT = 0.55
    prediction = None
    modelType = modelType.lower()
    if modelType == "cs":
        prediction = value >= CLEAN_SHEET_MIN_PREDICT # 1 or 0
    elif modelType == "pg":
        prediction = value >= PLAYER_GOALS_MIN_PREDICT
    else:
        raise Exception(f"Unknown model type '{modelType}'")
    return prediction

def monteCarlo(df, modelInfos: dict, numIterations=10000, iterOutputNumber=0,loadFile=False):
    
    csInfo: dict = modelInfos["cs"] #csInfo = {"model":modelObj,"indexes":[testDataIndexesInX...], "indexLookup":{testDataIndex:matchFixtureName}, "X":[trainedXData...]}
    pgInfo: dict = modelInfos["pg"]

    CLEAN_SHEET_JSON_NAME = "monteCarlo_CS.json"
    PLAYER_GOALS_JSON_NAME = "monteCarlo_PG.json"
    
    CS_MODEL_PREDICT_BLIND = 0.2647
    PG_MODEL_PREDICT_BLIND = 0.1847

    MODEL_PREDICT_UNIFORM = 0.5

    if loadFile:
        with open(CLEAN_SHEET_JSON_NAME, "r") as f:
            cs = json.load(f)
        with open(PLAYER_GOALS_JSON_NAME, "r") as f:
            pg = json.load(f)
    else:
        startTime = time.time()
        cs = {"cs":[],"csB":[], "csU":[]}
        pg = {"pg":[], "pgB":[],"pgU":[]}
        
        yCs = csInfo["model"].predict(np.array(csInfo["X"]))
        yPg = pgInfo["model"].predict(np.array(pgInfo["X"]))
        for iter in range(numIterations):
            cleanSheets = {}
            playerGoals = {}
            matches = simulateSeason(df)
            for fixture, matchArray in matches.items():
                home = fixture[:3]
                away = fixture[3:]

                data = {}

                data[home] = len(list(filter(lambda obj: obj["team"] == home,matchArray))) == 0
                data[away] = len(list(filter(lambda obj: obj["team"] == away,matchArray))) == 0

                cleanSheets[fixture] = data

                for entry in matchArray:
                    data = playerGoals.get(entry["name"], {})
                    data[fixture] = entry["goalsScored"]
                    playerGoals[entry["name"]] = data

            csAvg = 0
            csBAvg = 0
            csUAvg = 0
            
            for index in csInfo["indexes"]:
                y = yCs[index][0]
                predictedCleanSheet = modelInterpret("cs",y)


                predictedCleanSheetBlind = random.random() >= 1- CS_MODEL_PREDICT_BLIND
                predictedCleanSheetUniform = random.random() >= 1 - MODEL_PREDICT_UNIFORM

                matchDetails = csInfo["indexLookup"][index]
                matchFixture, teamPlayed = matchDetails.split("_")           
                simulatedCleanSheet = cleanSheets[matchFixture][teamPlayed]
                csAvg += predictedCleanSheet == simulatedCleanSheet
                csBAvg += predictedCleanSheetBlind == simulatedCleanSheet
                csUAvg += predictedCleanSheetUniform == simulatedCleanSheet
            
            csAvg /= len(csInfo["indexes"])
            csBAvg /= len(csInfo["indexes"])
            csUAvg /= len(csInfo["indexes"])
            cs["cs"].append(csAvg)
            cs["csB"].append(csBAvg)
            cs["csU"].append(csUAvg)
            
            pgAvg = 0
            pgUAvg = 0
            pgBAvg = 0
            for index in pgInfo["indexes"]:
                y = yPg[index][0]
                predictedGoalScored = modelInterpret("pg",y) 
                predictedGoalScoredBlind = random.random() >= 1- PG_MODEL_PREDICT_BLIND
                predictedGoalScoredUniform = random.random() >= 1- MODEL_PREDICT_UNIFORM

                playerDetails = pgInfo["indexLookup"][index]
                simulatedGoalScored = playerGoals.get(playerDetails["name"],{}).get(playerDetails["match"],0)
                
                pgAvg += predictedGoalScored == simulatedGoalScored
                pgUAvg += predictedGoalScoredUniform == simulatedGoalScored
                pgBAvg += predictedGoalScoredBlind == simulatedGoalScored
            
            pgAvg /= len(pgInfo["indexes"])
            pgUAvg /= len(pgInfo["indexes"])
            pgBAvg /= len(pgInfo["indexes"])

            pg["pg"].append(pgAvg)
            pg["pgU"].append(pgUAvg)
            pg["pgB"].append(pgBAvg)

            if iterOutputNumber != 0 and (iter+1) % iterOutputNumber == 0:
                currentTime = time.time()
                secondsTaken = currentTime - startTime
                fractionComplete = (iter+1)/numIterations
                predictedSeconds = secondsTaken/fractionComplete
                print(f"We have completed {iter+1} iterations in {secondsTaken} seconds ({fractionComplete*100}%). Predicted completed in {predictedSeconds-secondsTaken} seconds ({(predictedSeconds-secondsTaken)/60} minutes)")
        
        playerGoals = {playerName:{matchFixture: value/numIterations for matchFixture, value in dataObj.items()} for playerName, dataObj in playerGoals.items()}

        with open(CLEAN_SHEET_JSON_NAME,"w") as f:
            json.dump(cs,f)
        with open(PLAYER_GOALS_JSON_NAME,"w") as f:
            json.dump(pg,f)

    return cs, pg

def percentages(percs, graph=False):
    
    if graph:
        plt.title("Histogram of the percentage chance of an event occuring")
        plt.hist(percs,10,density=True)
        plt.show()
    avgVal = sum(percs)/len(percs)
    print(avgVal)
    res = bootstrap((np.array(percs),),lambda data: sum(data)/len(data), confidence_level=0.99,random_state=RANDOM_SEED)
    print(res.confidence_interval.low, res.confidence_interval.high)
    return res.confidence_interval

#sanitiseDf()

df = pd.read_csv("sanitised_gws.csv")
df["goals_scored_distribution"] = df.apply(lambda row: json.loads(row["goals_scored_distribution"]),axis=1)

assistDf2425 = pd.read_excel("Soccer-Stats-Premier-League-2024-2025.xlsx")

SCRAPED_GAMES = r"C:\Users\gabeg\Documents\Accurate xG\OptaScrape\xgDataScraped_pl.txt"
gameStats = []
with open(SCRAPED_GAMES,"r", encoding="UTF-8") as f:
    for line in f.readlines():
        obj = json.loads(line)
        obj["distribution"] = calculateStatsFromLines(obj["xGdata"].split(";"))
        obj["home_cleanSheet"] = obj["distribution"]["awayXG"][0]
        obj["away_cleanSheet"] = obj["distribution"]["homeXG"][0]
        gameStats.append(obj)


#for each input
#10 prior game data - elo diff, home or not, probability of keeping a clean sheet
#this games elo diff, home or not, y value is probability of keeping a clean sheet

            



gameStats = sorted(gameStats,key=lambda obj: dt.strptime(obj["date"],"%b %d, %Y").timestamp())



csPercs = sum([[matchObj["distribution"]["homeXG"][0],matchObj["distribution"]["awayXG"][0]] for matchObj in gameStats],[])
pgPercs = sum([[1 - playerObj[0] for playerObj in matchObj["distribution"]["playerData"].values()] for matchObj in gameStats], [])

csIntervals = percentages(csPercs)
pgIntervals = percentages(pgPercs)

X_CS, y_CS, indexLookup_CS = buildInputsCS(gameStats)

model_CS, indexes_CS = trainModelCS(X_CS,y_CS)


playerNames = set(df["name"])
X_PG, y_PG, indexLookup_PG = buildInputsPG(playerNames)

model_PG, indexes_PG = trainModelPG(X_PG,y_PG)
NUM_CARLO_ITERS = 10000
cs, pg = monteCarlo(df,{"cs":{"model":model_CS,"indexes":indexes_CS,"indexLookup":indexLookup_CS,"X":X_CS},"pg":{"model":model_PG,"indexes":indexes_PG,"indexLookup":indexLookup_PG,"X":X_PG}},NUM_CARLO_ITERS,max(int(NUM_CARLO_ITERS/100),1), True)

print(f"The clean sheet model was simulated and the percentage of the clean sheets that were, on average, accurately predicted were as follows:\nmodel - {sum(cs["cs"])/len(cs["cs"])*100}%\nBlind guess - {sum(cs["csB"])/len(cs["csB"])*100}% assuming the average value of the data lies between {csIntervals.low*100}% and {csIntervals.high*100}%\nUniform - {sum(cs["csU"])/len(cs["csU"])*100}% assuming the data follows a uniform distribution")

print(f"The player goals model was simulated and the percentage of the players that were, on average, accurately predicted to have scored at least 1 goal or not were as follows:\nmodel - {sum(pg["pg"])/len(pg["pg"])*100}%\nBlind guess - {sum(pg["pgB"])/len(pg["pgB"])*100}% assuming the average value of the data lies between {pgIntervals.low*100}% and {pgIntervals.high*100}%\nUniform - {sum(pg["pgU"])/len(pg["pgU"])*100}% assuming the data follows a uniform distribution")

#check how well the pg model deals with people who I have currently filtered out. 
#determine how well the pg model does at predicting clean sheets vs the cs model
#try and perform some analysis on how people from different teams get assists. Currently I have in my head that I argue this as 'different tactics will cause different people to get assists'. Therefore, we can algorithmically determine the top candidates per team for gathering assists given a particular player is going to score (perhaps using some sort of data mining technique?)

'''testNames = list(set(df["name"]))

mlData = {}

GW_ROLLING_AVERAGE = 10
maxGameWeeks = max(df["GW"])
for name in testNames:
    gwData = []
    for gwOffset in range(maxGameWeeks-GW_ROLLING_AVERAGE+1):
        filteredDf = df[(df["GW"] > gwOffset) & (df["GW"] <= gwOffset + GW_ROLLING_AVERAGE) & (df["name"] == name)]
        gwData.append(buildMlGoalsArray(filteredDf))
    mlData[name] = gwData

TEST_PERCENTAGE = 0.2
numberTrain = (1-TEST_PERCENTAGE)*len(testNames) // 1 + 1
trainNames = []
for i in range(numberTrain):
    name = random.choice(testNames)
    trainNames.append(name)
    testNames.remove(name)'''

#want to predict
#goals scored
#assists
#goals conceeded (by team)

#predicting goals scored
#position (isGk, isDef, isMid, isFwd)
#team elos diff
#history of expected goals scored (0,1,2,3+) + team elos
#percentage mins played
#isHome


#predicting goals conceeded
#team elos diff
#expected goals conceeded (0,1,2,3,4+) history

#Want to evaluate the accuracy of the model regardless of the accuracy of the xgData
#So, I want to say, how accurate was the model compared with the actual points from fpl, but there's a chance that the xgData is rubbish and doesn't accurately represent P(Scoring Goal)
#Therefore, I want to use my probability distribution for each game to determine my 'Monte Carlo' fantasy premier league to determine how accurate the model was given the data it had (I'm thinking simulate each game once per season, and then simulate thousands of seasons and get the average points accuracy across all seasons)
#Note, I am NOT trying to monte carlo new probability outcomes for matches (e.g clean sheets, goals scored, etc). I am instead trying to avoid saying 'the model predicted a clean sheet bonus'
#but there was none when the probability distribution claims a 95% chance of a clean sheet and so we would want to model to predict a clean sheet

