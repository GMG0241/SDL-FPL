import json
import numpy as np
import time
import datetime
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

ALPHA = 1.1

def win(x):
    return ALPHA**(-x)

def lose(x):
    return -(ALPHA**x)

def draw(x):
    return (win(x)+lose(x))/2

#get Poission Binomial Distibution from probability data
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

def calculatedWDL(gameData):
    data = {i:[] for i in range(2)}
    for line in gameData.split(";"):
        if line:
            team,value = line.split(",")[:2]
            data[int(team)].append(float(value))

    homeTeamStats = PoiBin(data[0])
    awayTeamStats = PoiBin(data[1])
    results = {}
    for result0 in homeTeamStats: #yes, I know I could have done this better
        for result1 in awayTeamStats:
            results[str(result0)+"-"+str(result1)] = homeTeamStats[result0]*awayTeamStats[result1]
    homeTeamWinProb, drawProb, homeTeamLoseProb = 0,0,0
    for result in results:
        team0, team1 = result.split("-")
        if int(team0) - int(team1) < 0:
            homeTeamLoseProb += results[result]
        elif int(team0) - int(team1) == 0:
            drawProb += results[result]
        else:
            homeTeamWinProb += results[result]
    return (homeTeamWinProb,drawProb,homeTeamLoseProb)

FILE_DATA = [r"C:\Users\gabeg\Documents\Accurate xG\OptaScrape\xgDataScraped_pl.txt", r"C:\Users\gabeg\Documents\Accurate xG\OptaScrape\xgDataScraped_ch.txt", r"C:\Users\gabeg\Documents\Accurate xG\OptaScrape\xgDataScraped_lo.txt", r"C:\Users\gabeg\Documents\Accurate xG\OptaScrape\xgDataScraped_lt.txt"]#,r"C:\Users\gabeg\Documents\Accurate xG\OptaScrape\xgDataScraped_bl.txt",r"C:\Users\gabeg\Documents\Accurate xG\OptaScrape\xgDataScraped_ll.txt"]
ROLLING_VALUE = 10
OPTA_POWER_ELO = {'MNU': 90.4,'FUL': 90.1,'WHU': 87.8,'AVL': 92,'NOT': 91.2,'BOU': 90.8,'NEW': 93.4,'SOU': 79.7,'EVE': 88.1,'BHA': 91.7,'ARS': 98.5,'WOV': 86.5,'IPS': 82.2,'LIV': 100,'CHE': 92.8,'MNC': 95.4,'BRE': 88.8,'CRY': 91.1,'LCI': 80.9,'TOT': 90.3}
DEFAULT_ELO = {0:0, 1:-6,2:-9,3:-12}
LEAGUE_IDS = {"pl":0,"ch":1,"lo":2,"lt":3}

matchData = []
for file in FILE_DATA:
    with open(file,"r",encoding="UTF-8") as f:
        fileData = [json.loads(line) for line in f.readlines()][::-1]
        dateLinkRel = {}
        for obj in fileData:
            obj["league"] = LEAGUE_IDS[file.split("_")[1][:2]]
            linkList = dateLinkRel.get(obj["date"], [])
            linkList.append(obj["link"])
            dateLinkRel[obj["date"]] = linkList
        for obj in fileData:
            obj["mid"] = datetime.datetime.strftime(datetime.datetime.strptime(obj["date"],"%b %d, %Y"),"%d%m%y") + str(obj["league"]).zfill(2) + str(dateLinkRel[obj["date"]].index(obj["link"])).zfill(4)
            print(obj["mid"])

        matchData.extend(fileData)
print(len(matchData))
teamElos = {}
teamWDL = {}
mlGameData = {}
info = {}
for match in matchData:
    w,d,l = calculatedWDL(match["xGdata"])
    homeTeam = match["home"]
    awayTeam = match["away"]
    
    if not (homeTeam in teamElos):
        teamElos[homeTeam] = [DEFAULT_ELO[match["league"]]]
    if not (awayTeam in teamElos):
        teamElos[awayTeam] = [DEFAULT_ELO[match["league"]]]
    
    homeElo = teamElos[homeTeam][-1]
    awayElo = teamElos[awayTeam][-1]

    homeD =  homeElo - awayElo
    homeDelta = w*win(homeD)+d*draw(homeD)+l*lose(homeD)
    teamElos[homeTeam].append(homeElo+homeDelta)
    teamElos[awayTeam].append(awayElo-homeDelta)
    #teamElos[homeTeam].append(OPTA_POWER_ELO[homeTeam])
    #teamElos[awayTeam].append(OPTA_POWER_ELO[awayTeam])

    homeTeamWDLArray = teamWDL.get(homeTeam,[])
    homeTeamWDLArray.append((w,d,l))
    awayTeamWDLArray = teamWDL.get(awayTeam,[])
    awayTeamWDLArray.append((l,d,w))
    teamWDL[homeTeam] = homeTeamWDLArray
    teamWDL[awayTeam] = awayTeamWDLArray

    #for a match, I want: myTeamElo, otherTeamElo, myTeamW, myTeamD, myTeamL, playedAtHome 
    homeArray = mlGameData.get(homeTeam,[])
    if not homeArray:
        mlGameData[homeTeam] = homeArray
    homeArray.extend([homeElo,awayElo,w,d,l,1])
    awayArray = mlGameData.get(awayTeam,[])
    if not awayArray:
        mlGameData[awayTeam] = awayArray
    awayArray.extend([awayElo,homeElo,l,d,w,0])

    info[homeTeam+awayTeam] = [homeElo,awayElo, w,d,l,datetime.datetime.strptime(match["date"],"%b %d, %Y").timestamp()]

NAMES_TO_SHORT = {"Arsenal":"ARS","Aston Villa":"AVL","Brighton & Hove Albion":"BHA","AFC Bournemouth":"BOU","Brentford":"BRE","Chelsea":"CHE","Crystal Palace":"CRY","Everton":"EVE","Fulham":"FUL","Ipswich Town":"IPS","Leicester City":"LCI","Liverpool":"LIV","Manchester City":"MNC","Manchester Utd":"MNU","Newcastle Utd":"NEW","Nottingham Forest":"NOT","Southampton":"SOU","Tottenham Hotspur":"TOT","West Ham Utd":"WHU","Wolverhampton Wanderers":"WOV","Bochum":"BOC","Frankfurt":"FKF","Heidenheim":"HIH","Holstein":"HOL","Stuttgart":"STU","Leverkusen":"B04","Bremen":"BRM","M'gladbach":"MGB","Augsburg":"AUS","Wolfsburg":"WOF","Mainz 05":"M05","Freiburg":"FBG","Union Berlin":"UBR","Bayern München":"FCB","Leipzig":"RBL","Dortmund":"DOR","St. Pauli":"STP","Hoffenheim":"HOF","Leeds":"LEE","Sheffield Utd":"SHU","Burnley":"BUN","Sunderland":"SUN","Coventry City":"COV","West Bromwich Albion":"WBA","Bristol City":"BRC","Middlesbrough":"MDB","Blackburn Rovers":"BBN","Watford":"WAT","Millwall":"MIL","Sheffield Wednesday":"SHW","Norwich City":"NOR","Preston North End":"PRE","Queens Park Rangers":"QPR","Swansea City":"SWN","Portsmouth":"POR","Oxford Utd":"OXF","Hull City":"HLC","Stoke City ":"STK","Cardiff City":"CAR","Derby County":"DER","Luton Town":"LUT","Plymouth Argyle":"PLY","Birmingham City":"BIR","Wrexham":"WRX","Wycombe Wanderers":"WYC","Charlton Athletic":"CHN","Stockport County":"SKP","Huddersfield Town":"HUD","Bolton Wanderers":"BLT","Reading":"RED","Leyton Orient":"LOR","Blackpool":"BLK","Barnsley":"BAR","Lincoln City":"LIN","Stevenage":"STV","Rotherham Utd":"ROT","Peterborough Utd":"PET","Exeter City":"EXE","Mansfield Town":"MNF","Wigan Athletic":"WIG","Northampton Town":"NHM","Bristol Rovers":"BRR","Burton Albion":"BRA","Crawley Town":"CRW","Cambridge Utd":"CMU","Shrewsbury Town":"SRW","Barcelona":"BFC","Real Madrid":"RMD","Atlético":"ATM","Athletic Club":"ATC","Villarreal":"VIL","Betis":"BET","Mallorca":"MAL","Celta Vigo":"CLV","Vallecano":"VLC","Sevilla":"SEV","Getafe":"GET","Real Sociedad":"RSC","Girona":"GIR","Osasuna":"OSA","Espanyol":"ESP","Valencia":"VAL","Alavés":"ALV","Leganés":"LEG","Las Palmas":"LPM","Valladolid":"VLD","Bournemouth":"BOU","Brighton":"BHA","Man City":"MNC","Newcastle":"NEW","Wolves":"WOV","Nottm Forest":"NOT","Tottenham":"TOT","Leicester":"LCI","Ipswich":"IPS","West Ham":"WHU","Man Utd":"MNU", "Luton":"LUT", "Preston":"PRE","Plymouth":"PLY", "Cardiff":"CAR","Derby":"DER", "QPR":"QPR", "Sheff Wed":"SHW", "Swansea":"SWN", "West Bromwich":"WBA", "Norwich":"NOR", "Hull": "HLC", "Oxford":"OXF", "Stoke":"STK","Coventry":"COV", "Blackburn":"BBN", "Sheff Utd":"SHU","Birmingham":"BIR","Peterborough":"PET","Wigan":"WIG","Rotherham":"ROT","L Orient":"LOR","Shrewsbury":"SRW","Crawley":"CRW","Exeter":"EXE","Stockport":"SKP","Charlton":"CHN","Lincoln":"LIN","Wycombe":"WYC","Northampton":"NHM","Huddersfield":"HUD","Bolton":"BLT","Wimbledon":"WIM","Doncaster":"DON","Harrogate":"HAR","Barrow":"BRW","Cheltenham":"CHL","Chesterfield":"CHF","Fleetwood":"FLW","Gillingham":"GIL","MK Dons":"MKD","Salford":"SAL","Tranmere":"TRN","Walsall":"WLS","Newport":"NWP","Port Vale":"PVL","Bradford":"BRD","Notts":"NTC","Bromley":"BRO","Carlisle":"CRL","Morecambe":"MRC","Swindon":"SWI","Grimsby":"GRM","Colchester":"COL","Crewe":"CRE","Accrington":"AST"}
SHORT_TO_NAMES = {value:[] for value in NAMES_TO_SHORT.values()}
for name, short in NAMES_TO_SHORT.items():
    SHORT_TO_NAMES[short].append(name)

with open(r"C:\Users\gabeg\Documents\Accurate xG\OptaScrape\exampleEnv\updateResultApp\teamIDTable.csv", "r") as f:
    fileContents = f.readlines()
    teamIdDict = {header: [] for header in fileContents[0].split(",")}
    data = fileContents[1:]
    for row in data:
        headerVals = row.split(",")
        for i,key in enumerate(teamIdDict):
            teamIdDict[key].append(headerVals[i])

shortToId = {}
print("starting")
for shortKey in teamElos:
    possibleLongNames = SHORT_TO_NAMES[shortKey]
    for rowID in range(len(teamIdDict["teamID"])):
        for header in teamIdDict:
            for name in possibleLongNames:
                if name in teamIdDict[header][rowID]:
                    shortToId[shortKey] = teamIdDict["teamID"][rowID]
                    
print("done")
with open("C:\\Users\\gabeg\\Documents\\Accurate xG\\OptaScrape\\exampleEnv/updateResultApp/existingElos.json", "r+") as f:
    obj = json.load(f)
    f.seek(0)
    f.truncate()
    for key,value in shortToId.items():
        obj[value] = {"eloHistory":teamElos[key], "mids":[obj["mid"] for obj in sorted(filter(lambda obj: obj["home"] == key or obj["away"] == key,matchData), key=lambda obj: datetime.datetime.strptime(obj["date"],"%b %d, %Y").timestamp())]}
    json.dump(obj,f)




teamsByElo = sorted(teamElos,reverse=True,key=lambda x: sum(teamElos[x])/len(teamElos[x]))
order = ""
for i,team in enumerate(teamsByElo):
    order += f"{i+1}. {team}: {sum(teamElos[team])/len(teamElos[team])}\n"

print(order)
allElos = []
for team in sorted(teamElos,reverse=True,key=lambda x: teamElos[x][-1]):
    print(f"Team {team} has a current ELO score of {teamElos[team][-1]}. Their average elo this season is {sum(teamElos[team])/len(teamElos[team])}")
    allElos += teamElos[team]

print(f"Mean of elos is {np.mean(allElos)} and the standard deviation is {np.std(allElos)}") #should be 0
exit()
"""mlInputs = []
mlTargets = []
LENGTH_MATCH_INFO = 6
for team in mlGameData:
    for i in range(0,len(mlGameData[team])-(ROLLING_VALUE)*LENGTH_MATCH_INFO,LENGTH_MATCH_INFO): #there is no 'next game' after the final rolling space is at the end of the array, so don't include that
        homeElo = mlGameData[team][0+i:ROLLING_VALUE*LENGTH_MATCH_INFO+i:LENGTH_MATCH_INFO]
        awayElo = mlGameData[team][1+i:ROLLING_VALUE*LENGTH_MATCH_INFO+i:LENGTH_MATCH_INFO]
        wdl = []
        for j in range(2,5):
            wdl += mlGameData[team][j+i:ROLLING_VALUE*LENGTH_MATCH_INFO+i:LENGTH_MATCH_INFO]
        homeOrAway = mlGameData[team][5+i:ROLLING_VALUE*LENGTH_MATCH_INFO+i:LENGTH_MATCH_INFO]
        nextElos = mlGameData[team][ROLLING_VALUE*LENGTH_MATCH_INFO+i:ROLLING_VALUE*LENGTH_MATCH_INFO+2+i]
        nextMatchWDL = mlGameData[team][ROLLING_VALUE*LENGTH_MATCH_INFO+2+i:ROLLING_VALUE*LENGTH_MATCH_INFO+5+i]
        nextMatchHomeOrAway = mlGameData[team][ROLLING_VALUE*LENGTH_MATCH_INFO+5+i]
        mlInputs.append(homeElo+awayElo+wdl+homeOrAway+nextElos+[nextMatchHomeOrAway])
        mlTargets.append(nextMatchWDL)
    '''for i in range(len(teamElos[team])-ROLLING_VALUE-1):
        avgElo = teamElos[team][i:i+ROLLING_VALUE] #sum(teamElos[team])/len(teamElos[team])#
        inputs = [avgElo]
        rollingWDL = teamWDL[team][i:i+ROLLING_VALUE]
        inputs += [(w,d,l) for w,d,l in rollingWDL]
        target = teamWDL[team][i+ROLLING_VALUE]
        mlTargets.append([item for item in target])
        mlInputs.append(inputs)'''
"""
    

mlInputs = []
mlTargets = []

def getMlInputsFromGames(dictKeys:list,matchData:dict,teamName:str):
    inputs = []

    minDay = None
    maxDay = 0
    for key in dictKeys:
        inputs.extend(matchData[key][:2])
        if key[:3] == teamName:
            home = 1
        else:
            home = 0
        if home:
            inputs.extend(matchData[key][2:5])
        else:
            inputs.extend(matchData[key][2:5][::-1])
        inputs.append(home)
        if minDay is None or matchData[key][5] < minDay:
            minDay = matchData[key][5]
        elif matchData[key][5] > maxDay:
            maxDay = matchData[key][5]
    """inputs.append(round((maxDay-minDay)/60/60/24,0))
    print(inputs[-1])"""
    return inputs
for team in teamElos:
    games = list(filter(lambda x: team in x,info.keys()))
    sortedGames = sorted(games,key=lambda x: list(info.keys()).index(x))
    predictionGames = sortedGames[ROLLING_VALUE:len(sortedGames)-1] #don't include the last game as it doesn't have 'next game' information
    previousGames = sortedGames[:ROLLING_VALUE] 
    for i,game in enumerate(predictionGames):
        homeTeam = game[:3]
        awayTeam = game[3:]

        if homeTeam != team:
            continue

        mlTargets.append(info[game][2:5])

        if homeTeam == team:
            home = True
            otherTeam = awayTeam
            thisTeam = homeTeam
        else:
            home = False
            otherTeam = homeTeam
            thisTeam = awayTeam
        

        otherTeamGames = list(filter(lambda x: (otherTeam == x[:3] or otherTeam == x[3:]) and info[x][5] < info[game][5],info.keys()))
        sortedOtherGames = sorted(otherTeamGames,reverse=True,key=lambda x: info[x][5])[:ROLLING_VALUE][::-1]
        mlInputs.append(getMlInputsFromGames(previousGames,info,thisTeam)+getMlInputsFromGames(sortedOtherGames,info,otherTeam)+info[game][0:2])
        previousGames.pop(0)
        previousGames.append(game)
#mlInputs = [ [element for tup in inputLayer[0:] for element in tup] for inputLayer in mlInputs]#[inputLayer[0]] +

# Example training data
# X is the input features, y is the target probabilities
x = np.array(mlInputs)
y = np.array(mlTargets)
print(x.shape,y.shape)
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

# Define the model
model = keras.Sequential([
    layers.Dense(400, activation='relu', input_shape=(x.shape[1],)),
    layers.Dense(400, activation='relu'),
    layers.Dense(400, activation='relu'),
    layers.Dense(400, activation='relu'),
    layers.Dense(400, activation='relu'),
    layers.Dense(3, activation='softmax')  # Ensure output sums to 1
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
model.fit(xTrain, yTrain, epochs=10, batch_size=32)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(xTest, yTest)

predictions = model.predict(xTest)

countCorrectOutcome = 0
countCorrectOutcomeConfident = 0
confidentPredictions = 0
for iter,test in enumerate(xTest):
    print(yTest[iter],predictions[iter])
    if list(yTest[iter]).index(max(yTest[iter])) == list(predictions[iter]).index(max(predictions[iter])):
        countCorrectOutcome += 1
    if list(yTest[iter]).index(max(yTest[iter])) == list(predictions[iter]).index(max(predictions[iter])) and max(predictions[iter]) > 0.6:
        countCorrectOutcomeConfident += 1
    if max(predictions[iter]) > 0.6:
        confidentPredictions += 1
print(f"Correct Outcome {round(countCorrectOutcome/len(xTest)*100,3)}%  ({countCorrectOutcome}/{len(xTest)}) and confident correct outcome: {round(countCorrectOutcomeConfident/confidentPredictions*100,3)}% ({countCorrectOutcomeConfident}/{confidentPredictions})")

print(f"Model was trained using {len(xTrain)} pieces of data, and tested on {len(xTest)} pieces of data. The results of the test were a test loss (MSE) of {loss} (RMSE: {loss**0.5}), and a test MAE of {accuracy}")
if input("Would you like to save model?\n").lower() == "yes":
    model.save_weights(f"./local/wdl_model_{int(time.time()*1000)}.weights.h5")


model = keras.Sequential([
    layers.Dense(400, activation='relu', input_shape=(x.shape[1],)),
    layers.Dense(400, activation='relu'),
    layers.Dense(400, activation='relu'),
    layers.Dense(400, activation='relu'),
    layers.Dense(400, activation='relu'),
    layers.Dense(3, activation='softmax')  # Ensure output sums to 1
])

model.load_weights(r"C:\Users\gabeg\Documents\Accurate xG\accuratexG\local\wdl_model_1744145532897.weights.h5")


while input("Do you want to continue? (Y/N):\n").lower() == "y":
    homeTeam = input("Enter the home team 3-letter acronym:\n")
    awayTeam = input("Enter the away team 3-letter acronym:\n")

    games = list(filter(lambda x: homeTeam in x, info.keys()))
    sortedGames = sorted(games,key=lambda x: list(info.keys()).index(x))
    previousGames = sortedGames[-ROLLING_VALUE:]
    otherTeamGames = list(filter(lambda x: (awayTeam == x[:3] or awayTeam == x[3:]) and info[x][5] < datetime.datetime.timestamp(datetime.datetime.now()),info.keys()))
    sortedOtherGames = sorted(otherTeamGames,reverse=True,key=lambda x: info[x][5])[:ROLLING_VALUE][::-1]
    inputs = getMlInputsFromGames(previousGames,info,homeTeam)+getMlInputsFromGames(sortedOtherGames,info,awayTeam)+[teamElos[homeTeam][-1],teamElos[awayTeam][-1]]
    inputs = np.array([inputs])
    results = model.predict(inputs)[0]
    print(f"Model Predicts that {homeTeam} vs {awayTeam} will have the following w/d/l percentages:\nwin - {round(results[0]*100,2)}%\ndraw - {round(results[1]*100,2)}%\nlose - {round(results[2]*100,2)}%")
