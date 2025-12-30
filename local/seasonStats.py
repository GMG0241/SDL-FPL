import re
import os
from numpy import arange
from numpy import poly
from math import comb
from itertools import permutations
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def loadFile(fileName=None):
    MAX_TEAMS = 2 #max teams per game
    data = {i:[] for i in range(MAX_TEAMS)}
    try:
        with open(fileName,"r") as f:
            for line in f.readlines():
                team,value = line.split(",")[:2] # solution currently doesn't use extra data provided by the file
                data[int(team)].append(float(value))
    except Exception as e:
        print("Cannot find file provided, or file is not structured correctly")
        fileName = None
    return data

def generateTable(teamData):
    SIZE = len(teamData)
    tablePDF = {}

    #generate polymonial coefficients, and then calculate the sum of roots
    polyCoeffs = poly(teamData)
    #print(polyCoeffs)
    calcs = [polyCoeffs[i]/polyCoeffs[0]*(-1)**i for i in range(len(polyCoeffs))] #the 'ith' value represents the sum of the unique combinations of size i
    for x in range(SIZE,-1,-1):
        calc = calcs[x]
        coef = 1
        for i in range(x,SIZE):
            coef = comb(i+1,x)
            calc -= coef*tablePDF[i+1]
        #print(x,coef)
        tablePDF[x] = calc
    return tablePDF

def filterFiles(files,season,teams,suffix=""):
    return [f for f in files if re.match("("+"|".join(teams)+")(?!\\1)("+"|".join(teams)+")"+season+suffix+"\\.txt",f)]

def calcWinDrawLoseProb(results):
    team0WinProb, drawProb, team0LoseProb = 0,0,0
    for result in results:
        team0, team1 = result.split("-")
        if int(team0) - int(team1) < 0:
            team0LoseProb += results[result]
        elif int(team0) - int(team1) == 0:
            drawProb += results[result]
        else:
            team0WinProb += results[result]
    return (team0WinProb,drawProb,team0LoseProb)

def expectedPoints(wld):
    

    points = {0:1} #100% chance of 0 points from 0 games
    for matchNum,match in enumerate(wld):
        possiblePoints = list(range(matchNum*3+4)) #calculate possible points from matchNum
        possiblePoints.remove(matchNum*3+2)
        pointsCopy = points.copy()
        for pointNum in possiblePoints:
            try:
                losePoints = pointsCopy[pointNum]
            except KeyError:
                losePoints = 0
            try:
                drawPoints = pointsCopy[pointNum-1]
            except KeyError:
                drawPoints = 0
            try:
                winPoints = pointsCopy[pointNum-3]
            except KeyError:
                winPoints = 0
            points[pointNum] = winPoints*match[0] + drawPoints*match[1] + losePoints*match[2] # points[pointNum] = winPoints*match[win] + drawPoints*match[draw] + losePoints*match[lose]
    return points,len(wld)


import collections
import numpy as np


class PoiBin(object):
    """Poisson Binomial distribution for random variables.

    This class implements the Poisson Binomial distribution for Bernoulli
    trials with different success probabilities. The distribution describes
    thus a random variable that is the sum of independent and not identically
    distributed single Bernoulli random variables.

    The class offers methods for calculating the probability mass function, the
    cumulative distribution function, and p-values for right-sided testing.
    """

    def __init__(self, probabilities):
        """Initialize the class and calculate the ``pmf`` and ``cdf``.

        :param probabilities: sequence of success probabilities :math:`p_i \\in
            [0, 1] \\forall i \\in [0, N]` for :math:`N` independent but not
            identically distributed Bernoulli random variables
        :type probabilities: numpy.array
        """
        self.success_probabilities = np.array(probabilities)
        self.number_trials = self.success_probabilities.size
        self.check_input_prob()
        self.omega = 2 * np.pi / (self.number_trials + 1)
        self.pmf_list = self.get_pmf_xi()
        self.cdf_list = self.get_cdf(self.pmf_list)

# ------------------------------------------------------------------------------
# Methods for the Poisson Binomial Distribution
# ------------------------------------------------------------------------------

    def pmf(self, number_successes):
        """Calculate the probability mass function ``pmf`` for the input values.

        The ``pmf`` is defined as

        .. math::

            pmf(k) = Pr(X = k), k = 0, 1, ..., n.

        :param number_successes: number of successful trials for which the
            probability mass function is calculated
        :type number_successes: int or list of integers
        """
        self.check_rv_input(number_successes)
        return self.pmf_list[number_successes]

    def cdf(self, number_successes):
        """Calculate the cumulative distribution function for the input values.

        The cumulative distribution function ``cdf`` for a number ``k`` of
        successes is defined as

        .. math::

            cdf(k) = Pr(X \\leq k), k = 0, 1, ..., n.

        :param number_successes: number of successful trials for which the
            cumulative distribution function is calculated
        :type number_successes: int or list of integers
        """
        self.check_rv_input(number_successes)
        return self.cdf_list[number_successes]

    def pval(self, number_successes):
        """Return the p-values corresponding to the input numbers of successes.

        The p-values for right-sided testing are defined as

        .. math::

            pval(k) = Pr(X \\geq k ),  k = 0, 1, ..., n.

        .. note::

            Since :math:`cdf(k) = Pr(X <= k)`, the function returns

            .. math::

                1 - cdf(X < k) & = 1 - cdf(X <= k - 1)
                               & = 1 - cdf(X <= k) + pmf(X = k),

                               k = 0, 1, .., n.

        :param number_successes: number of successful trials for which the
            p-value is calculated
        :type number_successes: int, numpy.array, or list of integers
        """
        self.check_rv_input(number_successes)
        i = 0
        try:
            isinstance(number_successes, collections.Iterable)
            pvalues = np.array(number_successes, dtype='float')
            # if input is iterable (list, numpy.array):
            for k in number_successes:
                pvalues[i] = 1. - self.cdf(k) + self.pmf(k)
                i += 1
            return pvalues
        except TypeError:
            # if input is an integer:
            if number_successes == 0:
                return 1
            else:
                return 1 - self.cdf(number_successes - 1)

# ------------------------------------------------------------------------------
# Methods to obtain pmf and cdf
# ------------------------------------------------------------------------------

    def get_cdf(self, event_probabilities):
        """Return the values of the cumulative density function.

        Return a list which contains all the values of the cumulative
        density function for :math:`i = 0, 1, ..., n`.

        :param event_probabilities: array of single event probabilities
        :type event_probabilities: numpy.array
        """
        cdf = np.empty(self.number_trials + 1)
        cdf[0] = event_probabilities[0]
        for i in range(1, self.number_trials + 1):
            cdf[i] = cdf[i - 1] + event_probabilities[i]
        return cdf

    def get_pmf_xi(self):
        """Return the values of the variable ``xi``.

        The components ``xi`` make up the probability mass function, i.e.
        :math:`\\xi(k) = pmf(k) = Pr(X = k)`.
        """
        chi = np.empty(self.number_trials + 1, dtype=complex)
        chi[0] = 1
        half_number_trials = int(
            self.number_trials / 2 + self.number_trials % 2)
        # set first half of chis:
        chi[1:half_number_trials + 1] = self.get_chi(
            np.arange(1, half_number_trials + 1))
        # set second half of chis:
        chi[half_number_trials + 1:self.number_trials + 1] = np.conjugate(
            chi[1:self.number_trials - half_number_trials + 1] [::-1])
        chi /= self.number_trials + 1
        xi = np.fft.fft(chi)
        if self.check_xi_are_real(xi):
            xi = xi.real
        else:
            raise TypeError("pmf / xi values have to be real.")
        xi += np.finfo(type(xi[0])).eps
        return xi

    def get_chi(self, idx_array):
        """Return the values of ``chi`` for the specified indices.

        :param idx_array: array of indices for which the ``chi`` values should
            be calculated
        :type idx_array: numpy.array
        """
        # get_z:
        exp_value = np.exp(self.omega * idx_array * 1j)
        xy = 1 - self.success_probabilities + \
            self.success_probabilities * exp_value[:, np.newaxis]
        # sum over the principal values of the arguments of z:
        argz_sum = np.arctan2(xy.imag, xy.real).sum(axis=1)
        # get d value:
        exparg = np.log(np.abs(xy)).sum(axis=1)
        d_value = np.exp(exparg)
        # get chi values:
        chi = d_value * np.exp(argz_sum * 1j)
        return chi

# ------------------------------------------------------------------------------
# Auxiliary functions
# ------------------------------------------------------------------------------

    def check_rv_input(self, number_successes):
        """Assert that the input values ``number_successes`` are OK.

        The input values ``number_successes`` for the random variable have to be
        integers, greater or equal to 0, and smaller or equal to the total
        number of trials ``self.number_trials``.

        :param number_successes: number of successful trials
        :type number_successes: int or list of integers """
        try:
            for k in number_successes:
                assert (type(k) == int or type(k) == np.int64), \
                        "Values in input list must be integers"
                assert k >= 0, 'Values in input list cannot be negative.'
                assert k <= self.number_trials, \
                    'Values in input list must be smaller or equal to the ' \
                    'number of input probabilities "n"'
        except TypeError:
            assert (type(number_successes) == int or \
                type(number_successes) == np.int64), \
                'Input value must be an integer.'
            assert number_successes >= 0, "Input value cannot be negative."
            assert number_successes <= self.number_trials, \
                'Input value cannot be greater than ' + str(self.number_trials)
        return True

    @staticmethod
    def check_xi_are_real(xi_values):
        """Check whether all the ``xi``s have imaginary part equal to 0.

        The probabilities :math:`\\xi(k) = pmf(k) = Pr(X = k)` have to be
        positive and must have imaginary part equal to zero.

        :param xi_values: single event probabilities
        :type xi_values: complex
        """
        return np.all(xi_values.imag <= np.finfo(float).eps)

    def check_input_prob(self):
        """Check that all the input probabilities are in the interval [0, 1]."""
        if self.success_probabilities.shape != (self.number_trials,):
            raise ValueError(
                "Input must be an one-dimensional array or a list.")
        if not np.all(self.success_probabilities >= 0):
            raise ValueError("Input probabilities have to be non negative.")
        if not np.all(self.success_probabilities <= 1):
            raise ValueError("Input probabilities have to be smaller than 1.")

TEAMS = {"ARS":"Arsenal","AVL":"Aston Villa","BHA":"Brighton","BOU":"Bournemouth","BRE":"Brentford","CHE":"Chelsea","CRY":"Crystal Palace","EVE":"Everton","FUL":"Fulham","IPS":"Ipswich","LCI":"Leicester City","LIV":"Liverpool","MNC":"Manchester City","MNU":"Manchester United","NEW":"Newcastle","NOT":"Nottingham Forrest","SOU":"Southampton","TOT":"Tottenham","WHU":"West Ham","WOV":"Wolves"}
CURRENT_SEASON = "2425"
STANDINGS_HALF = {"LIV":[19,46],"ARS":[20,40],"NOT":[20,40],"CHE":[20,36],"NEW":[20,35],"MNC":[20,34],"BOU":[20,33],"FUL":[20,30],"AVL":[20,32],"BHA":[20,28],"TOT":[20,24],"BRE":[20,27],"WHU":[20,23],"MNU":[20,23],"CRY":[20,21],"EVE":[19,17],"WOV":[20,16],"IPS":[20,16],"LCI":[20,14],"SOU":[20,6]}

'''#ConstraintSolver code from Claude 3.5
from functools import lru_cache

class ConstraintSolver:
    def __init__(self, n_vars, possible_values, constraints):
        self.n_vars = n_vars
        self.possible_values = possible_values
        self.constraints = constraints
        self.solutions = []

    @lru_cache(maxsize=None)
    def evaluate_constraint(self, constraint_idx, partial_solution):
        constraint = self.constraints[constraint_idx]
        return constraint(partial_solution)

    def solve(self):
        self.backtrack([None] * self.n_vars, 0)
        return self.solutions

    def backtrack(self, partial_solution, var_idx):
        if var_idx == self.n_vars:
            self.solutions.append(tuple(partial_solution))
            if len(self.solutions) % 1000 == 0:
                print("identified %d solutions" %(len(self.solutions)))
            return

        for value in self.possible_values:
            partial_solution[var_idx] = value
            if self.is_valid_partial(partial_solution, var_idx + 1):
                self.backtrack(partial_solution, var_idx + 1)

    def is_valid_partial(self, partial_solution, length):
        return all(
            self.evaluate_constraint(i, tuple(partial_solution[:length])) 
            for i, constraint in enumerate(self.constraints)
            if constraint.can_evaluate(length)
        )

class Constraint:
    def __init__(self, func, funcNumInv, funcN, required_vars):
        self.func = func
        self.funcN = funcN
        self.funcNumInv = funcNumInv
        self.required_vars = required_vars

    def __call__(self, partial_solution):
        return self.func(self.funcNumInv,self.funcN,*[partial_solution[i] for i in self.required_vars])

    def can_evaluate(self, length):
        return all(i < length for i in self.required_vars)

# Example usage

def constraintGeneral(numInv,n,*args):
    calc = 0
    for indexCount,variableValue in enumerate(args):
        if indexCount < numInv:
            calc += (2-variableValue)**2 + 2 - variableValue
        else:
            calc += variableValue**2 + variableValue
    return calc < n

constraints = [ #numbers represent variable i.e. 0=a, 1=b, 2=c
    Constraint(constraintGeneral, 0,8, (0, 1)), 
    Constraint(constraintGeneral,1,10, (0, 2)),
    Constraint(constraintGeneral,2,8, (1, 2))
]

solver = ConstraintSolver(3, [0, 1, 2], constraints)
solutions = solver.solve()

print("Valid combinations:")
for solution in solutions:
    print(solution)

constraints = [ #numbers represent variable i.e. 0=a, 1=b, 2=c
    Constraint(constraintGeneral, 0,8, (0, 1)), 
    Constraint(constraintGeneral,0,6, (0, 1)),
]
solver = ConstraintSolver(2, [2,1,0], constraints)
solutions = solver.solve()

print("Valid combinations:")
for solution in solutions:
    print(solution)

constraints = [ #numbers represent variable i.e. 0=a, 1=b, 2=c
    Constraint(constraintGeneral, 0,8, [0]), 
    Constraint(constraintGeneral,1,6, [0]),
]
solver = ConstraintSolver(1, [0, 1, 2], constraints)
solutions = solver.solve()

print("Valid combinations:")
for solution in solutions:
    print(solution)
#solving for when A=DDW
numTeams = 4#len(TEAMS)
numVariablesSolving = int((numTeams-2)/2*(numTeams-1))
wdlEncoding = [0, 1, 2]
aResult = [1,1,2]
xorTable = [[None]*(numTeams-1) for i in range(numTeams)]
uniqueCount = 0
for x,row in enumerate(xorTable):
    for y,column in enumerate(xorTable[x]):
        if xorTable[x][y] is None:
            xorTable[x][y] = uniqueCount
            xorTable[y+1][x] = uniqueCount
            uniqueCount +=1
print(xorTable)
#trim values of A
xorTable.pop(0)
for i in range(len(xorTable)):
    xorTable[i].pop(0)
print(xorTable)
#reduce numerical values
smallest = min(min(xorTable))
for x in range(len(xorTable)):
    for y in range(len(xorTable[x])):
        xorTable[x][y] = xorTable[x][y]-smallest
print(xorTable)


constraints = [Constraint(constraintGeneral,i,sum([x**2+x for x in aResult])-((-aResult[i]+2)**2 -aResult[i]+2),xorTable[i]) for i in range(numTeams-1)]
for constraint in constraints:
    print(constraint.funcN,constraint.funcNumInv,constraint.required_vars)
constraints = [ #numbers represent variable i.e. 0=a, 1=b, 2=c
    Constraint(constraintGeneral,0,8, (0, 1)), 
    Constraint(constraintGeneral,1,8, (0, 2)),
    Constraint(constraintGeneral,2,12, (1, 2))
]
solver = ConstraintSolver(numVariablesSolving, wdlEncoding, constraints)
solutions = solver.solve()

print("Valid combinations:")
for solution in solutions:
    print(solution)

#solving for when A=DDW
numTeams = len(TEAMS)
numVariablesSolving = int((numTeams-2)/2*(numTeams-1))
wdlEncoding = [2,1,0]
aResult = [2]*19#+[1]*9+[0]
xorTable = [[None]*(numTeams-1) for i in range(numTeams)]
uniqueCount = 0
for x,row in enumerate(xorTable):
    for y,column in enumerate(xorTable[x]):
        if xorTable[x][y] is None:
            xorTable[x][y] = uniqueCount
            xorTable[y+1][x] = uniqueCount
            uniqueCount +=1
print(xorTable)
#trim values of A
xorTable.pop(0)
for i in range(len(xorTable)):
    xorTable[i].pop(0)
print(xorTable)
#reduce numerical values
smallest = min(min(xorTable))
for x in range(len(xorTable)):
    for y in range(len(xorTable[x])):
        xorTable[x][y] = xorTable[x][y]-smallest
print(xorTable)


constraints = [Constraint(constraintGeneral,i,sum([x**2+x for x in aResult])-((-aResult[i]+2)**2 -aResult[i]+2),xorTable[i]) for i in range(numTeams-1)]
for constraint in constraints:
    print(constraint.funcN,constraint.funcNumInv,constraint.required_vars)
constraints = [ #numbers represent variable i.e. 0=a, 1=b, 2=c
    Constraint(constraintGeneral,0,8, (0, 1)), 
    Constraint(constraintGeneral,1,8, (0, 2)),
    Constraint(constraintGeneral,2,12, (1, 2))
]
solver = ConstraintSolver(numVariablesSolving, wdlEncoding, constraints)
solutions = solver.solve()

print("Valid combinations:")
for solution in solutions:
    print(solution)
exit()'''
'''result = generateTable([1]*250)
print(sum([result[x] for x in result]))
print(result)

exit()'''
files = [f for f in os.listdir("C:\\Users\\gabeg\\Documents\\Accurate xG\\accuratexG\\local\\") if os.path.isfile(os.path.join("C:\\Users\\gabeg\\Documents\\Accurate xG\\accuratexG\\local\\", f)) and f.endswith(".txt")]
print(len(files))
currentFiles = filterFiles(files,CURRENT_SEASON,list(TEAMS))
print(len(currentFiles))

expectedPointsPerTeam = {}
for selectedTeam in TEAMS:
    wld = []
    for file in currentFiles:
        if selectedTeam in file:
            data = loadFile("local/"+file)
            if file.startswith(selectedTeam):
                tblTeam0 = generateTable(data[0])
                tblTeam1 = generateTable(data[1])
            else:
                tblTeam0 = generateTable(data[1])
                tblTeam1 = generateTable(data[0])
            results = {}
            for result0 in tblTeam0: #yes, I know I could have done this better
                for result1 in tblTeam1:
                    results[str(result0)+"-"+str(result1)] = tblTeam0[result0]*tblTeam1[result1]
            wld.append(calcWinDrawLoseProb(results))
    points,games = expectedPoints(wld)
    sortedResults = sorted(points,reverse=True,key=lambda x: points[x])
    sortedResults = {sortedResult: points[sortedResult] for sortedResult in sortedResults}
    expectedPointsPerTeam[selectedTeam] = sortedResults

table = {} #to be taken with a pinch of salt - doesn't accurately represent the probability of place finish
for team in expectedPointsPerTeam:
    pointsProb = expectedPointsPerTeam[team]
    table[team] = list(pointsProb.keys())[0]
sortedTable = sorted(table,reverse=True,key=lambda x: table[x])
sortedTable = {sortedResult: table[sortedResult] for sortedResult in sortedTable}
print(sortedTable)
columns = "Team|Actual Place|xG Position|Position Diff|Games Played|Actual Points|xG Points|xPoints (Expected Value)|Points Diff".split("|")
data = ["%s|%d|%d|%d|%d|%d|%d|%.2f|%d" %(TEAMS[team],i+1,list(sortedTable.keys()).index(team)+1,list(sortedTable.keys()).index(team)-i,STANDINGS_HALF[team][0],STANDINGS_HALF[team][1],sortedTable[team],sum([expectedPointsPerTeam[team][x]*x for x in expectedPointsPerTeam[team]]),STANDINGS_HALF[team][1]-sortedTable[team]) for i,team in enumerate(STANDINGS_HALF)]
data = [x.split("|") for x in data]
print("|".join(columns)) #printing values for alt text
print("\n".join(["|".join(team) for team in data])) #printing values for alt text
fig = go.Figure(data=[go.Table(header=dict(values=columns),
                 cells=dict(values=[[data[j][i] for j in range(len(TEAMS))] for i in range(len(columns))]))
                     ])
fig.show()
selectedTeam = input("Enter the acronym for the team you want to find the expected points for:\n")
wld = []
allGoals = []
for file in currentFiles:
    if selectedTeam in file:
        data = loadFile("local/"+file)
        if file.startswith(selectedTeam): #turn into one if statement, where it determines the index i.e. home team =0, or home team  = 1, and then use home team as the variable index
            tblTeam0 = generateTable(data[0])
            tblTeam1 = generateTable(data[1])
            allGoals += data[0]
        else:
            tblTeam0 = generateTable(data[1])
            tblTeam1 = generateTable(data[0])
            allGoals += data[1]
        results = {}
        for result0 in tblTeam0: #yes, I know I could have done this better
            for result1 in tblTeam1:
                results[str(result0)+"-"+str(result1)] = tblTeam0[result0]*tblTeam1[result1]
        wld.append(calcWinDrawLoseProb(results))
points,games = expectedPoints(wld)
values = [points[x] for x in points]
print(sum(values))
poiBinCalc = PoiBin(allGoals)
allGoalsPdf = {}
for i in range(len(allGoals)+1):
    allGoalsPdf[i] = poiBinCalc.pmf(i)
print(allGoalsPdf)
print(sum([allGoalsPdf[x] for x in allGoalsPdf]))
sortedResults = sorted(points,reverse=True,key=lambda x: points[x])
sortedResults = {sortedResult: points[sortedResult] for sortedResult in sortedResults}
print(sortedResults)
print(sum([key*sortedResults[key] for key in sortedResults]))#expected value
plt.bar(points.keys(),[prob*100 for prob in values])
plt.title("Expected Points for %s after %d games played" %(selectedTeam,games))
plt.ylabel("Probability (%)")
plt.xlabel("Points")
plt.xticks(arange(min(points.keys()), max(points.keys())+1, 1.0))
plt.show()

#prob(A = 1st) = Prob(A > B N B >= C) + Prob(A > C N C >= B) 
# P(A > B) = 0.72*(0.18+0.19+0.55+0.01+0.04) + 0.211*(0.55+0.01+0.19+0.18) + 0.049*(0.01+0.19+0.18) + 0.0095*(0.19+0.18) + 0.01*(0.18) + 0 = 0.918565
# P(B >= C) = 0.03*(1) + 0.04*(0.04+0.166+0.038+0.274+0.48) + 0.55*(0.166+0.038+0.274+0.48) + 0.01*(0.038+0.274+0.48) + 0.19*(0.274+0.48)+0.18*0.48 = 0.8344
# P(A > B N B >= C) = 0.918565*0.8344 = 0.766450636
# P(A > C) = 0.72*(0.04+0.038+0.166+0.274+0.48) + 0.211*(0.166+0.038+0.274+0.48) + 0.049*(0.038+0.274+0.48) + 0.0095*(0.274+0.48) + 0.01*0.48 + 0 = 0.971469
# P(C >= B) = 0.002*(1) + 0.04*(0.04+0.55+0.01+0.19+0.18) + 0.166*(0.55+0.01+0.19+0.18) + 0.038*(0.01+0.19+0.18) + 0.274*(0.19+0.18) + 0.48*(0.18) = 0.3974
# P(A > C N C >= B) = 0.971469*0.3974 = 0.3860617806
# prob(A = 1st) = Prob(A > B N B >= C) + Prob(A > C N C >= B) = 0.766450636 + 0.3860617806 = 1.1525124166
'''
1 game = [(0.2914628937016591, 0.34654378107851136, 0.36199332521982946)]

P(Points=3) = W
P(Points=1) = D
P(Points=0) = L

2 games = [(0.2914628937016591, 0.34654378107851136, 0.36199332521982946),(0.5692615152613695, 0.2744084233196509, 0.15633006141897965)]
P(Points=6) = W*W
P(Points=4) = W*D + D*W
P(Points=3) = W*L + L*W
P(Points=2) = D*D
P(Points=1) = D*L + L*D
P(Points=0) = L*L

3 games = [(0.2914628937016591, 0.34654378107851136, 0.36199332521982946), (0.5692615152613695, 0.2744084233196509, 0.15633006141897965), (0.8739774019048397, 0.09643973901861597, 0.029582859076544357)]

P(Points=9) = W*W*W
P(Points=7) = W*W*D + W*D*W + D*W*W
P(Points=6) = W*W*L + W*L*W + L*W*W
P(Points=5) = W*D*D + D*W*D + D*D*W
P(Points=4) = W*D*L + W*L*D + D*W*L + D*L*W + L*W*D +L*D*W
P(Points=3) = W*L*L + L*W*L + L*L*W + D*D*D
P(Points=2) = L*D*D + D*L*D + D*D*L
P(Points=1) = L*L*D + L*D*L + D*L*L
P(Points=0) = L*L*L

N games =
W*(N-1 games) + D*(N-1 games) + L*(N-1 games)  = (N-1 games)*(W+D+L)

E.G:

1 game =
P(Points=3) = 0.2914628937016591
P(Points=1) = 0.34654378107851136
P(Points=0) = 0.36199332521982946

2 games =
P(Points=6) = P(Points=3)*W + P(Points=5)*D + P(Points=6)*L = 0.2914628937016591*0.5692615152613695 + 0 + 0
P(Points=4) = P(Points=1)*W + P(Points=3)*D + P(Points=4)*L = 0.34654378107851136*0.5692615152613695 + 0.2914628937016591*0.2744084233196509 + 0
P(Points=3) = P(Points=0)*W + P(Points=2)*D + P(Points=3)*L  = 0.36199332521982946*0.5692615152613695 + 0 + 0.2914628937016591*0.15633006141897965
P(Points=2) = P(Points=-1)*W + P(Points=1)*D + P(Points=2)*L = 0 + 0.34654378107851136*0.2744084233196509 + 0
P(Points=1) = P(Points=-2)*W + P(Points=0)*D + P(Points=1)*L = 0 + 0.36199332521982946*0.2744084233196509 + 0.34654378107851136*0.15633006141897965
P(Points=0) = P(Points=-3)*W + P(Points=-1)*D + P(Points=0)*L = 0 + 0 + 0.36199332521982946*0.15633006141897965

Checksum 2 games:
6 -> 0.16591860851106993
4 -> 0.277253911038013
3 -> 0.25163328090287573
2 -> 0.09509453257698457
1 -> 0.15350922820618013
0 -> 0.05659043876487661
total ->  1

recursFunc(gameNumber, arrayOfProbs):
    if gameNumber == 1:
        return 3->arrayOfProbs[win],1->arrayOfProbs[draw],0->arrayOfProbs[lose]
    else:
        prevGame = recursFunc(gameNumber-1,arrayOfProbs[:(gameNumber-1)]) #dictionary or something
        points_N = prevGame[N-3]*arrayOfProbs[-1][win] + prevGame[N-1][draw] + prevGame[N][lose]
    for points:
        return n--> points_N


points = {0:1} #100% chance of 0 points from 0 games
for matchNum,match in enumerate(results):
    possiblePoints = #calculate possible points from matchNum
    for pointNum in possiblePoints:
        try:
            losePoints = points[pointNum]
        except KeyError:
            losePoints = 0
        try:
            drawPoints = points[pointNum-1]
        except KeyError:
            drawPoints = 0
        try:
            winPoints = points[pointNum-3]
        except KeyError:
            winPoints = 0
        points[pointNum] = winPoints*match[win] + drawPoints*match[draw] + losePoints*match[lose]
'''

'''
1 team table:
P(1st) = 1

2 team table (1 match)
P(1st) = P(points=3)*P(3 is enough for 1st) + P(points=1)*P(1 is enough for 1st) + P(points=0)*P(0 is enough for 1st)
P(2nd) = P(points=3)*P(3 is enough for 2nd AND 3 is not enough for 1st) + P(points=1)*P(1 is enough for 2nd AND 1 is not enough for 1st) + P(points=0)*P(0 is enough for 2nd AND 0 is not enough for 1st)

match (w,d,l) :
AvsB = (0.9,0.05,0.05)
P(1st) = 0.9*(1)+0.05*(1) +0.05*(0) = 0.95
P(2nd) = 0.9*(0)+0.05*(0)+0.05*(1) = 0.05

2 team table (2 matches):
P(1st) = P(points=6)*P(6 is enough for 1st) + P(points=4)*P(4 is enough for 1st) + P(points=3)*P(3 is enough for 1st) + P(points=2)*P(2 is enough for 1st) + P(points=1)*P(1 is enough for 1st) + P(points=0)*P(0 is enough for 1st)
P(2nd) = P(points=6)*P(6 is enough for 2nd AND 6 is not enough for 1st) + P(points=4)*P(4 is enough for 2nd AND 4 is not enough for 1st) + P(points=3)*P(3 is enough for 2nd AND 3 is not enough for 1st) + P(points=2)*P(2 is enough for 2nd AND 2 is not enough for 1st) + P(points=1)*P(1 is enough for 2nd AND 1 is not enough for 1st) + P(points=0)*P(0 is enough for 2nd AND 0 is not enough for 1st)

Assume No GD
2 team table example:

team A 2 games (w,d,l) = (0.9,0.05,0.05), (0.4,0.2,0.4)
P(1st) = (0.9*0.4)*(1) + (0.9*0.2+0.05*0.4)*(1) + (0.9*0.4+0.05*0.4)*(1) + (0.05*0.2)*(1) + P(points=1)*0 + P(points=0)*0 = 0.95
P(2nd) = P(points=6)*0 + P(points=4)*0 + (0.9*0.4+0.05*0.4)*(0) + (0.05*0.2)*(0) + (0.05*0.2+0.05*0.4)*(1) + (0.05*0.4)*(1) = 0.05

3 team table 1 game:

P(1st) = P(A = 6 N B < 6 N C < 6) + P(A = 4 N B < 4 N C < 4) + P(A = 3 N B < 3 N C < 3) + P(A = 2 N B < 2 N C < 2) + P(A = 1 N B < 1 N C < 1) + P(A = 0 N B < 0 N C < 0)
P(Joint 1st) = P(A = 6 N B = 6 U C = 6) + P(A = 4 N B = 4 U C = 4) + P(A = 3 N B = 3 U C = 3) + P(A = 2 N B = 2 U C = 2) + P(A = 1 N B = 1 U C = 1) + P(A = 0 N B = 0 U C = 0)
P(2nd) = P(A = 6 N (B> 6 U C > 6 N !(B > 6 N C > 6)) + P(A = 4 N (B> 4 U C > 4 N !(B > 4 N C > 4)) + P(A = 3 N (B> 3 U C > 3 N !(B > 3 N C > 3)) + P(A = 2 N (B> 2 U C > 2 N !(B > 2 N C > 2)) + P(A = 1 N (B> 1 U C > 1 N !(B > 1 N C > 1)) + P(A = 0 N (B> 0 U C > 0 N !(B > 0 N C > 0))
.
.

P(1st) = P(A = 6)*P(B < 6 N C < 6 | A = 6) + P(A = 4)*P(B < 4 N C < 4 | A = 4) + P(A = 3)*P(B < 3 N C < 3 | A = 3) + P(A = 2)*P(B < 2 N C < 2 | A = 2) + P(A = 1)*P(B < 1 N C < 1 | A = 1) + P(A = 0)*P(B < 0 N C < 0 | A = 0)
P(2nd) = P(A = 6)*P((B > 6 U C > 6 N !(B > 6 N C > 6) | A = 6) + P(A = 4)*P((B > 4 U C > 4 N !(B > 4 N C > 4) | A = 4) + P(A = 3)*P((B > 3 U C > 3 N !(B > 3 N C > 3) | A = 3) + P(A = 2)*P((B > 2 U C > 2 N !(B > 2 N C > 2) | A = 2) + P(A = 1)*P((B > 1 U C > 1 N !(B > 1 N C > 1) | A = 1) + P(A = 0)*P((B > 0 U C > 0 N !(B > 0 N C > 0) | A = 0)
3 team table example:

games (w,l,d):
team A vs team B (0.9,0.05,0.05)
team A vs team C (0.8,0.19,0.01)
team B vs team C (0.6,0.2,0.2)

  1,2,3 (SUM)
A 0.9463,0.0491,0.0046 (1)
B 0.0872,0.6868,0.226 (1)
C 0.0576,0.3381,0.6043 (1)

columns don't add up to 1, I think because of joint places

0.9*0.8*0.6 --> A:6,B:3,C:0 --> A:1st,B:2nd,C:3rd
0.9*0.8*0.2 --> A:6,B:1,C:1 --> A:1st,B:j2nd,C:j2nd
0.9*0.8*0.2 --> A:6,B:0,C:3 --> A:1st,B:3rd,C:2nd
0.9*0.19*0.6 --> A:4,B:3,C:1 --> A:1st,B:2nd,C:3rd
0.9*0.19*0.2 --> A:4,B:1,C:2 --> A:1st,B:3rd,C:2nd
0.9*0.19*0.2 --> A:4,B:0,C:4 --> A:j1st,B:3rd,C:j1st
0.05*0.8*0.6 --> A:4,B:4,C:0 --> A:j1st,B:j1st,C:3rd
0.05*0.8*0.2 --> A:4,B:2,C:1 --> A:1st,B:2nd,C:3rd
0.05*0.8*0.2 --> A:4,B:1,C:3 --> A:1st,B:3rd,C:2nd
0.9*0.01*0.6 --> A:3,B:3,C:3 --> A:j1st,B:j1st,C:j1st
0.9*0.01*0.2 --> A:3,B:1,C:4 --> A:2nd,B:3rd,C:1st
0.9*0.01*0.2 --> A:3,B:0,C:6 --> A:2nd,B:3rd,C:1st
0.05*0.8*0.6 --> A:3,B:6,C:0 --> A:2nd,B:1st,C:3rd
0.05*0.8*0.2 --> A:3,B:4,C:1 --> A:2nd,B:1st,C:2rd
0.05*0.8*0.2 --> A:3,B:3,C:3 --> A:j1st,B:j1st,C:j1st
0.05*0.19*0.6 --> A:2,B:4,C:1 --> A:2nd,B:1st,C:3rd
0.05*0.19*0.2 --> A:2,B:2,C:2 --> A:j1st,B:j1st,C:j1st
0.05*0.19*0.2 --> A:2,B:1,C:4 --> A:2nd,B:3rd,C:1st
0.05*0.01*0.6 --> A:1,B:4,C:3 --> A:3rd,B:1st,C:2nd
0.05*0.01*0.2 --> A:1,B:2,C:4 --> A:3rd,B:2nd,C:1st
0.05*0.01*0.2 --> A:1,B:1,C:6 --> A:j2nd,B:j2nd,C:1st
0.05*0.19*0.6 --> A:1,B:6,C:1 --> A:j2nd,B:1st,C:j2nd
0.05*0.19*0.2 --> A:1,B:4,C:2 --> A:3rd,B:1st,C:2nd
0.05*0.19*0.2 --> A:1,B:3,C:4 --> A:3rd,B:2nd,C:1st
0.05*0.01*0.6 --> A:0,B:6,C:3 --> A:3rd,B:1st,C:2nd
0.05*0.01*0.2 --> A:0,B:4,C:4 --> A:3rd,B:j1st,C:j1st
0.05*0.01*0.2 --> A:0,B:3,C:6 --> A:3rd,B:2nd,C:1st

by inspection:
P(A=1st) = 0.9*0.8*0.6+0.9*0.8*0.2+0.9*0.8*0.2+0.9*0.19*0.6+0.9*0.19*0.2+0.05*0.8*0.2+0.05*0.8*0.2 = 0.8728
P(A=j1st) = 0.9*0.19*0.2+0.05*0.8*0.6+0.9*0.01*0.6+0.05*0.8*0.2+0.05*0.19*0.2 = 0.0735
P(A=2nd) = 0.9*0.01*0.2+0.9*0.01*0.2+0.05*0.8*0.6+0.05*0.8*0.2+0.05*0.19*0.6+0.05*0.19*0.2 = 0.0432
P(A=j2nd) = 0.05*0.01*0.2+0.05*0.19*0.6 = 0.0058
P(A=3rd) = 0.05*0.01*0.6+0.05*0.01*0.2+0.05*0.19*0.2+0.05*0.19*0.2+0.05*0.01*0.6+0.05*0.01*0.2+0.05*0.01*0.2 = 0.0047
P(A=j3rd) = 0
P(B=1st) = 0.05*0.8*0.6+0.05*0.8*0.2+0.05*0.19*0.6+0.05*0.01*0.6+0.05*0.19*0.6+0.05*0.19*0.2+0.05*0.01*0.6 = 0.0459
P(B=j1st) = 0.05*0.8*0.6+0.9*0.01*0.6+0.05*0.8*0.2+0.05*0.19*0.2+0.05*0.01*0.2 = 0.0394
P(C=1st) = 0.9*0.01*0.2+0.9*0.01*0.2+0.05*0.19*0.2+0.05*0.01*0.2+0.05*0.01*0.2+0.05*0.19*0.2+0.05*0.01*0.2 = 0.0077
P(C=j1st) = 0.9*0.19*0.2 +0.9*0.01*0.6+0.05*0.8*0.2+0.05*0.19*0.2+0.05*0.01*0.2 = 0.0496
SUM = 0.8728+0.0735+0.0432+0.0058+0.0047+0 = 1
by (attempted) calculation
A:
P(1st) = (0.9*0.8)*(1) + (0.9*0.19+0.05*0.8)*((0.6+0.2+0.2+0.2)/(0.6+0.2+0.2+0.6+0.2+0.2)) + (0.9*0.01+0.05*0.8)*(0) + (0.5*0.19)*(0) + (0.05*0.01+0.05*0.19)*(0) + (0.05*0.01)*0 = 0.8466
P(Joint 1st) = (0.9*0.8)*(0) + (0.9*0.19+0.05*0.8)*((0.6+0.2)/2) + (0.9*0.01+0.05*0.8)*((0.6+0.2)/2) + (0.5*0.19)*(0.2) + (0.05*0.01+0.05*0.19)*(0) + (0.05*0.01)*0 = 0.123
P(1st U Joint 1st) = 0.8466+0.123 = 0.9696
P(2nd) = (0.9*0.8)*(0) + (0.9*0.19+0.05*0.8)*(0) + (0.9*0.01+0.05*0.8)*((0.2+0.2+0.6+0.2)/2) + (0.5*0.19)*(0.6+0.2) + (0.05*0.01+0.05*0.19)*(0) + (0.05*0.01)*0 = 0.1..

Note: 
A:
P(6 is enough for 1st) = P(B_points <= 6 | A beat B)*P(P(C_points <= 6 | A beat C)|P(B_points <= 6 | A beat B)) = 1*((0.6+0.2+0.2)/(0.6+0.2+0.2)) = 1
P(4 is enough for 1st) = P(B_points <= 4 | A beat B)*P(C_points <= 4 | A Drew C) U P(B_points <=4 | A drew B)*P(C_points <=4 | A beat C) = 1*1 + 1*1 - (1*1?) = 1

P(3 is enough for 1st) = P(B_points <= 3 | A lost to B) N P(C_points <= 3 | A beat C) U P(B_points <=3 | A beat B) N P(C_points <=3 | A lost to C)
= P(B_points <= 3 | A lost to B) * P(P(C_points <= 3 | A beat C)|P(B_points <= 3 | A lost to B)) U P(B_points <=3 | A beat B)*P(P(C_points <=3 | A lost to C)|P(B_points <=3 | A beat B))
= 0.2*(0.2/0.2) U 1*0.6 = 0.2+0.6 - P(B_points <= 3 | A lost to B) N P(C_points <= 3 | A beat C) * P(P(B_points <=3 | A beat B) N P(C_points <=3 | A lost to C)|P(B_points <= 3 | A lost to B) N P(C_points <= 3 | A beat C))
= 0.8 - 0.2 *0 = 0.8


A Beats B -> A Loses C -> C beats B -> 0.2
A Beats B -> A Loses C -> C draws B -> 0.2
A Beats B -> A Loses C -> C loses B -> 0.6
A loses B -> A beats C -> B beats C -> 0.6
A loses B -> A beats C -> B draws C -> 0.2
A loses B -> A beats C -> B loses C -> 0.2
P(3 is enough for 1st) = (0.6+0.2)/(0.2+0.2+0.6+0.6+0.2+0.2) = 0.8/2 = 0.4

P(2 is enough for 1st) = P(B_points <= 2 | A drew B)*P(C_points <=2 | A drew C) = (0.2+0.2)*(0.2+0.6) = 0.4*0.8 = 0.32  #before calc, obvious the only way this happens is if B draws C, for Prob of 0.2 
= P(B_points <= 2 | A drew B) * P(P(C_points <=2 | A drew C)|(P(B_points <= 2 | A drew B)) = 0.4 * (0.2/(0.2+0.2)) = 0.4*0.5 = 0.2

A draws B -> A draws C -> C beats B -> 0.2
A draws B -> A draws C -> C draws B -> 0.2
A draws B -> A draws C -> C loses B -> 0.6
P(2 is enough for 1st) = 0.2/(0.2+0.2+0.6) = 0.2


P(1 is enough for 1st) = P(B_points <=1|A lost to B)N P(C_points <=1 | A drew C) U P(B_points <=1| A drew B)*P(C_points <=1 | A lost to C)
= P(B_points <=1|A lost to B)*P(P(C_points <=1 | A drew C)|P(B_points <=1|A lost to B)) U P(B_points <=1| A drew B)*P(P(C_points <=1 | A lost to C)|P(B_points <=1| A drew B))

A draws B -> A Loses C -> C beats B -> 0.2
A draws B -> A Loses C -> C draws B -> 0.2
A draws B -> A Loses C -> C loses B -> 0.6
A loses B -> A draws C -> B beats C -> 0.6
A loses B -> A draws C -> B draws C -> 0.2
A loses B -> A draws C -> B loses C -> 0.2

P(1 is enough for 1st) = 0/2 = 0

P(3 is enough for 2nd AND 3 is not enough for 1st) = P(B_points > 3 | A lost to B) U P(C_points >3 | A beat C) N NOT(P(B_points > 3 | A lost to B) N P(C_points >3 | A beat C))  U   P(B_points > 3 | A beat B) U P(C_points >3 | A lost to C) N NOT(P(B_points > 3 | A beat B) N P(C_points >3 | A lost to C))

A Beats B -> A Loses C -> C beats B -> 0.2
A Beats B -> A Loses C -> C draws B -> 0.2
A Beats B -> A Loses C -> C loses B -> 0.6
A loses B -> A beats C -> B beats C -> 0.6
A loses B -> A beats C -> B draws C -> 0.2
A loses B -> A beats C -> B loses C -> 0.2
P(3 is enough for 2nd AND 3 is not enough for 1st) = (0.2+0.2+0.6+0.2)/2 = (1.2)/2 = 0.6


P(2 is enough for 2nd AND 2 is not enough for 1st)
A draws B -> A draws C -> C beats B -> 0.2
A draws B -> A draws C -> C draws B -> 0.2
A draws B -> A draws C -> C loses B -> 0.6
P(2 is enough for 2nd AND 2 is not enough for 1st) = (0.2+0.6)/1 = 0.8

P(1 is enough for 2nd AND 1 is not enough for 1st)
A draws B -> A Loses C -> C beats B -> 0.2
A draws B -> A Loses C -> C draws B -> 0.2
A draws B -> A Loses C -> C loses B -> 0.6
A loses B -> A draws C -> B beats C -> 0.6
A loses B -> A draws C -> B draws C -> 0.2
A loses B -> A draws C -> B loses C -> 0.2
P(1 is enough for 2nd AND 1 is not enough for 1st) = (0.2+0.6)/2 = 0.4

P(A=1st) = P(A=6 points N B < 6 N C < 6) U P(A=4 points N B < 4 N C < 4) U ...
= P(A=WW N (B=WD U B=WL U B=DW U B=DD U B=DL U B=LW  B=LD U B=LL) N (C=WD U C=WL U C=DW U C=DD U C=DL U C=LW  C=LD U C=LL)) U ...
= P(A=WW N (B=LW U B=LD U B=LL) N (C=LW  C=LD U C=LL))
= P(A=WW N B=LW N C=LL) U P(A=WW N B=LD N C=LD) U P(A=WW N B=LL N C=LW) 
P((A=WD U A=DW) N (B=DD U B=DL U B=LW U B=LD U B=LL) N (C=DD U C=DL U C=LW U C=LD U C=LL))

probs of winning:
A vs B (x1,y1,z1)
A vs C (x2,y2,z2)
B vs C (x3,y3,z3)

|A=WL
A vs B (x1,0,0)
A vs C (0,0,z2)
B vs C (x3,y3,z3)

|A=WL N B=LW
A vs B (x1,0,0)
A vs C (0,0,z2)
B vs C (x3,0,0)

|A=LW
A vs B (0,0,z1)
A vs C (x2,0,0)
B vs C (x3,y3,z3)

|A=LW N B=WL
A vs B (0,0,z1)
A vs C (x2,0,0)
B vs C (0,0,z3)

P(j1st) = P(A=6 N B=6 N C=6) U P(A=4 N B=4 N C=4) U P(A=3 N B=3 N C=3) U ...
P((A=WL U A=LW) N (B=LW U B=WL) N (C=LW U C=WL)) U ...
(simplified)
= P(A=WL N B=LW N C=WL) U P(A=LW N B=WL N C=LW)
= P(A=WL) * P(B=LW N C=WL | A=WL) + P(A=LW)*P(B=WL N C=LW | A=LW)
= P(A=WL) * P(B=LW | A=WL)*P(C=WL | A=WL N B=LW) + P(A=LW)*P(B=WL | A=LW)*P(C=LW | B=WL N A=LW)
=x1*z2*(x1/(x1+0+0)*x3/(x3+y3+z3))*(z2/(z2+0+0)*x3/(x3+0+0)) + z1*x2*(z1/(z1+0+0)*z3/(x3+y3+z3))*(x2/(x2+0+0)*z3/(z3+0+0))
=x1*z2*x3 + z1*x2*z3

4 teams, 3 games total
P(A=1st) = P(A=9 N B < 9 N C < 9 N D < 9) U P(A=7 N B < 7 N C < 7 N D < 7) U ...
= P(A=WWW N (B=LWW U B=LWD U B=LWL U B=LDW U B=LDD U B=LDL U B=LLW U B=LLD U B=LLL) N (C=LWW U C=LWD U C=LWL U C=LDW U C=LDD U C=LDL U C=LLW U C=LLD U C=LLL) N (D=LWW U D=LWD U D=LWL U D=LDW U D=LDD U D=LDL U D=LLW U D=LLD U D=LLL)) U ...
= P(A=WWW N B=LWW N C=LLW N D=LLL) U P(A=WWW N B=LWW N C=LLD N D=LLD) U P(A=WWW N B=LWW N C=LLL N D=LLW) U P(A=WWW N B=LWD N C=LLW N D=LDL) U ...


algorithm for generating groups of results:
let W=2, D=1, L=0
for a 2 team game:
A vs B
A=x1
B=x2

XOR:
x1 <=> x2

B < A
(-x1+2)^2+(-x1+2) < x1^2+x1
max x1 = 2
so we are done when x1=2

for a 3 team game:
A vs B
A vs C
B vs C

A = x1,y1
B = x2,y2
C = x3,y3

XOR:
x1 <=> x2
y1 <=> x3
y2 <=> y3

(-x1+2)^2 + (-x1+2) + y2^2 +y2 < x1^2+x1+y1^2+y1 #B

(-y1+2)^2 + (-y1+2) + (-y2+2)^2 + (-y2+2) < x1^2+x1+y1^2+y1 #C

max for C when A is WD:
(-y2+2)^2 + (-y2+2) < 6

max y2 is 1

next, considering B:
y2^2 + y2 < 8
the max is y2 is 2, so no changes will be made to a list

as max y2 is 1, possibles are y2 is 0 or 1
as x1 and y1 are given, y2 is now worked out, we can calculate the remaining x2,x3,y3

for a 4 team game:
A vs B
A vs C
A vs D
B vs C
B vs D
C vs D

A = x1,y1,z1
B = x2,y2,z2
C = x3,y3,z3
D = x4,y4,z4

None,None,None
None,None,None
None,None,None
None,None,None

0,1,2
0,3,4
1,3,5
2,4,5

#trim A

3,4
3,5
4,5
#reduce nums:
0,1
0,2
1,2

XOR table
x1 <=> x2
y1 <=> x3
z1 <=> x4
y2 <=> y3
z2 <=> y4
z3 <=> z4

to calculate XOR table, identify the 'centre' for each team.
A is x1, B is y2, C is z3, D doesn't have one as there is no '_4'
Using the following table:
A = x1,y1,z1
B = x2,y2,z2
C = x3,y3,z3
D = x4,y4,z4
apply this algorithm:
next, to find the XOR pair for the selected team's variable, go n steps back to the centre variable, and then go n+1 steps down from the center variable where 1 step left is +1 step, and 1 step down is +1 step. Do not go from the left of a centre variable to the centre variable
You will then be on that variables partner. You COULD perform a similar algorithm, but in reverse (i.e identify distance to column centre, then go right from there)

(-x1+2)^2 -x1+2 +y2^2+y2 + z2^2 + z2 < x1^2 + x1 + y1^2 + y1 + z1^2 + z1 #B
(-y1+2)^2 -y1+2 + (-y2+2)^2 -y2 + 2 + z3^2 <  x1^2 + x1 + y1^2 + y1 + z1^2 + z1 #C
(-z1+2)^2 -z1+2 + (-z2+2)^2 -z2+2 + (-z3+2)^2 (-z3+2) <  x1^2 + x1 + y1^2 + y1 + z1^2 + z1 #D

Given A is WDD:
(z2,z3) Max (0,2) inv: [0,2], nonInv:[]
(y2,z3) Max (2,2) inv: [2], nonInv: [2]
(y2,z2) Max (2,1) inv: [], nonInv: [1,2]

(z2,z3) banned (0,1), (0,0) inv: [(0,1), (0,0)], nonInv:[] (< 8)
(*y2,z3) banned (0,2),(1,2) inv: [(0),(1)], nonInv: [(2),(2)] (< 8)
(*y2,*z2) banned (2,2) inv: [], nonInv: [(2,2)] (< 12)

trying to find:
(y2,z2,z3)
()

'''
