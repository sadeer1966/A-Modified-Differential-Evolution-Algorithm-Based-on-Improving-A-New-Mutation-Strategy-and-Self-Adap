# from asyncio.windows_events import NULL
import numpy as np
import math as ma
import time as tm


 

def DEalgorithm(PopSize, fn, N, MaxItr, lower = -30, upper = 30):
    besteval = float('inf')
    worse_eval = float('-inf')    
    if fn == 'sphare':
        func = lambda x: sphare(x)
    elif fn == 'Rastrigin':
        func = lambda x: Rastrigin(x)
    elif fn == 'Ackley':
        func = lambda x: Ackley(x)
    elif fn == 'Rosenbrock':
        func = lambda x: Rosenbrock(x)
    elif fn == 'Beale':
        func = lambda x: Beale(x)
    elif fn == 'Goldstein_Price':
        func = lambda x: Goldstein_Price(x)
    elif fn == 'bohachevsky':
        func = lambda x: bohachevsky(x)
    elif fn == 'trid':
        func = lambda x: trid(x)
    elif fn == 'booth':
        func = lambda x: booth(x)
    elif fn == 'matyas':
        func = lambda x: matyas(x)
    elif fn == 'zakharov':
        func = lambda x: zakharov(x)
    elif fn == 'six_hump':
        func = lambda x: six_hump(x)
    elif fn == 'Easom':
        func = lambda x: Easom(x)
    elif fn == 'Himmelblau':
        func = lambda x: Himmelblau(x)
    elif fn == 'Bird':
        func = lambda x: Bird(x)
    elif fn == 'McCormick':
        func = lambda x: McCormick(x)
    elif fn == 'Three_Hump':
        func = lambda x: Three_Hump(x)
    elif fn == 'xinSheYangN4':
        func = lambda x: xinSheYangN4(x)
    elif fn == 'Salomon':
        func = lambda x: Salomon(x)
    elif fn == 'adjiman':
        func = lambda x: adjiman(x)
    elif fn == 'Dixon_Price':
        func = lambda x: Dixon_Price(x)
    elif fn == 'drop_wave':
        func = lambda x: drop_wave(x)
    elif fn == 'eggholder':
        func = lambda x: eggholder(x)
    elif fn == 'Shubert':
        func = lambda x: Shubert(x)
    # elif fn == 'Holder_table':
    #     func = lambda x: Holder_table(x)
    elif fn == 'griewank':
        func = lambda x: griewank(x)
    elif fn == 'schaffer':
        func = lambda x: schaffer(x)
    elif fn == 'bukin':
        func = lambda x: bukin(x)
    elif fn == 'CROSS_IN_TRAY':
        func = lambda x: CROSS_IN_TRAY(x)
    elif fn == 'Deckkers_Aarts':
        func = lambda x: Deckkers_Aarts(x)
    elif fn == 'keane':
        func = lambda x: keane(x)
    elif fn == 'colville':
        func = lambda x: colville(x)



        
    
           

    
    # Initialization
    start = tm.time()
    population = {}
    for i in range(PopSize):
        population[i] = np.random.uniform(lower, upper, N)
        SolEval = func(population[i])
        if SolEval < besteval:
            bestsol = population[i]
            besteval = SolEval
        if SolEval > worse_eval:
            worst_sol = population[i]
            worse_eval = SolEval
    for k in range(MaxItr):
        for i in range(PopSize):
            # Select two random solutions from the population that isn't i
            rnd = [j for j in range(PopSize) if j != i]
            rnd = np.random.choice(rnd,len(rnd), replace = False)        
            sol2 = population[rnd[1]]
            # sol2 = bestsol
            sol3 = population[rnd[2]]
            # Mutation
            
            # newsol = population[i] + np.random.uniform(beta_min, beta_max) * (sol2 - sol3)
            F = np.random.uniform(0, 1)
            # F = np.random.normal(0, 3)
            # newsol = current + F * (best - current) + F * (best - random)
            newsol = population[i] + F * (bestsol - population[i]) + F * (bestsol - sol2)
            # newsol = population[i] + F * (bestsol - population[i]) + F * (sol2 - sol3)
            # Crossover
            crossover_sol = np.zeros(N)
            if np.mod(k, 2) == 0:
                pCR = np.random.uniform(0, 0.25)
            else:
                pCR = np.random.uniform(0.25, 0.5)
                
            for j in range(N):
                # pCR = np.random.uniform()
                if np.random.uniform() <= pCR:
                    crossover_sol[j] = population[i][j]
                else:
                    crossover_sol[j] = newsol[j]
            
            # Selction
            # The evaluation of the current solution
            current_eval = func(population[i])
            # The evalution of the crossover solution
            crossover_sol_eval = func(crossover_sol)
            # The evalution of the mutated solution
            # newsol_eval = func(newsol)
            if crossover_sol_eval < current_eval:
                population[i] = crossover_sol
                current_eval = crossover_sol_eval
            # elif newsol_eval < current_eval:
            #     population[i] = newsol
            #     current_eval = newsol_eval
            
            if current_eval < besteval:
                bestsol = population[i]
                besteval = current_eval
    end = tm.time() - start
    return besteval, bestsol, end

def cosd(n):
    return np.cos(n * np.pi /180)

def sind(n):
    return np.sin(n * np.pi /180)


# 1
def sphare(x):    
    return sum(x ** 2)
# 2
def Rastrigin(x):
    N = len(x)
    result = 10 * N
    for i in x:
        result += i ** 2 - 10 * cosd(2 * ma.pi * i)
    return result
# 3
def Ackley(x):
    return -20 * np.exp(-0.2 * np.sqrt(1/len(x) * sum([i ** 2 for i in x]))) - np.exp(1/len(x) *\
        sum([np.cos(i * 2 * np.pi) for i in x])) + 20 + np.exp(1)
    # return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2)))

# 4
def Rosenbrock(x):
    result = 0
    for i in range(len(x)-1):
        result += 100 * (x[i+1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2
    return result

# 5
def Beale(x):
    return (1.5 - x[0] + x[0] * x[1]) ** 2 + (2.25 - x[0] + x[0] * (x[1] ** 2)) ** 2 + (2.625 - x[0] + x[0] * (x[1]**3)) ** 2

# 6
def Goldstein_Price(x):
    return (1+(x[0]+x[1]+1)**2*(19-14*x[0]+3*x[0]**2-14*x[1]+6*x[0]*x[1]+3*x[1]**2))*(30+(2*x[0]-3*x[1])**2*(18-32*x[0]+12*x[0]**2
    +48*x[1]-36*x[0]*x[1]+27*x[1]**2))

# 7
def bohachevsky(x):
    return x[0]**2 + 2*x[1]**2 - 0.3*cosd(3*ma.pi*x[0])- 0.4*cosd(4*ma.pi*x[1]) + 0.7

# 8 
def trid(x):
    return sum([(x[i]-1)**2 for i in range(len(x))]) - sum([x[i] * x[i+1] for i in range(len(x)-1)])

# 9
def booth(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

# 10
def matyas(x):
    return 0.26*(x[0]**2 + x[1]**2) - 0.48*x[0]*x[1]

# 11
def zakharov(x):
    return (x[0]*2 + x[1]**2) + (0.5*x[0] + x[1])**2 + (0.5*x[0] + x[1])**4

# 12
def six_hump(x):
    return (4 - 2.1*x[0]**2 + x[0]**4/3)*x[0]**2 + x[0]*x[1] + (-4 + 4*x[1]**2)*x[1]**2

# 13
def Easom(x):
    return -cosd(x[0]) * cosd(x[1]) * np.exp(-((x[0] - np.pi)**2 + x[1] - np.pi)**2)

# 14
def Himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 -7)**2

# 15
def Bird(x):
    return sind(x[0])*(np.exp(1-cosd(x[1]))**2)+cosd(x[1])*(np.exp(1-sind(x[0]))**2)+(x[0]-x[1])**2

# 16
def McCormick(x):
    return (sind(x[0]+x[1])+(x[0]-x[1])**2-1.5*x[0]+2.5*x[1]+1)

# 17
def Three_Hump(x):
    return (2*(x[0]**2)-1.05*(x[0]**4)+((x[0]**6)/6)+(x[0]*x[1])+(x[1]**2))

# 18
def xinSheYangN4(x):
    return (sind(x[0])*sind(x[0])-np.exp(-(x[0]**2+x[1]**2)))*np.exp(-((pow(sind(np.sqrt(abs(x[0]))),2))))+(sind(x[1])*sind(x[1])-
    np.exp(-(x[0]**2+x[1]**2)))*np.exp(-pow(sind(np.sqrt(abs(x[1]))),2))

# 19
def Salomon(x):
    return 1 - cosd(2 * np.pi * np.sqrt(sphare(x))) + 0.1 * np.sqrt(sphare(x))

# 20
def adjiman(x):
    return (cosd(x[0])*sind(x[1]) - (x[0]/(x[1]**2+1)))

# 21
def Dixon_Price(x):
    return (x[0]-1)**2 + sum(i*(2*x[i]**2-x[i-1])**2 for i in range(1, len(x)))

# 22
def drop_wave(x):
    return -(1+cosd(12* np.sqrt(x[0] ** 2+x[1] ** 2)))/(0.5*(x[0] ** 2+x[1] ** 2)+2)

# 23
def eggholder(x):
    return -(x[1] + 47) * sind(np.sqrt(abs(x[1] + x[0]/2 + 47))) - x[0] * sind(np.sqrt(abs(x[0] - (x[1] +47))))

# 24
def Shubert(x):
    return sum([i + 1 * cosd((i + 2) * x[0] + i + 1) for i in range(5)]) * \
        sum([i + 1 * cosd((i + 2) * x[1] + i + 1) for i in range(5)])

# # 25
# def Holder_table(x):
#     return - np.abs(np.sin(x[0]) * np.cos(x[1]) * np.exp(np.abs(1 - (np.sqrt(x[0]**2 + x[1]**2)/np.pi))))

# 26
# def griewank(x):
#    return 1 - np.prod(cosd(x[0] / np.sqrt(1. + np.arange(len(x[0]))))) + sum(x**2)/4000

def griewank(x):
    ii = range(len(x)+1)[1:]
    return (sum(x ** 2 /4000) - np.prod(np.cos(x/np.sqrt(ii))) + 1)


# 27
def schaffer(x):
    return 0.5 + (sind((x[0]**2 + x[1]**2)**2)**2) - 0.5/(1 + 0.001*(x[0]**2 + x[1]**2))**2 

# 28
def bukin(x):
    return 100 * np.abs(x[1] - 0.01 * x[0]**2)**0.5 + 0.01 * np.abs(x[0] + 10) 


# 29 !!!!!
def CROSS_IN_TRAY(x):
    return -0.0001 * (np.fabs(np.sin(x[0]) * np.sin(x[1]) * np.exp(np.fabs(100 - np.sqrt(x[0]**2 + x[1]**2)/np.pi))) + 1) ** 0.1 

# 30
def Deckkers_Aarts(x):
    return (10**5) * (x[0] ** 2) + (x[1] ** 2) - (x[0] ** 2 + x[1] ** 2) ** 2 + (10**(-5)) * (x[0]**2 + x[1]**2) ** 4

# 31
def keane(x):
   return -np.sin(x[0] - x[1]) **2 * np.sin(x[0] + x[1]) **2/ np.sqrt(x[0] ** 2 + x[1] ** 2) 

# 32
def colville (x):
   return 100 * (x[0]** 2 - x[1])** 2 + (x[0] - 1)** 2 + (x[2] - 1)** 2 + 90 * (x[2]** 2 - x[3])** 2 + 10.1 * ((x[1]-1)**2 + (x[3] - 1)**2) + 19.8 * (x[1] - 1) * (x[3] - 1)
