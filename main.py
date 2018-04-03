import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import random
import math
import datetime
import time
import glob
import pandas as pd

#read in data
STOCKS = [
        'AAPL',
        'AMZN',
        'CAT',
        'ETE',
        'FB',
        'GOOGL',
        'JPM',
        'KMI',
        'LMT',
        'MSFT',
        'NFLX',
        'NKTR',
        'OKE'
        ]
allFiles = []
for s in STOCKS:
    allFiles.append('stock_data/Stocks/'+s.lower()+'.us.txt')

cutoff_date = pd.to_datetime('2012-06-04')
datasets = dict()
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    
    #process the data to get only weeks with data for monday and friday after cutoff
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df[(df['Date'] >= cutoff_date)]
    df = df[(df['Date'].dt.weekday == 0) | (df['Date'].dt.weekday == 4)]
    df['weekday'] = df['Date'].dt.weekday
    df['prev_day'] = df['weekday'].shift()
    df['next_day'] = df['weekday'].shift(-1)
    df = df[((df['weekday'] != df['prev_day']) & (df['weekday'] == 4)) | ((df['weekday'] != df['next_day']) & (df['weekday'] == 0))]
    
    datasets[file_.split('/')[2].split('.')[0].upper()] = df


## GENETIC ALGORITHM

NUM_STOCKS = len(STOCKS)
MUTATE_RATE = 0.1
#keep pop size a multiple of 4
POP_SIZE = 100
NUM_GENS = 240
CROSS_RATE = 0.5
GENOME_SIZE = 2*NUM_STOCKS
MUTATE_BOUND_PROBS = [-0.5,0.50001]
MUTATE_BOUND_AMT = [-5,6]
PROB_WINDOW = [0,1]
AMT_WINDOW = [0,100]
NUM_ELITES = 1

def get_fitness(genome):
    return 0
    
def select(genomes):
    random.shuffle(genomes)
    fittest = []
    for i in range(int(POP_SIZE/2)):
        p1 = genomes[i]
        p2 = genomes[2*i]
        if get_fitness(p1) >= get_fitness(p2):
            fittest.append(p1)
        else:
            fittest.append(p2)
    return fittest
        
    
def crossover(fittest):
    offspring = []
    elite = sorted(fittest.copy(), key=get_fitness, reverse=True)[0:NUM_ELITES]
    while len(offspring) < POP_SIZE:
        random.shuffle(fittest)
        females = fittest[0:int(len(fittest)/2)]
        males = fittest[int(len(fittest)/2):len(fittest)]
        for i in range(len(males)):
            mom = females[i]
            dad = males[i]
            if np.random.rand() < CROSS_RATE:
                child = []
                for i in range(len(dad)):
                    if np.random.rand() < 0.5:
                        child.append(dad[i])
                    else:
                        child.append(mom[i])
                offspring.append(child)
            else:
                #clone both parents
                offspring.append(dad)
                offspring.append(mom)
                
    return [*elite,*random.sample(offspring,POP_SIZE-NUM_ELITES)]
                
def get_mutation_rate(gen_num):
    #use 0.03 for ~80-120
    return MUTATE_RATE*math.exp(-(MUTATE_RATE/2)*gen_num)    

def mutate(offspring, gen_num):
    for child in offspring:
        for gene in range(GENOME_SIZE):
            if gene < NUM_STOCKS:
                mutation_amt = np.random.uniform(*MUTATE_BOUND_PROBS);
                in_range = PROB_WINDOW[0] < child[gene] + mutation_amt < PROB_WINDOW[1]
            else:
                mutation_amt = np.random.randint(*MUTATE_BOUND_AMT);
                in_range = AMT_WINDOW[0] < child[gene] + mutation_amt < AMT_WINDOW[1]
            if np.random.rand() < get_mutation_rate(gen_num) and in_range:
                child[gene] = child[gene] + mutation_amt
    
#initialize population all at starting point
investors = []
for i in range(POP_SIZE):
    investor = dict()
    investor['value'] = 10000
    investor['strategy'] = [*[random.uniform(0, 1) for _ in range(NUM_STOCKS)],*[random.randint(1, 10) for _ in range(NUM_STOCKS)]]
    investors.append(investor)


rounds = []
for i in range(NUM_GENS):
    week_data = dict()
    for s in STOCKS:
        week_data[s] = datasets[s].iloc[(2*i):((2*i)+2),[0,4]]
    
    #TODO invest at beginning of week here
    #TODO in select() check friday values for fitness
    top = select(investors)
    offspring = crossover(top)
    mutate(offspring, i)
    genomes = offspring
    top = sorted(investors.copy(), key=get_fitness, reverse=True)
    if i % 10 == 0:
        print(get_fitness(top[0]),top[0])
    
    
    #stop condition
#    if get_fitness(top[0]) > 8.3728:
#        break   



















#visualization

def fun(x, y):
  return math.exp(-(math.cos(x/2)+math.cos(y/4))) + math.cos(x)

delta = 0.05
x = np.arange(0, 20, delta)
y = np.arange(0, 50, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

norm = cm.colors.Normalize(vmax=abs(Z).max(), vmin=-abs(Z).max())
cmap = cm.PRGn

levels = np.arange(0, 10, 2)

filenames = []
for i,points in enumerate(rounds):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contourf(X, Y, Z, levels,
                cmap=cm.get_cmap(cmap, len(levels)-1),
                norm=norm)
    ax.autoscale(False) # To avoid that the scatter changes limits
    
    ax.scatter(*zip(*points), color='red')
    #plt.show()
    if i % 2 == 0:
        file = 'plot'+ str(i) +'.png'
        fig.savefig(file)
        filenames.append(file)
    plt.close(fig)


import imageio
images = []
for filename in filenames:
    images.append(imageio.imread(filename))

st = datetime.datetime.now().strftime('%Y%m%d%H%M')
imageio.mimsave('climber_'+ st +'.gif', images)
