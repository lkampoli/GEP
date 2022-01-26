import geppy as gep
from deap import creator, base, tools

import numpy as np
import random

s = 10
random.seed(s)
np.random.seed(s)

LINEAR_SCALING = False

I_train = np.genfromtxt('lamda_train.dat')
T_train = np.genfromtxt('T_train.dat').reshape([-1, 3, 6])
bij = np.genfromtxt('bij_train.dat')
I1=I_train[:,0]
I2=I_train[:,1]
I3=I_train[:,2]
T1=T_train[:,0,:]
T2=T_train[:,1,:]
T3=T_train[:,2,:]
size=np.shape(I1)[0]

def protected_div(a, b):
    if np.isscalar(b):
        if abs(b) < 1e-6:
            b = 1
        else:
            b[abs(b) < 1e-6] = 1
    return a / b

import operator

pset = gep.PrimitiveSet('Main', input_names=['I1','I2','I3','T1','T2','T3'])
pset.add_function(operator.add, 2)
pset.add_function(operator.sub, 2)
pset.add_function(operator.mul, 2)
pset.add_function(protected_div, 2)
pset.add_ephemeral_terminal(name='enc', gen=lambda: random.uniform(-5, 5)) # each ENC is a random integer within [-10, 10]

from deap import creator, base, tools

creator.create("FitnessMin", base.Fitness, weights=(-1,)) # to minimize the objective (fitness)
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin, a=float, b=float)

h = 7 # head length
n_genes = 2 # number of genes in a chromosome

toolbox = gep.Toolbox()
toolbox.register('gene_gen', gep.Gene, pset=pset, head_length=h)
toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes, linker=operator.add)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
compile utility: which translates an individual into an executable function (Lambda)

toolbox.register('compile', gep.compile_, pset=pset)

def evaluate(individual):
    """Evalute the fitness of an individual: MSE (mean squared error)"""
    func = toolbox.compile(individual)
    yp = np.array(list(map(func,I1,I2,I3,T1,T2,T3))) # predictions with the GEP model
    a=0
    b=0
    c=0

    for i in range(size):
        Yp=np.zeros(6)
        #print(np.shape(yp))
        if np.shape(yp)!=(size,6):
            # print(yp,np.shape(yp[i]))
            Yp=yp[i]np.array([1,1,1,0,0,0])
        else:
            Yp=yp[i]
            #print(Yp)
            #print("------------")
            #print(np.shape(yp[i]))
            #print("**********")
            Ri=np.array([[bij[i,0],bij[i,1],bij[i,2]],
            [bij[i,1],bij[i,3],bij[i,4]],
            [bij[i,2],bij[i,4],bij[i,5]]])
            Rp_i=np.array([[Yp[0],Yp[1],Yp[2]],
            [Yp[1],Yp[3],Yp[4]],
            [Yp[2],Yp[4],Yp[5]]])
            a=a+np.tensordot(Rp_i,Ri)
            b=a+np.tensordot(Ri,Ri.T)
            c=c+np.tensordot(Rp_i,Rp_i.T)
            #print(a,b,c)
            #print (a,b,c,a/(bc))

        return a/(bc),


def evaluate_linear_scaling(individual):
    """Evaluate the fitness of an individual with linearly scaled MSE.
    Get a and b by minimizing (a*Yp + b - Y)"""
    func = toolbox.compile(individual)
    Yp = np.array(list(map(func,I1,I2,I3,T1,T2,T3))) # predictions with the GEP model

    # special cases: (1) individual has only a terminal 
    # (2) individual returns the same value for all test cases, like 'x - x + 10'. np.linalg.lstsq will fail in such cases.

    if isinstance(Yp, np.ndarray):
        Q = np.hstack((np.reshape(Yp, (-1, 1)), np.ones((len(Yp), 1))))
        (individual.a, individual.b), residuals, _, _ = np.linalg.lstsq(Q, bij, rcond=None)   
        # residuals is the sum of squared errors
        if residuals.size > 0:
            return residuals[0] / len(Y),   # MSE

        # for the above special cases, the optimal linear scaling is just the mean of true target values
        individual.a = 0
        individual.b = np.mean(Y)

        return np.mean((Y - individual.b) ** 2),


if LINEAR_SCALING:
    toolbox.register('evaluate', evaluate_linear_scaling)
else:
    toolbox.register('evaluate', evaluate)

toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=0.05, pb=1)
toolbox.register('mut_invert', gep.invert, pb=0.1)
toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.1)
toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.1)
toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.1)
toolbox.register('cx_1p', gep.crossover_one_point, pb=0.4)
toolbox.register('cx_2p', gep.crossover_two_point, pb=0.2)
toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)
toolbox.register('mut_ephemeral', gep.mutate_uniform_ephemeral, ind_pb='1p') # 1p: expected one point mutation in an individual
toolbox.pbs['mut_ephemeral'] = 1 # we can also give the probability via the pbs property

stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

n_pop = 10
n_gen = 200

pop = toolbox.population(n=n_pop)
hof = tools.HallOfFame(3) # only record the best three individuals ever found in all generations

pop, log = gep.gep_simple(pop, toolbox, n_generations=n_gen, n_elites=1, stats=stats, hall_of_fame=hof, verbose=True)

print(hof[0])

for i in range(3):
    ind = hof[i]
    symplified_model = gep.simplify(ind)

if LINEAR_SCALING:
    symplified_model = ind.a * symplified_model + ind.b
    print('Symplified best individual {}: '.format(i))
    print(symplified_model)

rename_labels = {'add': '+', 'sub': '-', 'mul': '*', 'protected_div': '/'}
best_ind = hof[0]
gep.export_expression_tree(best_ind, rename_labels, 'tree.png')

from IPython.display import Image
Image(filename='tree.png')`
