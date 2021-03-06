#!/usr/bin/env python
# coding: utf-8

import os
import geppy as gep
from deap import creator, base, tools
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

import operator 
import math
import datetime

from numba import jit

# for reproduction
s = 0
random.seed(s)
np.random.seed(s)

# doublecheck the data is there
print(os.listdir("../../data/k_ci/processes_VT/"))

# read in the data to pandas
#data = pd.read_csv("../data/k_ci/processes_VT/VT_RATES-N2-N2-vt_down_T.csv")

from numpy import genfromtxt
T = genfromtxt("../../data/k_ci/processes_VT/Temperatures.csv", delimiter=',')
k = genfromtxt("../../data/k_ci/processes_VT/VT_RATES-N2-N2-vt_down.csv", delimiter=',')
#k = genfromtxt("../../data/k_ci/processes_VT/VT_RATES-NO-NO-vt_down.csv", delimiter=',')
#k = genfromtxt("../../data/k_ci/processes_VT/VT_RATES-O2-O_-vt_up__.csv", delimiter=',')

print("T = ",T.shape)
print("k = ",k.shape)

#data = np.hstack((T,x))
#data = np.concatenate((T, x), axis=None)
#data = np.loadtxt("../../data/k_ci/processes_VT/VT_RATES-N2-N2-vt_up_T.txt")

#print(data.describe())

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#x_train, x_test, y_train, y_test = train_test_split(T, k, train_size=0.75, test_size=0.25, random_state=666)
#
#sc_x = StandardScaler()
#sc_y = StandardScaler()
#
## fit scaler
#sc_x.fit(x_train.reshape(-1,1))
#
## transform training dataset
#x_train = sc_x.transform(x_train.reshape(-1,1))
#
## transform test dataset
#x_test = sc_x.transform(x_test.reshape(-1,1))
#
## fit scaler on training dataset
#sc_y.fit(y_train.reshape(-1,1))
#
## transform training dataset
#y_train = sc_y.transform(y_train.reshape(-1,1))
#
## transform test dataset
#y_test = sc_y.transform(y_test.reshape(-1,1))
#
#trainT   = x_train
#holdoutT = x_test
#
#trainK   = y_train
#holdoutK = y_test

# Split my data into Train and Test chunks, 20/80
msk     = np.random.rand(len(T)) < 0.8
#msk     = np.random.rand(len(data)) < 0.8
#train   = data[msk]
#holdout = data[~msk]

trainT   = T[msk]
holdoutT = T[~msk]

trainK   = k[msk]
holdoutK = k[~msk]

T = trainT
k = trainK[:,0:1] # consider only the 1st rate coeff. ... how to do multi-target regression with GEP?

print(T.shape)
print(k.shape)

#T = train[:,0:1].reshape(-1,1)
#k = train[:,1:2].reshape(-1,1)

# check the number of records we'll validate our MSE with
#print(holdout.describe())

# check the number of records we'll train our algorithm with
#print(train.describe())

# NOTE: I'm only feeding in the TRAIN values to the algorithms. 
# Later I will independently check the MSE myself using a holdout test dataset

Y = k # this is our target, now mapped to Y

# Creating the primitives set
# The first step in GEP (or GP as well) is to specify the primitive set, 
# which contains the elementary building blocks to formulate the model. 
# For this problem, we have:
# + function set: the standard arithmetic operators:
#   addition (+), subtraction (-), multiplication (*), and division (/).
# + terminal set: only the single input 'x' and random numerical 
#   constants (RNC).
# 
# NOTE:
# 
# - We define a *protected division* to avoid dividing by zero.
# - Even there may be multiple RNCs in the model, we only need 
#   to call `PrimitiveSet.add_rnc` once.

def protected_div(x1, x2):
    if abs(x2) < 1e-6:
       return 1
    return x1 / x2

def protected_div1(x1, x2):
    if np.isscalar(x2):
        if abs(x2) < 1e-6:
            x2 = float('nan')
    else:
        x2[x2<1e-6] = float('nan')
    return x1 / x2

def protected_exp(x1):
    try:
        temp = math.exp(x1)
        if temp < 10000:
            ans = temp
        else:
            ans = 0
    except OverflowError:
        ans = 0
    return ans

def root(x1):
    return np.sqrt(x1)

def protected_root(x1):
    try:
        temp = np.sqrt(x1)
        if temp >= 0:
            ans = temp
        else:
            ans = 0
    except OverflowError:
        ans = 0
    return ans

def epow(x1):
    return np.exp(x1)

def power(x1,x2):
    return np.power(x1,x2)

def protected_pow(x1,x2):
    if abs(x2) < 1e-6:
        return 1
    if abs(x2) > 3:
        return x1
    print(x1,x2)
    return np.power(x1,x2)

def protected_log(x1):
    if x1 < 2.718:
        return 2.718
    return math.log(x1)

# Map our input data to the GEP variables
# Here we map the input data to the GEP algorithm:
# 
# We do that by listing the field names as "input_names".
# 
# In reviewing geppy code, in the file:  
#   geppy/geppy/core/symbol.py
#   
# we find how terminals in the gene are named correctly to match input
# data.
# 
# Oh - notice, I only mapped in below the input data columes, 
#      and not the TARGET "PE" which is sitting in var Y.
# I didn't notice where to map that - so suggest you force the target 
# variable to "Y" when reading in data.

pset = gep.PrimitiveSet('Main', input_names=['T'])

# Define the operators
# Here we define and pass the operators we'll construct our final 
# symbolic regression function with
pset.add_function(operator.add, 2)
pset.add_function(operator.sub, 2)
pset.add_function(operator.mul, 2)
#pset.add_function(operator.pow, 2)
#pset.add_function(np.exp, 1)
#pset.add_function(operator.truediv, 2)
pset.add_function(protected_div, 2)
#pset.add_function(protected_exp, 1)
#pset.add_function(protected_log, 1)
#pset.add_function(protected_root, 1)
#pset.add_function(protected_pow, 2)
pset.add_function(math.sin, 1) 
pset.add_function(math.cos, 1)
pset.add_function(math.tan, 1)
pset.add_rnc_terminal()
#pset.add_pow_terminal('T') #attention: Must the same as input in primitive set
#pset.add_pow_terminal('Y')
#pset.add_ephemeral_terminal(name='enc', gen=lambda: random.randint(-10, 10)) # each ENC is a random integer within [-10, 10]
#pset.add_ephemeral_terminal(name='enc', gen=lambda: 1)
#pset.add_constant_terminal(1.0)

# Create the individual and population
# Our objective is to **minimize** the MSE (mean squared error) for 
# data fitting.
# Define the indiviudal class, a subclass of *gep.Chromosome*
creator.create("FitnessMin", base.Fitness, weights=(-1,))  # to minimize the objective (fitness)
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)

# Register the individual and population creation operations
# In DEAP, it is recommended to register the operations used in evolution into a *toolbox* to make full use of DEAP functionality. The configuration of individuals in this problem is:
# + head length: 6
# + number of genes in each chromosome: 2
# + RNC array length: 8
# 
# Generally, more complicated problems require a larger head length and longer chromosomes formed with more genes. **The most important is that we should use the `GeneDc` class for genes to make use of the GEP-RNC algorithm.**

# Your Core Settings Defined Here

h = 7            # head length
n_genes = 2      # number of genes in a chromosome
r = h*2 + 1      # length of the RNC array
enable_ls = True # whether to apply the linear scaling technique
N_eval = 1

# **NOTE** Above you define the gene structure which sets out the maximum complexity of the symbolic regression

toolbox = gep.Toolbox()
#toolbox.register('rnc_gen', random.randint, a=-5, b=5) # each RNC is random integer within [0, 10]
#toolbox.register('rnc_gen', random.choice, np.arange(0.1,10.0,0.1))
toolbox.register('rnc_gen', random.uniform, a=-0.01, b=1)
toolbox.register('gene_gen', gep.GeneDc, pset=pset, head_length=h, rnc_gen=toolbox.rnc_gen, rnc_array_length=r)
toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes, linker=operator.add)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# compile utility: which translates an individual into an executable function (Lambda)
toolbox.register('compile', gep.compile_, pset=pset)

# Define the fitness evaluation function
# In DEAP, the single objective optimization problem is just a special case of 
# more general multiobjective ones. Since *geppy* is built on top of DEAP, it 
# conforms to this convention. **Even if the fitness only contains one measure, 
# keep in mind that DEAP stores it as an iterable.** 
# 
# Knowing that, you can understand why the evaluation function must return a 
# tuple value (even if it is a 1-tuple). That's also why we set:
# ``weights=(-1,)`` when creating the ``FitnessMax`` class.

#@jit
def evaluate(individual):
    """Evalute the fitness of an individual: MAE (mean absolute error)"""
    func = toolbox.compile(individual)
    
    # below call the individual as a function over the inputs
    Yp = np.array(list(map(func,T))) 
    
    # return the MSE as we are evaluating on it anyway,
    # then the stats are more fun to watch...
    return np.mean((Y - Yp) ** 2),


# [optional] Enable the linear scaling technique. It is hard for GP to determine 
# real constants, which are important in regression problems. Thus, we can 
# (implicitly) ask GP to evolve the shape (form) of the model and we help GP to 
# determine constans by applying the simple least squares method (LSM).

#@jit
def evaluate_ls(individual):
    """
    First apply linear scaling (ls) to the individual 
    and then evaluate its fitness: MSE (mean squared error)
    """
    func = toolbox.compile(individual)

    Yp = np.array(list(map(func,T))) 
    
    # special cases which cannot be handled by np.linalg.lstsq: (1) individual has only a terminal 
    #  (2) individual returns the same value for all test cases, like 'x - x + 10'. np.linalg.lstsq will fail in such cases.
    # That is, the predicated value for all the examples remains identical, which may happen in the evolution.
    if isinstance(Yp, np.ndarray):
        Q = np.hstack((np.reshape(Yp, (-1, 1)), np.ones((len(Yp), 1))))
        (individual.a, individual.b), residuals, _, _ = np.linalg.lstsq(Q, Y, rcond=-1)
        #(individual.a, individual.b), residuals, _, _ = np.linalg.lstsq(Q, Y)
        # residuals is the sum of squared errors
        if residuals.size > 0:
            return residuals[0] / len(Y),   # MSE
    
    # regarding the above special cases, the optimal linear scaling w.r.t LSM is just the mean of true target values
    individual.a = 0
    individual.b = np.mean(Y)
    return np.mean((Y - individual.b) ** 2),


if enable_ls:
    toolbox.register('evaluate', evaluate_ls)
else:
    toolbox.register('evaluate', evaluate)

# size of population and number of generations
n_pop  = 60
n_gen  = 30
champs = 3

# Register genetic operators
# Compared with GP and other genetic algorithms, GEP has its own set 
# of genetic operators aside from common mutation and crossover. For 
# details, please check the tutorial:
# [Introduction to gene expression programming](https://geppy.readthedocs.io/en/latest/intro_GEP.html).
# 
# In the following code, the selection operator is ``tools.selTournament`` 
# provided by DEAP, while all other operators are specially designed for GEP in *geppy*.
toolbox.register('select', tools.selTournament, tournsize=3)
#toolbox.register('select', tools.selTournament, k=round((2.0/3.0)*n_pop)+1, tournsize=2)
#toolbox.register('select', tools.selRoulette)

# 1. general operators
#toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=0.05, pb=1)
#toolbox.register('mut_invert', gep.invert, pb=0.1)
#toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.1)
#toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.1)
#toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.1)
#toolbox.register('cx_1p', gep.crossover_one_point, pb=0.3)
#toolbox.register('cx_2p', gep.crossover_two_point, pb=0.2)
#toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)

toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb='4p', pb=0.1)
#toolbox.register('mut_invert', gep.invert, pb=0.1)
toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.025)
toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.025)
toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.025)
toolbox.register('cx_1p', gep.crossover_one_point, pb=0.05)
toolbox.register('cx_2p', gep.crossover_two_point, pb=0.05)
toolbox.register('cx_gene', gep.crossover_gene, pb=0.025)

toolbox.register('mut_ephemeral', gep.mutate_uniform_ephemeral, ind_pb='1p')  # 1p: expected one point mutation in an individual
toolbox.pbs['mut_ephemeral'] = 1  # we can also give the probability via the pbs property

# 2. Dc-specific operators
#toolbox.register('mut_dc', gep.mutate_uniform_dc, ind_pb=0.05, pb=1)
#toolbox.register('mut_dc', gep.mutate_uniform_dc, ind_pb='4p', pb=0.8)
#toolbox.register('mut_invert_dc', gep.invert_dc, pb=0.1)
#toolbox.register('mut_transpose_dc', gep.transpose_dc, pb=0.1)

# for some uniform mutations, we can also assign the ind_pb a string to indicate our expected number of point mutations in an individual
toolbox.register('mut_rnc_array_dc', gep.mutate_rnc_array_dc, rnc_gen=toolbox.rnc_gen, ind_pb='4p', pb=0.1)
#toolbox.register('mut_rnc_array_dc', gep.mutate_rnc_array_dc, rnc_gen=toolbox.rnc_gen, ind_pb='0.5p')
#toolbox.pbs['mut_rnc_array_dc'] = 1  # we can also give the probability via the pbs property

# Statistics to be inspected
# We often need to monitor of progress of an evolutionary program. 
# DEAP offers two classes to handle the boring work of recording statistics. 
# Details are presented in [Computing statistics](http://deap.readthedocs.io/en/master/tutorials/basic/part3.html). 
# In the following, we are intereted in the average/standard 
# deviation/min/max of all the individuals' fitness in each generation.
stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

Max_evolution = empty_lists = [ [] for i in range(N_eval) ]
Hof_save = empty_lists = [ [] for i in range(N_eval) ]
Max_fit = np.zeros(N_eval)

# Launch evolution
# We make use of *geppy*'s builtin algorithm ``gep_rnc`` here to perform 
# the GEP-RNC evolution. A special class from DEAP, `HallOfFame`, is 
# adopted to store the best individuals ever found. Besides, it should 
# be noted that in GEP [*elitism*](https://en.wikipedia.org/wiki/Genetic_algorithm#Elitism) 
# is highly recommended because some genetic operators in GEP are 
# destructive and may destroy the best individual we have evolved.

pop = toolbox.population(n=n_pop)
hof = tools.HallOfFame(champs) # only record the best three individuals ever found in all generations

startDT = datetime.datetime.now()
print (str(startDT))

# start evolution
for i in range(N_eval):
    pop, log = gep.gep_simple(pop, toolbox, n_generations=n_gen, n_elites=1,
                              stats=stats, hall_of_fame=hof, verbose=True)

    Max_evolution[i] = log.select("min")
    Hof_save[i] = hof

print ("Evolution times were:\n\nStarted:\t", startDT, "\nEnded:   \t", str(datetime.datetime.now()))

# **Let's check the best individuals ever evolved.**
print(hof[0])

# extract statistics:
maxFitnessValues, meanFitnessValues = log.select("max", "avg")

# plot statistics:
sns.set_style("whitegrid")
plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Generation')
plt.ylabel('Max / Average Fitness')
plt.title('Max and Average fitness over Generations')
plt.savefig("fit.pdf", dpi=150, crop='true')
plt.show()

# Present our Work and Conclusions
# Symbolic simplification of the final solution
# The symbolic tree answer may contain many redundancies, for example, 
# `protected_div(x, x)` is just 1. We can perform symbolic simplification
# of the final result by `geppy.simplify` which depends on `sympy` package. 
# We can also leverage sympy to better present our work to others

# print the best symbolic regression we found
#best_ind = Hof_save[0][0]
best_ind = hof[0]
symplified_best = gep.simplify(best_ind)
best_func = toolbox.compile(best_ind)

# convergence
plt.figure()
plt.plot(Max_evolution[0])

temp = np.zeros(N_eval)
for i in range(N_eval):
    print("Hof_save:",Hof_save[i][0])
    temp[i] = Max_evolution[i][-1]


if enable_ls:
    symplified_best = best_ind.a * symplified_best + best_ind.b

key= '''
#Given training examples of
#
#    T = 
#
#we trained a computer using Genetic Algorithms to predict the 
#
#    k = 
#
#Our symbolic regression process found the following equation offers our best prediction:
#
#'''

print('\n', key,'\t', str(symplified_best), '\n\nwhich formally is presented as:\n\n')

from sympy import *
init_printing()
print(symplified_best)
print(str(symplified_best))

# output the top 3 champs
champs = 3
for i in range(champs):
    ind = hof[i]
    symplified_model = gep.simplify(ind)

    print('\nSymplified best individual {}: '.format(i))
    print(symplified_model)
    print("raw indivudal:")
    print(hof[i])

# we want to use symbol labels instead of words in the tree graph
rename_labels = {'add':'+','sub':'-','mul':'*','protected_div':'/','protected_exp':'exp','protected_log':'log','sin':'sin','cos':'cos','tan':'tan'}  
gep.export_expression_tree(best_ind, rename_labels, 'numerical_expression_tree.png')

# As we can see from the above simplified expression, the *truth model* has been successfully found. 
# Due to the existence of Gaussian noise, the minimum mean absolute error ???MAE) is still not zero even the best individual represents the true model.

# ## Visualization
# If you are interested in the expression tree corresponding to the individual, i.e., the genotype/phenotype system, *geppy* supports tree visualization by the `graph` and the `export_expression_tree` functions:
# 
# - `graph` only outputs the nodes and links information to describe the tree topology, with which you can render the tree with tools you like;
# - `export_expression_tree` implements tree visualization with data generated by `graph` internally using the `graphviz` package. 
# 
# **Note**: even if the linear scaling is applied, here only the raw individual in GP (i.e., the one without linear scaling) is visualized.

# show the above image here for convenience
from IPython.display import Image
Image(filename='numerical_expression_tree.png')

# # DoubleCheck our final Test Statistics
# Earlier, we split our data into train and test chunks.
# 
# The GEPPY program never saw 20% of our data, so lets doublecheck the reported errors on our holdout test file are accurate:

def CalculateBestModelOutput(T, model):
#    # pass in a string view of the "model" as str(symplified_best)
#    # this string view of the equation may reference any of the other inputs, T we registered
#    # we then use eval of this string to calculate the answer for these inputs
    return eval(model) 

pred_k = CalculateBestModelOutput(T,str(symplified_best))
pred_k = np.array(pred_k)
print("pred_k.shape=",pred_k.shape)
#pred_k = pred_k[0,:,:]
pred_k = np.transpose(pred_k)
print("pred_k.shape=",pred_k.shape)

print("pred_k=",pred_k)
print("pred_k.shape=",pred_k.shape)

print("pred_k=",type(pred_k))
print("k.shape=",k.shape)

# Validation MSE
# https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, max_error, mean_absolute_error, mean_squared_log_error, median_absolute_error, mean_poisson_deviance, mean_gamma_deviance, mean_absolute_percentage_error

test_mse = mean_squared_error(k, pred_k) 
test_r2  = r2_score(k, pred_k)

print("Mean squared error: %.2f" % mean_squared_error(k, pred_k))
print("R2 score: %.2f" % r2_score(k, pred_k))
print("Explained variance score: %.2f" % explained_variance_score(k, pred_k))
print("Max error: %.2f" % max_error(k, pred_k))
print("Mean absolute error: %.2f" % mean_absolute_error(k, pred_k))
#print("Mean squared log error: %.2f" % mean_squared_log_error(k, pred_k))
print("Median absolute error: %.2f" % median_absolute_error(k, pred_k))
#print("Mean poisson deviance: %.2f" % mean_poisson_deviance(k, pred_k))
#print("Mean gamma deviance: %.2f" % mean_gamma_deviance(k, pred_k))
print("Mean absolute percentage error: %.2f" % mean_absolute_percentage_error(k, pred_k))

# Let's eyeball predicted vs actual data
from matplotlib import pyplot
#pyplot.rcParams['figure.figsize'] = [20, 5]
#plotlen=200
#pyplot.plot(pred_k) # predictions are in blue
#pyplot.plot(k)      # actual values are in orange
#pyplot.savefig("k.png", dpi=150, crop='true')
#pyplot.show()

import matplotlib.pyplot as plt
# shear viscosity vs temperature
#plt.scatter(T, shear,      s=2, c='k', marker='o', label='truth')
#plt.scatter(T, pred_shear, s=2, c='g', marker='+', label='prediction')
plt.plot(T, k, label='truth')
plt.plot(T, pred_k, label='prediction')
plt.ylabel('k')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("k.pdf", dpi=150, crop='true')
plt.show()

# Histogram of prediction Errors on the holdout dataset
pyplot.rcParams['figure.figsize'] = [10, 5]
hfig = pyplot.figure()
ax = hfig.add_subplot(111)

numBins = 100
ax.hist(k-pred_k,numBins,color='green',alpha=0.8)
pyplot.savefig("hist.pdf", dpi=150, crop='true')
pyplot.show()

best_ind = hof[0]
for gene in best_ind:
    print(gene.kexpression)


path = 'results_log.txt'

if os.path.exists(path):
    os.remove(path)
file=open(path, "w")
file.writelines("%s \n%s %s \n%s %s\n%s %s\n%s %s\n%s %s \n%s %s %s \n%s %s\n%s %s\n%s %s\n%s %s\n%s \n" % ('settings',
      'head =',      h,
      '#genes =',    n_genes,
      'len of RNC =',r,
      '# of pop =',  n_pop,
      '# of gen =',  n_gen,
      'best indices =', best_ind.a, best_ind.b,
        'best model =',str(symplified_best),
        'Target model =','...',
        'Test_MSE = ', test_mse,
        'Test_R2 = ',  test_r2,
          log))

file.close()    
