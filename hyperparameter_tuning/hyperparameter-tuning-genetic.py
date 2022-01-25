# Import essential DEAP modules
from deap import base
from deap import creator
from deap import tools

import random
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import hyperparameter_tuning_genetic_test
import elitism

# Boundaries for ADABOOST parameters
# "n_estimators": 1..100
# "learning_rate": 0.01..100
# "loss": 0, 1, 2
# [n_estimators, learning_rate, loss]:
BOUNDS_LOW =  [  1, 0.01, 0]
BOUNDS_HIGH = [100, 1.00, 1]

NUM_OF_PARAMS = len(BOUNDS_HIGH)

# Genetic Algorithm constants
POPULATION_SIZE   = 20   # number of individuals in population
P_CROSSOVER       = 0.9  # probability for crossover
P_MUTATION        = 0.5  # probability for mutating an individual
MAX_GENERATIONS   = 10   # max number of generations for stopping condition
HALL_OF_FAME_SIZE = 5
CROWDING_FACTOR   = 20.0 # crowding factor for crossover and mutation

# Set the random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Create the regressor accuracy test class
test = hyperparameter_tuning_genetic_test.HyperparameterTuningGenetic(RANDOM_SEED)

# Instantiate toolbox variable
toolbox = base.Toolbox()

# Define a single objective, maximizing fitness strategy.
# This will yield a creator.FitnessMax class extending 
# the base.Fitness class, with the weights class attribute 
# initialized to a value of (1.0,).
# Note the trailing comma in the weights definition when a 
# single weight is defined. The comma is required because
# weights is a tuple.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))                  # maximizing the fitness value  
#creator.create("FitnessMin", base.Fitness, weights=(-1.0,))                # minimizes  the fitness value
#creator.create("FitnessCompound", base.Fitness, weights=(1.0, 0.2, -0.5))  # optimizing more than one objective, and with varying degrees of importance

# Create the Individual class based on list.
# The created Individual class extends the Python list class. 
# This means that the chromosome used is of the list type.
# Each instance of this Individual class will have an attribute 
# called fitness, of the FitnessMax class we previously created.
creator.create("Individual", list, fitness=creator.FitnessMax)

# Define the hyperparameter attributes individually
for i in range(NUM_OF_PARAMS):
    # "hyperparameter_0", "hyperparameter_1", ...
    toolbox.register("hyperparameter_" + str(i),
                     random.uniform,
                     BOUNDS_LOW[i],
                     BOUNDS_HIGH[i])

# Create a tuple containing an attribute generator for each param searched
hyperparameters = ()
for i in range(NUM_OF_PARAMS):
    hyperparameters = hyperparameters + \
                      (toolbox.__getattribute__("hyperparameter_" + str(i)),)

# Create the individual operator to fill up an Individual instance
toolbox.register("individualCreator",
                 tools.initCycle,
                 creator.Individual,
                 hyperparameters,
                 n=1)

# Create the population operator to generate a list of individuals
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

# Fitness calculation
def regressionAccuracy(individual):
    return test.getAccuracy(individual),

# Use the evaluate alias for calculating the fitness function
toolbox.register("evaluate", regressionAccuracy)

# Genetic operators
# select is registered as an alias to the existing tools function,
# selTournament(), with the tournsize argument set to 3. This creates a
# toolbox.select operator that performs tournament selection with a
# tournament size of 3.
toolbox.register("select", tools.selTournament, tournsize=2)

# mate is registered as an alias to the existing tools function, 
# cxSimulatedBinaryBounded. This results in a toolbox.mate operator 
# that performs two-point crossover.
toolbox.register("mate",
                 tools.cxSimulatedBinaryBounded,
                 low=BOUNDS_LOW,
                 up=BOUNDS_HIGH,
                 eta=CROWDING_FACTOR)

# mutate is registered as an alias to the existing tools function, 
# mutPolynomialBounded, with the indpb argument set to 1.0, providing 
# a toolbox.mutate operator that performs flip bit mutation with 1.0
# as the probability for each attribute to be flipped
toolbox.register("mutate",
                 tools.mutPolynomialBounded,
                 low=BOUNDS_LOW,
                 up=BOUNDS_HIGH,
                 eta=CROWDING_FACTOR,
                 indpb=1.0 / NUM_OF_PARAMS)


# Genetic Algorithm flow
def main():

    # Create initial population (generation 0)
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # Prepare the statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    # Define the hall-of-fame object
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # Perform the Genetic Algorithm flow with hof feature added
    population, logbook = elitism.eaSimpleWithElitism(population,
                                                      toolbox,
                                                      cxpb=P_CROSSOVER,
                                                      mutpb=P_MUTATION,
                                                      ngen=MAX_GENERATIONS,
                                                      stats=stats,
                                                      halloffame=hof,
                                                      verbose=True)

    # Print best solution found
    print("- Best solution is: ")
    print("params = ", test.formatParams(hof.items[0]))
    print("Accuracy = %1.5f" % hof.items[0].fitness.values[0])

    # Extract statistics
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    # Plot statistics
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average fitness over Generations')
    plt.show()


if __name__ == "__main__":
    main()
