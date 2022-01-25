import numpy as np
import time
import random

from sklearn import model_selection
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV

from pandas import read_csv


class HyperparameterTuningGrid:

    NUM_FOLDS = 5

    def __init__(self, randomSeed):

        self.randomSeed = randomSeed
        self.initDataset()
        self.initRegressor()
        self.initKfold()
        self.initGridParams()

    def initDataset(self):
        data = np.loadtxt("../data/STS/shear_viscosity.txt")
        self.X = data[:,0:51]           # press, T, TVCO2, TVO2, TVCO, x[5]
        self.y = data[:,51:52].ravel()  # shear viscosity

    def initRegressor(self):
        self.regressor = AdaBoostRegressor(random_state=self.randomSeed)

    def initKfold(self):
        self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS, shuffle=True,
                                           random_state=self.randomSeed)

    def initGridParams(self):
        self.gridParams = {
            'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'learning_rate': np.logspace(-2, 0, num=10, base=10),
            'loss': ['linear', 'square', 'exponential'],
        }

    def getDefaultAccuracy(self):
        cv_results = model_selection.cross_val_score(self.regressor,
                                                     self.X,
                                                     self.y,
                                                     cv=self.kfold,
                                                     scoring='r2')
        return cv_results.mean()

    def gridTest(self):
        print("performing grid search...")

        gridSearch = GridSearchCV(estimator=self.regressor,
                                  param_grid=self.gridParams,
                                  cv=self.kfold,
                                  scoring='r2',
                                  #iid='False',
                                  n_jobs=2)

        gridSearch.fit(self.X, self.y)
        print("best parameters: ", gridSearch.best_params_)
        print("best score: ", gridSearch.best_score_)

    def geneticGridTest(self):
        print("performing Genetic grid search...")

        gridSearch = EvolutionaryAlgorithmSearchCV(estimator=self.regressor,
                                                   params=self.gridParams,
                                                   cv=self.kfold,
                                                   scoring='r2',
                                                   verbose=True,
                                                   iid='False',
                                                   n_jobs=2,
                                                   population_size=20,
                                                   gene_mutation_prob=0.30,
                                                   tournament_size=2,
                                                   generations_number=5)
        gridSearch.fit(self.X, self.y)
