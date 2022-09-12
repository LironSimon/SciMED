# library imports
import pandas as pd

# project imports
from algo.population import Population
from utills.logger_config import Logger
from algo.multi_tpot_analysis import MultiTPOTrunner


class Fitness:
    """
    An AutoML wrapper class that responsible to find the best
    ML model + hyperparameters for a given dataset
    """

    def __init__(self):
        pass

    @staticmethod
    def tpot(run_times: int,
             train_data_x: pd.DataFrame,
             train_data_y: pd.DataFrame,
             test_data_x: pd.DataFrame,
             test_data_y: pd.DataFrame,
             generations: int,
             population: Population,
             population_size: int,
             k_fold: int,
             performance_metric,
             save_dir: str,
             n_jobs: int = -1):
        """
        Loops over all genes in the population, reduces the dataset
        according to their sequence, and uses a TPOTRegressor to
        find the best ML model + hyperparameters for the reduced data.

        Saves the fittness score and best ML model to the memory of each gene,
        and returns the whole population.
        """
        for gene_index, gene in enumerate(population.genes):
            Logger.print("\nAssigning to gene #{}/{}:".format(gene_index, population.size()-1))
            # reduce data according to gene sequence
            reduced_train_data_x = train_data_x.iloc[:, gene.feature_indexes]
            reduced_test_data_x = test_data_x.iloc[:, gene.feature_indexes]
            # run TPOT analysis multiple times on the reduced data
            results, best_model = MultiTPOTrunner.run_and_analyze(run_times=run_times,
                                                                  train_data_x=reduced_train_data_x,
                                                                  train_data_y=train_data_y,
                                                                  test_data_x=reduced_test_data_x,
                                                                  test_data_y=test_data_y,
                                                                  generations=generations,
                                                                  population_size=population_size,
                                                                  k_fold=k_fold,
                                                                  performance_metric=performance_metric,
                                                                  save_dir=save_dir,
                                                                  n_jobs=n_jobs)
            # save the best performing model & scoring history to gene's data
            gene.model_object = best_model
            gene.scoring_history = results
            # assign fittness acc to performance score of the model
            gene.fitness = best_model.score(reduced_test_data_x, test_data_y)
        return population

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<Fitness-GA-operator>"
