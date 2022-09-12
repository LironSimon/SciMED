# library imports
import pandas as pd
import os

# project imports
from algo.operators.fitness import Fitness
from algo.population import Population
from algo.operators.mutation import Mutation
from utills.logger_config import Logger
from algo.operators.crossover import Crossover
from algo.operators.next_generation import NextGeneration


class GAFS:
    """
    A classical genetic algorithm for grouped feature selection with
    a wrapper of AutoML (MultiTPOTrunner-driven) pipeline search
    """

    def __init__(self):
        pass

    @staticmethod
    def run(tpot_run_times: int,
            feature_generations: int,
            tpot_regressor_generations: int,
            feature_population_size: int,
            tpot_regressor_population_size: int,
            mutation_rate: float,
            feature_indexes_ranges: list,
            mutation_w: list,
            royalty: float,
            k_fold: int,
            performance_metric: str,
            train_data_x: pd.DataFrame,
            train_data_y: pd.DataFrame,
            test_data_x: pd.DataFrame,
            test_data_y: pd.DataFrame,
            save_dir: str,
            cores: int = -1):
        """
        Run the GAFS algorithm with some hyper-parameters
        """
        assert len(mutation_w) == len(feature_indexes_ranges)
        assert feature_generations > 0
        assert tpot_regressor_generations > 0
        assert feature_population_size > 0
        assert (feature_population_size % 2) == 0
        assert tpot_regressor_population_size > 0
        assert k_fold > 0
        assert 0 < royalty < 1
        assert train_data_x.shape[0] == train_data_y.shape[0]
        assert test_data_x.shape[0] == test_data_y.shape[0]
        assert test_data_x.shape[1] == test_data_x.shape[1]

        # generate population of genes dictating how to trim data
        pop = Population.random(size=feature_population_size,
                                feature_count=len(feature_indexes_ranges),
                                feature_indexes_ranges=feature_indexes_ranges)
        # create a dict to store selected features through generations
        selected_fs = {"feature_indices": [],
                       "feature_names": []}
        for generation in range(feature_generations):
            # manipulate gene population
            pop = Mutation.simple(population=pop,
                                  feature_indexes_ranges=feature_indexes_ranges,
                                  mutation_rate=mutation_rate,
                                  w=mutation_w)
            pop = Crossover.simple(population=pop)
            # assign fitness score and best ML pipeline to each gene in pop
            Logger.print("\nGeneration #{}/{} | Assign Fitness and Pipeline to each gene:".format(generation + 1,
                                                                                                  feature_generations))
            pop = Fitness.tpot(run_times=tpot_run_times,
                               train_data_x=train_data_x,
                               train_data_y=train_data_y,
                               test_data_x=test_data_x,
                               test_data_y=test_data_y,
                               generations=tpot_regressor_generations,
                               population=pop,
                               population_size=tpot_regressor_population_size,
                               k_fold=k_fold,
                               performance_metric=performance_metric,
                               n_jobs=cores,
                               save_dir=save_dir)
            # alert user
            current_best_gene = pop.get_best()
            feature_names = list(test_data_x.columns[current_best_gene.feature_indexes])
            selected_fs["feature_indices"].append(current_best_gene.feature_indexes)
            selected_fs["feature_names"].append(feature_names)
            Logger.print("Generation #{}/{} | Best gene's fitness: {:.3f} selected features: {}".format(generation + 1,
                                                                                                        feature_generations,
                                                                                                        current_best_gene.fitness,
                                                                                                        feature_names))
            # prepare population for next generation
            pop = NextGeneration.tournament_with_royalty(population=pop,
                                                         royalty=royalty)
        # save selected features from all generations
        pd.DataFrame(selected_fs, dtype=object).to_csv(os.path.join(save_dir, "selected_features_history.csv"),
                                                       index=False)
        return pop.get_best()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<GAFS>"