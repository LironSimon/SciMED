# library imports
import os
import time
import pandas as pd
from datetime import timedelta
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# project imports
from utills.consts import *
from utills.fitness_methods import *
from utills.logger_config import Logger
from algo.equation_brute_search import EBS
from utills.result_tracker import ResultTracker
from algo.multi_tpot_analysis import MultiTPOTrunner
from algo.genetic_algorithm_symbolic_fit import GASF
from data_generators.drag_force_data_generator import DragForceDataGenerator


class ExpDragFroce:
    """
    Program receives a dataset with all essential features needed
    to deduce a "noisy" target (drag on sphere).
    Success of both numerical and analytical parts of the program prove
    that the program is able to learn a complex polynomial relation between
    features, even with noisy data.
    """

    def __init__(self):
        pass

    @staticmethod
    def run(numerical_bool: bool,
            analytical_bool: bool,
            force_ebs_bool: bool):
        """
        Entry point
        """
        # config logging
        start_time = time.time()

        # prepare IO
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), DRAG_FORCE_RESULTS_FOLDER_NAME),
                    exist_ok=True)
        Logger(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            DRAG_FORCE_RESULTS_FOLDER_NAME,
                            "run.log"))

        # 1) generate data
        data_path = os.path.join(os.path.dirname(__file__), "..", "data",
                                 "drag_force_{}_samples.csv".format(DRAG_FORCE_NUM_SAMPLES))
        DragForceDataGenerator.generate(samples=DRAG_FORCE_NUM_SAMPLES,
                                        cd_range=(1, 10),
                                        rho_range=(30, 50),
                                        v_range=(1, 10),
                                        d_range=(0.01, 0.1),
                                        noise_range= DRAG_FORCE_NOISE_RANGE,
                                        save_path=data_path)
        # 1.1) load data, normalize and split
        df = pd.read_csv(data_path)
        Logger.print('Generated data:\n{}'.format(df.describe()))
        y_col = df.keys()[-1]
        normalized_df = (df - df.min()) / (df.max() - df.min())
        train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(normalized_df.drop([y_col], axis=1),
                                                                                normalized_df[y_col],
                                                                                shuffle=True,
                                                                                test_size=DRAG_FORCE_TEST_SIZE_PORTION,
                                                                                random_state=RANDOM_STATE)
        # 1.2) log elapsed time
        data_end_time = time.time()
        Logger.print("   --- Finished. Elapsed time: {} ---".format(timedelta(seconds=data_end_time - start_time)))

        # 2.1) continue to the MultiTPOTrunner regression
        Logger.print('Training MultiTPOTrunner:')
        if numerical_bool:
            # 2.1) find the best ML model for data
            all_t_scores, best_t_model = MultiTPOTrunner.run_and_analyze(run_times=DRAG_FORCE_NUMERICAL_RUN_TIMES,
                                                                         train_data_x=train_data_x,
                                                                         train_data_y=train_data_y,
                                                                         test_data_x=test_data_x,
                                                                         test_data_y=test_data_y,
                                                                         generations=DRAG_FORCE_NUMERICAL_GENERATION_COUNT,
                                                                         population_size=DRAG_FORCE_NUMERICAL_POP_SIZE,
                                                                         k_fold=K_FOLD,
                                                                         performance_metric=neg_mean_squared_error_scorer,
                                                                         save_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                                                               DRAG_FORCE_RESULTS_FOLDER_NAME),
                                                                         n_jobs=-1)
            # 2.2) save results of best model from all runs
            ResultTracker.run(program_part="tpot",
                              run_times=DRAG_FORCE_NUMERICAL_RUN_TIMES,
                              all_scores=all_t_scores,
                              model=best_t_model,
                              train_data_x=train_data_x,
                              train_data_y=train_data_y,
                              test_data_x=test_data_x,
                              test_data_y=test_data_y,
                              save_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                    DRAG_FORCE_RESULTS_FOLDER_NAME))
        # 2.2) log elapsed time
        tpot_end_time = time.time()
        Logger.print("   --- Finished. Elapsed time: {} ---".format(timedelta(seconds=tpot_end_time - data_end_time)))

        # 3) continue to the symbolic regression
        Logger.print('Searching for a symbolic expression:')
        if analytical_bool:
            # 3.1) run symbolic regressor multiple times
            all_s_scores, best_s_model = GASF.run_and_analyze(run_times=DRAG_FORCE_NUMERICAL_RUN_TIMES,
                                                              non_normalized_data=df,
                                                              performance_metric=function_mapper["better_symbolic_reg_fitness"],
                                                              generations=DRAG_FORCE_ANALYTICAL_GENERATION_COUNT,
                                                              population_size=DRAG_FORCE_ANALYTICAL_POP_SIZE,
                                                              k_fold=K_FOLD,
                                                              cores=-1,
                                                              parsimony_coefficient=DRAG_FORCE_ANALYTICAL_PARSIMONY_COEFFICIENT,
                                                              expected_eq='mul(0.392, mul(cd, mul(rho, mul(v, mul(v, mul(d, d))))))',
                                                              save_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                                                    DRAG_FORCE_RESULTS_FOLDER_NAME))
            # 3.2) save results of best model from all runs
            ResultTracker.run(program_part="symbolic",
                              run_times=DRAG_FORCE_NUMERICAL_RUN_TIMES,
                              all_scores=all_s_scores,
                              model=best_s_model,
                              train_data_x=train_data_x,
                              train_data_y=train_data_y,
                              test_data_x=test_data_x,
                              test_data_y=test_data_y,
                              save_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                    DRAG_FORCE_RESULTS_FOLDER_NAME))
            # 3.3) save a summary of the eqs found & figure whether to continue to ebf
            ebs_flag = ResultTracker.summaries_symbolic_results(run_times=DRAG_FORCE_NUMERICAL_RUN_TIMES,
                                                                percent_of_majority=SYMBOLIC_PERCENT_OF_MAJORITY,
                                                                eq_ranking_metric=SYMBOLIC_EQ_RANKING_METRIC,
                                                                top_eqs_max_num=SYMBOLIC_TOP_EQS_MAX_NUM,
                                                                save_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                                                      DRAG_FORCE_RESULTS_FOLDER_NAME))
            # 3.4) log elapsed time
            symbolic_end_time = time.time()
            Logger.print("Finished. Elapsed time: {}".format(timedelta(seconds=symbolic_end_time - tpot_end_time)))

            # 4) continue to the EBS
            if ebs_flag or force_ebs_bool:
                Logger.print('Searching for a symbolic expression using EBF:')
                # 4.1) run EBS multiple times
                all_ebs_scores, best_ebs_model = EBS.run_and_analyze(run_times=DRAG_FORCE_NUMERICAL_RUN_TIMES,
                                                                     non_normalized_data=df,
                                                                     performance_metric=function_mapper["better_symbolic_reg_fitness"],
                                                                     cores=-1,
                                                                     size_range=DRAG_FORCE_EBS_SIZE_RANGE,
                                                                     expected_eq='mul(0.392, mul(cd, mul(rho, mul(v, mul(v, mul(d, d))))))',
                                                                     save_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                                                           DRAG_FORCE_RESULTS_FOLDER_NAME))
                # 4.2) save the fitting score results
                ResultTracker.ebs_results(model=best_ebs_model,
                                          all_scores=all_ebs_scores,
                                          save_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                                DRAG_FORCE_RESULTS_FOLDER_NAME))
            else:
                Logger.print("EBS search of a symbolic equation wasn't needed")
            # 4.3) log elapsed time
            ebs_end_time = time.time()
            Logger.print("Finished. Elapsed time: {}".format(timedelta(seconds=ebs_end_time - symbolic_end_time)))

        # 5) alert results to the user
        Logger.print("TOTAL TIME ELAPSED TIME: {}".format(timedelta(seconds=time.time() - start_time)))
