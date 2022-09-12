# library imports

# project imports
from utills.consts import *
from experiments.exp_steady_free_fall_with_drag import ExpSFF
from data_generators.steady_free_fall_with_drag_data_generator import SFFWDdataGenerator


class ExpSFF3(ExpSFF):
    """
    The third case of the SFF -
    Program receives a dataset with all non-dimensional
    combinations of possible dimensional variables, relating to the
    target (a total of 34 features created from 5 variables).

    The program selects the best feature from each group of
    similar non-dimensional numbers, to create an improved
    dataset. This is then used to find numerical and analytical relations.

    Success of both numerical and analytical parts of the program proves
    that:
      1) The program is capable of discovering the governing non-dimensional
         numbers that explain the physical phenomena, with no prior physical
         knowledge given by the user.
      2) The program is able to learn the physical relation between
         non-dimensional features and the target
      3) The program was able to learn the gravitational acceleration constant.
    """

    def __init__(self):
        ExpSFF.__init__(self)

    @staticmethod
    def perform(numerical_bool: bool,
                analytical_bool: bool,
                force_ebs_bool: bool):
        """
        Entry point
        """
        ExpSFF.run(numerical_bool=numerical_bool,
                   analytical_bool=analytical_bool,
                   force_ebs_bool=force_ebs_bool,
                   results_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), SFF_3_RESULTS_FOLDER_NAME),
                   data_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), SFF_3_DATA_FOLDER_NAME),
                   data_generation_function=SFFWDdataGenerator.generate_case_3,
                   numerical_run_times=SFF_NUMERICAL_RUN_TIMES,
                   numerical_generations=SFF_NUMERICAL_GENERATION_COUNT,
                   numerical_population=SFF_NUMERICAL_POP_SIZE,
                   analytical_run_times=SFF_ANALYTICAL_RUN_TIMES,
                   analytical_generations=SFF_ANALYTICAL_GENERATION_COUNT,
                   analytical_population=SFF_NUMERICAL_POP_SIZE,
                   parsimony_coefficient=SFF_ANALYTICAL_PARSIMONY_COEFFICIENT,
                   k_fold=K_FOLD,
                   samples=SFF_NUMERICAL_NUM_SAMPLES,
                   rhoa_range=SFF_RHOA_RANGE,
                   rhop_range=SFF_RHOP_RANGE,
                   nu_range=SFF_NU_RANGE,
                   re_range=SFF_RE_RANGE,
                   expected_eq="1) 1.33 * (delta_rho/rhoa) * (g*d/V**2)",
                   feature_selection_generations=FEATURE_SELECTION_GENERATIONS_COUNT,
                   feature_selection_pop_size=FEATURE_SELECTION_POP_SIZE,
                   feature_selection_mutation_rate=FEATURE_SELECTION_MUTATION_RATE,
                   feature_selection_royalty=FEATURE_SELECTION_ROYALTY,
                   ebs_size_range=SFF_DIMENSIONAL_EBS_SIZE_RANGE_3
                   )
