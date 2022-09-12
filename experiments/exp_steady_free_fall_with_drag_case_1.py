# library imports

# project imports
from utills.consts import *
from experiments.exp_steady_free_fall_with_drag import ExpSFF
from data_generators.steady_free_fall_with_drag_data_generator import SFFWDdataGenerator


class ExpSFF1(ExpSFF):
    """
    The first case of the SFF -
    Program receives a dataset that is missing an essential feature
    (particle velocity), that is needed to deduce the target (drag coefficient).

    Failure of both numerical and analytical parts of the program prove
    that if the user neglects to measure a key physical component in the
    unknown physical phenomena, the user is alerted by bad results.
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
                   results_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), SFF_1_RESULTS_FOLDER_NAME),
                   data_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), SFF_1_DATA_FOLDER_NAME),
                   data_generation_function=SFFWDdataGenerator.generate_case_1,
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
                   expected_eq="unknown",
                   ebs_size_range=SFF_DIMENSIONAL_EBS_SIZE_RANGE_1_2
                   )
