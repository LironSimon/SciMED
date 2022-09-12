# library imports

# project imports
from utills.consts import *
from experiments.exp_steady_free_fall_with_drag import ExpSFF
from data_generators.steady_free_fall_with_drag_data_generator import SFFWDdataGenerator


class ExpSFF2(ExpSFF):
    """
    The second case of the SFF -
    Program receives a dataset with all essential features needed
    to deduce a "noisy" target (drag coefficient), except for
    the gravitational acceleration constant.

    Success of both numerical and analytical parts of the program prove
    that:
      1) The program is able to learn the non-linear physical relation between
         features, even with noisy data.
      2) The program was able to learn the gravitational acceleration constant.
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
                   results_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), SFF_2_RESULTS_FOLDER_NAME),
                   data_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), SFF_2_DATA_FOLDER_NAME),
                   data_generation_function=SFFWDdataGenerator.generate_case_2,
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
                   expected_eq="13.08 * (rhop - rhoa) * d  / (rhoa * V * V)",
                   ebs_size_range=SFF_DIMENSIONAL_EBS_SIZE_RANGE_1_2
                   )
