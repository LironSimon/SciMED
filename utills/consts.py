# library imports
import os

# consts #

# 1) General for all experiment:
DATA_FOLDER = "data"
RESULTS_FOLDER = "results"
g_force = 9.81
REL_ERR_OF_STD = 0.05
DEFAULT_FIG_SIZE = 8
DEFAULT_DPI = 600
FEATURE_IMPORTANCE_SIMULATION_COUNT = 5 #100
JSON_INDENT = 2
K_FOLD = 5
RANDOM_STATE = 73
SYMBOLIC_PERCENT_OF_MAJORITY = 0.6
SYMBOLIC_P_VALUE_THRESHOLD = 0.8
SYMBOLIC_EQ_RANKING_METRIC = "r2"
SYMBOLIC_TOP_EQS_MAX_NUM = 5


# 2) Constant acceleration exp:
# - data generation:
CONST_ACCELERATION_NUM_SAMPLES = 400
CONST_ACCELERATION_TEST_SIZE_PORTION = 0.75
# - experiment run:
CONST_ACCELERATION_NUMERICAL_RUN_TIMES = 20
CONST_ACCELERATION_NUMERICAL_GENERATION_COUNT = 5
CONST_ACCELERATION_NUMERICAL_POP_SIZE = 30
CONST_ACCELERATION_ANALYTICAL_RUN_TIMES = 20
CONST_ACCELERATION_ANALYTICAL_GENERATION_COUNT = 5
CONST_ACCELERATION_ANALYTICAL_POP_SIZE = 50
CONST_ACCELERATION_NOISE_RANGE = (0, 0.02)
CONST_ACCELERATION_ANALYTICAL_PARSIMONY_COEFFICIENT = 0.02
CONST_ACCELERATION_EBS_SIZE_RANGE = (5,)
# - result path
CONST_ACCELERATION_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER,
                                                      "constant_acceleration_results",
                                                      "{}_samples".format(CONST_ACCELERATION_NUM_SAMPLES))


# 3) Steady free fall with drag exp:
# - data generation:
N_FREQ_SUFFIX = ['_11', '_12', '_13', '_21', '_23', '_31', '_32']
STEADY_FALL_MINIZE_TOL = 1e-25
SFF_RHOA_RANGE = (998., 1300.)   # fresh water -> salt water at 20C [kg/m3]
SFF_RHOP_RANGE = (0, 5000)
SFF_NU_RANGE = (1e-6, 1.4e-6)    # viscosity corresponding to rhoa [m2/s]
SFF_RE_RANGE = (1., 100.)        # Reynolds range where Cd changes significantly
SFF_CASE_2_NOISE_RANGE = (0, 0.02)
SFF_TEST_SIZE_PORTION = 0.2
SFF_DIMENSIONAL_EBS_SIZE_RANGE_1_2 = (11,)
SFF_DIMENSIONAL_EBS_SIZE_RANGE_2_easy = (7,)
SFF_DIMENSIONAL_EBS_SIZE_RANGE_3 = (3,)
SFF_1_DROP_PARAM = "V"
FORCE_DATA_OVERRIDE_FLAG = False
# - experiment run:
SFF_NUMERICAL_NUM_SAMPLES = 10**4
SFF_NUMERICAL_RUN_TIMES = 20
SFF_NUMERICAL_GENERATION_COUNT = 3
SFF_NUMERICAL_POP_SIZE = 25
SFF_ANALYTICAL_RUN_TIMES = 20
SFF_ANALYTICAL_GENERATION_COUNT = 10
SFF_ANALYTICAL_POP_SIZE = 2000
SFF_ANALYTICAL_PARSIMONY_COEFFICIENT = 0.025
# - feature selection:
FEATURE_SELECTION_GENERATIONS_COUNT = 2
FEATURE_SELECTION_POP_SIZE = 8
FEATURE_SELECTION_MUTATION_RATE = 0.1
FEATURE_SELECTION_ROYALTY = 0.05
# - data and result paths
SFF_N_SAMPLES_STR = str(round(SFF_NUMERICAL_NUM_SAMPLES/1000)) + "k"
SFF_1_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER,
                                         "steady_fall_case_1_results",
                                         "{}_samples_without_{}".format(SFF_N_SAMPLES_STR, SFF_1_DROP_PARAM))
SFF_1_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER,
                                      "case_1_steady_fall_with_drag_data_" +
                                      "{}_samples_no_{}.csv".format(SFF_N_SAMPLES_STR, SFF_1_DROP_PARAM))

SFF_2_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER,
                                         "steady_fall_case_2_results_{}_samples".format(SFF_N_SAMPLES_STR))
SFF_2_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER,
                                      "case_2_steady_fall_with_drag_data_{}_samples.csv".format(SFF_N_SAMPLES_STR))

SFF_2_WITH_GUESS_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER,
                                                    "steady_fall_case_2_with_guess_results_{}_samples".format(SFF_N_SAMPLES_STR))
SFF_2_WITH_GUESS_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER,
                                                 "case_2_with_guess_steady_fall_with_drag_data_{}_samples.csv".format(SFF_N_SAMPLES_STR))

SFF_3_RESULTS_FOLDER_NAME = os.path.join(RESULTS_FOLDER,
                                         "steady_fall_case_3_results_{}_samples".format(SFF_N_SAMPLES_STR))
SFF_3_DATA_FOLDER_NAME = os.path.join(DATA_FOLDER,
                                      "case_3_steady_fall_with_drag_data_{}_samples.csv".format(SFF_N_SAMPLES_STR))

# end - consts #
