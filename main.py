# library imports
import os

# project imports
from experiments.exp_steady_free_fall_with_drag_case_1 import ExpSFF1
from experiments.exp_steady_free_fall_with_drag_case_2 import ExpSFF2
from experiments.exp_steady_free_fall_with_drag_case_3 import ExpSFF3
from experiments.exp_constant_acceleration import ExpConstantAcceleration
from experiments.exp_steady_free_fall_with_drag_case_2_with_educated_guess import ExpSFF2WithGuess


class Main:
    """
    Single entry point for the project.
    This file runs all the experiments in the project and save the raw results
    for the manuscript
    """

    # CONSTS #
    RESULTS_FOLDER_NAME = "results"

    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def run(const_acc_numerical_bool: bool = True,
            const_acc_analytical_bool: bool = True,
            const_acc_force_ebs_bool: bool = True,
            sff1_numerical_bool: bool = True,
            sff1_analytical_bool: bool = True,
            sff1_force_ebs_bool: bool = True,
            sff2_numerical_bool: bool = True,
            sff2_analytical_bool: bool = True,
            sff2_force_ebs_bool: bool = True,
            sff2_with_guess_numerical_bool: bool = True,
            sff2_with_guess_analytical_bool: bool = True,
            sff2_with_guess_force_ebs_bool: bool = True,
            sff3_numerical_bool: bool = True,
            sff3_analytical_bool: bool = True,
            sff3_force_ebs_bool: bool = True):
        """
        Single method to use in the class.
        Run the experiments, if requested
        """
        # prepare IO
        os.makedirs(os.path.join(os.path.dirname(__file__), Main.RESULTS_FOLDER_NAME),
                    exist_ok=True)
        # run all the experiments
        if const_acc_numerical_bool or const_acc_analytical_bool or const_acc_force_ebs_bool:
            ExpConstantAcceleration.run(numerical_bool=const_acc_numerical_bool,
                                        analytical_bool=const_acc_analytical_bool,
                                        force_ebs_bool=const_acc_force_ebs_bool)

        if sff1_numerical_bool or sff1_analytical_bool or sff1_force_ebs_bool:
            ExpSFF1.perform(numerical_bool=sff1_numerical_bool,
                            analytical_bool=sff1_analytical_bool,
                            force_ebs_bool=sff1_force_ebs_bool)

        if sff2_numerical_bool or sff2_analytical_bool or sff2_force_ebs_bool:
            ExpSFF2.perform(numerical_bool=sff2_numerical_bool,
                            analytical_bool=sff2_analytical_bool,
                            force_ebs_bool=sff2_force_ebs_bool)

        if sff2_with_guess_numerical_bool or sff2_with_guess_analytical_bool or sff2_with_guess_force_ebs_bool:
            ExpSFF2WithGuess.perform(numerical_bool=sff2_with_guess_numerical_bool,
                                     analytical_bool=sff2_with_guess_analytical_bool,
                                     force_ebs_bool=sff2_with_guess_force_ebs_bool)

        if sff3_numerical_bool or sff3_analytical_bool or sff3_force_ebs_bool:
            ExpSFF3.perform(numerical_bool=sff3_numerical_bool,
                            analytical_bool=sff3_analytical_bool,
                            force_ebs_bool=sff3_force_ebs_bool)


if __name__ == '__main__':
    Main.run(const_acc_numerical_bool=False,
             const_acc_analytical_bool=False,
             const_acc_force_ebs_bool=False,

             sff1_numerical_bool=False,
             sff1_analytical_bool=False,
             sff1_force_ebs_bool=False,

             sff2_numerical_bool=False,
             sff2_analytical_bool=False,
             sff2_force_ebs_bool=False,

             sff2_with_guess_numerical_bool=False,
             sff2_with_guess_analytical_bool=False,
             sff2_with_guess_force_ebs_bool=False,

             sff3_numerical_bool=False,
             sff3_analytical_bool=True,
             sff3_force_ebs_bool=False)
