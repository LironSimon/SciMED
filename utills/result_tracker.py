# library imports
import os
import json
import pickle
import pandas as pd

# project imports
from utills.consts import *
from utills.plotter import Plotter
from utills.symbolic_regression_to_latex_text import SymbolicRegressionToLatexText


class ResultTracker:
    """
    This class is responsible for saving  plots and data
    for each part of the program.
    """

    def __init__(self):
        pass

    @staticmethod
    def run(program_part: str,
            run_times: int,
            all_scores: pd.DataFrame,
            model,
            train_data_x: pd.DataFrame,
            train_data_y: pd.DataFrame,
            test_data_x: pd.DataFrame,
            test_data_y: pd.DataFrame,
            save_dir: str):
        """

        """
        assert program_part in ["tpot", "symbolic"]

        # 1) save model
        if program_part == "tpot":
            model.export(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                      save_dir,
                                      "tpot_exported_pipeline.py"))
        else:
            with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), save_dir,
                                   "symbolic_model"), "wb") as symbolic_fit_file:
                pickle.dump(model, symbolic_fit_file)

        # 2) save scoring history of model as a whole and as averaged
        all_scores.to_csv(os.path.join(save_dir, program_part + "_scoring_history.csv"), index=False)
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), save_dir,
                               program_part + "_fit_results.json"), "w") as test_file:
            json.dump({key: all_scores[key].mean() for key in all_scores.keys()[:-1]},
                      test_file,
                      indent=JSON_INDENT)

        # 3) plot model's predictions vs true values
        Plotter.y_test_vs_y_pred(model=model,
                                 x_test=pd.concat([train_data_x, test_data_x]),
                                 y_test=pd.concat([train_data_y, test_data_y]),
                                 save_path=os.path.join(save_dir, program_part + "_target_vs_pred.pdf"))

        # 4) varify that mae scores are stable
        if run_times > 1:
            Plotter.std_check(data=all_scores["mae"],
                              save_path=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                     save_dir,
                                                     program_part + "_mae_stability.pdf"))
        # 5) plot feature importance
        dataset_x = pd.concat([train_data_x, test_data_x])
        dataset_y = pd.concat([train_data_y, test_data_y])
        dataset = pd.concat([dataset_x, dataset_y], axis=1)

        # if program_part == "tpot":
        #     Plotter.feature_importance(model=model,
        #                                dataset=dataset,
        #                                save_dir=save_dir,
        #                                program_part=program_part,
        #                                simulations=FEATURE_IMPORTANCE_SIMULATION_COUNT)

        if program_part == "symbolic":
            p_value = all_scores["t_test_p_value"].mean()
            if p_value < SYMBOLIC_P_VALUE_THRESHOLD:
                continue_to_ebs_flag = True
            else:
                continue_to_ebs_flag = False
            return continue_to_ebs_flag

    @staticmethod
    def summaries_symbolic_results(run_times: int,
                                   percent_of_majority: float,
                                   eq_ranking_metric: str,
                                   top_eqs_max_num: int,
                                   save_dir: str):
        """

        """
        # load data
        eqs = pd.read_csv(os.path.join(save_dir, "symbolic_scoring_history.csv"))["found_eq"]
        eq_ranking = pd.read_csv(os.path.join(save_dir, "symbolic_scoring_history.csv"))[eq_ranking_metric]
        # write summary file:
        with open(os.path.join(save_dir, "symbolic_results_summary.txt"), 'w') as f:
            f.write("Symbolic run count: {}\n\n".format(run_times))
            # check if program needs to continue to ebf search
            if list(eqs.value_counts())[0] >= percent_of_majority * run_times:
                f.write("The function that repeated in {}% of the runs: \n  {}\n\n".format(
                    round(list(eqs.value_counts())[0] * 100 / run_times, 2),
                    eqs.value_counts().index[0]#SymbolicRegressionToLatexText.run(eq=str(eqs.value_counts().index[0]))
                ))
                continue_to_ebs_flag = False
            else:
                f.write("No function was found for at least {} of the runs\n\n".format(round(percent_of_majority * 100,
                                                                                             2)))
                continue_to_ebs_flag = True
            # rank the eqs found by metric:
            top_eqs_index = eq_ranking.sort_values(ascending=False)[:top_eqs_max_num].index
            f.write("{} best equations found (according to {} score):\n".format(len(eqs[top_eqs_index].unique()),
                                                                                eq_ranking_metric))
            for i, eq in enumerate(eqs[top_eqs_index].unique()):
                f.write(" {}) {}\n".format(i + 1, eq)) #SymbolicRegressionToLatexText.run(eq=str(eq))))
            # alert user of findings
            return continue_to_ebs_flag

    @staticmethod
    def ebs_results(model,
                    all_scores: pd.DataFrame,
                    save_dir: str):
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), save_dir,
                               "ebs_fit_score.json"), "w") as ebs_fit_score_file:
            answer = {"k_fold": K_FOLD}
            for key in all_scores.keys()[:-1]:
                answer[key] = all_scores[key].mean()
            json.dump(answer,
                      ebs_fit_score_file,
                      indent=JSON_INDENT)
        # save best fitted model
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               CONST_ACCELERATION_RESULTS_FOLDER_NAME,
                               "ebs_model"),
                  "wb") as ebs_fit_file:
            pickle.dump(model, ebs_fit_file)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<ResultTracker>"
