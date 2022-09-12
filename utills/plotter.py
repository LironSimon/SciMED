# library imports
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# project imports
from utills.consts import *
from utills.logger_config import Logger

# fix for windows
matplotlib.use('Agg')


class Plotter:
    """
    A plotter class for the results of the model
    """

    def __init__(self):
        pass

    @staticmethod
    def noise_graph(noise_range: list,
                    y_list: dict,
                    save_path: str):
        colors = ["red", "blue", "green"]
        symbols = ["o", "*", "P"]

        fig = plt.figure(figsize=(DEFAULT_FIG_SIZE, DEFAULT_FIG_SIZE))
        # calc predictions and save to csv
        index = 0
        for name, y in y_list.items():
            plt.plot(noise_range,
                     y,
                     "-{}".format(symbols[index]),
                     color=colors[index],
                     s=20,
                     alpha=0.5,
                     label="{}".format(name))
            index += 1
        # set parameters and save plot
        plt.xlim((min(noise_range), max(noise_range)))
        plt.ylim((0, 1))
        plt.xlabel("Noise Level", fontsize=16)
        plt.ylabel("Successful rate", fontsize=16)
        plt.legend(frameon=True, fontsize=13)
        plt.grid(alpha=0.5)
        ax = plt.gca()
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(save_path, dpi=DEFAULT_DPI)
        plt.close()

    @staticmethod
    def y_test_vs_y_pred(model,
                         x_test,
                         y_test,
                         save_path: str):
        fig = plt.figure(figsize=(DEFAULT_FIG_SIZE, DEFAULT_FIG_SIZE))
        # calc predictions and save to csv
        y_pred = model.predict(x_test)
        pd.DataFrame({'y_pred': y_pred, 'y_true': y_test}).to_csv(save_path[:-4] + '.csv', index=False)
        pts_range = (min([min(y_test), min(y_pred)]), max([max(y_test), max(y_pred)]))
        # plot predictions them against actual values
        y_test = np.array(y_test).reshape(-1, 1)
        lg = LinearRegression().fit(y_test, y_pred)
        r2 = lg.score(y_test, y_pred)
        plt.scatter(x=y_test,
                    y=y_pred,
                    color="blue",
                    s=20,
                    alpha=0.5)
        # plot y_pred = y_true for ref
        plt.plot([min(y_test), max(y_test)],
                 [min(y_test), max(y_test)],
                 "-",
                 color="black",
                 linewidth=1,
                 alpha=0.75)
        # plot actual y_pred = f(y_true) relation
        plt.plot([min(y_test), max(y_test)],
                 [lg.predict([min(y_test)])[0], lg.predict([max(y_test)])[0]],
                 "--",
                 color="gray",
                 linewidth=2,
                 alpha=0.75,
                 label="$R^2$ = " + str(round(r2, 3)) + " | $y_{pred} = y_{exp} * " + str(
                     round(lg.coef_[0], 3)) + " + " + str(round(lg.intercept_, 3)) + "$")
        # set parameters and save plot
        plt.xlim(pts_range)
        plt.ylim(pts_range)
        plt.xlabel("True value", fontsize=16)
        plt.ylabel("Predicted value", fontsize=16)
        plt.legend(frameon=True, fontsize=13)
        plt.grid(alpha=0.5)
        ax = plt.gca()
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(save_path, dpi=DEFAULT_DPI)
        plt.close()

    @staticmethod
    def parameter_sensitivity_graph(model,
                                    baseline_x: list,
                                    parameter_col_index: int,
                                    parameter_start_range: float,
                                    parameter_end_range: float,
                                    parameter_steps_count: int,
                                    parameter_name: str,
                                    target_name: str,
                                    save_path: str):
        fig = plt.figure(figsize=(DEFAULT_FIG_SIZE, DEFAULT_FIG_SIZE))
        x_data = []
        x_values = []
        # prepare data
        step_size = (parameter_end_range - parameter_start_range) / parameter_steps_count
        for i in range(parameter_steps_count):
            new_row = baseline_x.copy()
            new_row[parameter_col_index] = parameter_start_range + i * step_size
            x_values.append(new_row[parameter_col_index])
            x_data.append(new_row)
        df = pd.DataFrame(x_data)
        y_pred = model.predict(df)
        plt.plot(x_values,
                 y_pred,
                 "-o",
                 color="black")
        plt.xlim((parameter_start_range, parameter_end_range))
        plt.ylim((min(y_pred), max(y_pred)))
        plt.xlabel(parameter_name, fontsize=16)
        plt.xlabel(target_name, fontsize=16)
        ax = plt.gca()
        plt.grid()
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.savefig(save_path, dpi=DEFAULT_DPI)
        plt.close()

    @staticmethod
    def std_check(data, save_path):
        # calc std on increasing num of samples
        vals = []
        for i in data.index[1:]:
            vals.append(float(data.loc[:i].std()))
        # plot std development over iterations
        fig = plt.figure(figsize=(DEFAULT_FIG_SIZE, DEFAULT_FIG_SIZE))
        plt.scatter(x=range(1, 1 + len(vals)),
                    y=vals,
                    color="blue",
                    s=20,
                    alpha=0.5)
        plt.axhline(y=vals[-1] * (1 + REL_ERR_OF_STD),
                    linestyle="--",
                    color="red",
                    linewidth=1)
        plt.axhline(y=vals[-1] * (1 - REL_ERR_OF_STD),
                    linestyle="--",
                    color="red",
                    linewidth=1)  # set parameters and save plot
        plt.xlim([1, len(vals)])
        plt.xlabel("Iteration", fontsize=16)
        plt.ylabel("Standard Deviation", fontsize=16)
        plt.grid(alpha=0.5)
        ax = plt.gca()
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(save_path, dpi=DEFAULT_DPI)
        plt.close()

    @staticmethod
    def feature_importance(model,
                           dataset: pd.DataFrame,
                           save_dir: str,
                           program_part: str,
                           simulations: int = 100):
        # alert user
        Logger.print("\nTest feature importance with best {} ML model:".format(program_part))
        fig = plt.figure(figsize=(DEFAULT_FIG_SIZE, DEFAULT_FIG_SIZE))
        # create a df to save r2 scores of simulations
        sim_results = pd.DataFrame()
        y_col = dataset.keys()[-1]
        for sim in range(simulations):
            # prepare data
            train_Xs, test_Xs, train_y, test_y = train_test_split(dataset.drop(y_col, axis=1),
                                                                  dataset[y_col],
                                                                  shuffle=True)
            # train & test model on data
            sim_model = model
            sim_model.fit(train_Xs, train_y)
            pred = sim_model.predict(test_Xs)
            sim_results.at[sim, 'r2'] = r2_score(test_y, pred)
            # check r2 loss on data without a specific feature
            for feature in train_Xs.keys():
                new_train_Xs, new_test_Xs = train_Xs.drop(feature, axis=1), test_Xs.drop(feature, axis=1)
                new_model = model
                new_model.fit(new_train_Xs, train_y)
                new_pred = new_model.predict(new_test_Xs)
                sim_results.at[sim, '{}_r2_loss'.format(feature)] = sim_results.loc[sim, 'r2'] - \
                                                                    r2_score(test_y, new_pred)
        # save all r2 scores
        sim_results.to_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), save_dir,
                                        "{}_feature_importance.csv".format(program_part)),
                           index=False)
        # prepare new df of avereged feature importance acc to r2 loss
        f_importance = pd.DataFrame()
        for i, f in enumerate(train_Xs.keys()):
            name = f + '_r2_loss'
            f_importance.at[i, 'feature'] = f
            f_importance.at[i, ['r2_loss', 'r2_err']] = sim_results[name].mean(), sim_results[name].std()
        # plot data
        f_importance.plot.barh(x='feature',
                               y='r2_loss',
                               xerr=f_importance['r2_err'].T.values,
                               color="grey")
        plt.ylabel("Feature Name", fontsize=16)
        plt.xlabel("$R^2$ Loss Without Feature", fontsize=16)
        ax = plt.gca()
        plt.grid(axis='x')
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), save_dir,
                                 "{}_feature_importance.pdf".format(program_part)),
                    dpi=DEFAULT_DPI)
        plt.close()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<Plotter>"
