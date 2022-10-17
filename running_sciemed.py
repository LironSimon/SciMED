"""
1. Needed library imports
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
"""
2. Import SciMED's instance
"""
from scimed import scimed
"""
3. Load the data into a pandas.DataFrame and split it into the source and target features 
"""


def run():
    df = pd.read_csv("412_dataset.csv")
    Y_COL_NAME = "recovery_time*N_Verso"
    x = df.drop([Y_COL_NAME], axis=1)
    y = df[Y_COL_NAME]
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.2,   # most of the time, we divide 80%-20%
                                                        random_state=73) # Sheldon's number - just for fun
    feature_indexes_ranges =[[0,0], [1,4], [5,20], [21,21], [22,27], [28,412]]
    """
    4. Run SciMED and observe the results in the 'results_folder'
    """
    scimed.run(train_data_x=x_train,
               train_data_y=y_train,
               test_data_x=x_test,
               test_data_y=y_test,
               results_folder=os.path.join(os.path.dirname(__file__), "results"),
               k_fold = 5,
               numerical_bool = True,
               numerical_run_times = 1,
               numerical_generations = 25,
               numerical_population = 40,
               analytical_bool = False,
               force_ebs_bool = True,
               ebs_size_range = (3, 13),
               feature_indexes_ranges = feature_indexes_ranges,
               feature_selection_generations = 30,
               feature_selection_pop_size = 26,
               feature_selection_mutation_rate = 0.03,
               feature_selection_royalty=0.05
               )


if __name__ == '__main__':
    run()
