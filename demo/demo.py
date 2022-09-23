"""
This file provides a demo on how to use SciMed on a captured datasets that is stored in a CSV file format
"""

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
df = pd.read_csv("demo.csv")
Y_COL_NAME = ""
x = df.drop([Y_COL_NAME], axis=1)
y = df[Y_COL_NAME]
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2, # most of the time, we divide 80%-20%
                                                    random_state=73) # Sheldon's number - just for fun
"""
4. Run SciMED and observe the results in the 'results_folder'
"""
scimed.run(train_data_x=x_train,
           train_data_y=y_train,
           test_data_x=x_test,
           test_data_y=y_test,
           results_folder=os.path.join(os.path.dirname(__file__), "results")
           )