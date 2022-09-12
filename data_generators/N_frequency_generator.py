# project imports
import numpy as np
import pandas as pd

# project imports
from utills.consts import *


class NFrequencyGenerator:
    """
    This class is responsible for creating all possible combinations
    representing a fluid frequency N [1/s].
    """

    def __init__(self):
        pass

    @staticmethod
    def add_all_combos(df: pd.DataFrame,
                       g: float):
        """
        Single entry point.
        Adds all possible combination of N as new columns in a given df.
        returns the modified df and list os suffixes that indicate
        how N was calculated.
        """
        for suff in N_FREQ_SUFFIX:
            # choose rho_up
            if suff[-2] == '1':
                rho_up = df["rhop"] - df["rhoa"]
            elif suff[-2] == '2':
                rho_up = df["rhop"]
            else:
                rho_up = df["rhoa"]

            # choose rho_down
            if suff[-1] == '1':
                rho_down = 0.5 * (df["rhop"] + df["rhoa"])
            elif suff[-1] == '2':
                rho_down = df["rhop"]
            else:
                rho_down = df["rhoa"]

            # calc N and add to Ns
            df.at[:, "N{}".format(suff)] = np.sqrt((g * rho_up) / (df["d"] * rho_down))
        # return answer
        return df, N_FREQ_SUFFIX
