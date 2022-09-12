# project imports
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# library imports
from utills.consts import *
from utills.logger_config import Logger
from data_generators.N_frequency_generator import NFrequencyGenerator


class SFFWDdataGenerator:
    """
    This class is responsible for the generation of the physical
    steady (no acceleration) free fall with drag to test the model.
    The motion is dictated by:
        0 = Weight - Buoyancy - 0.5*Cd*rhoa*V*V*area
    ->  Cd = (rhop - rhoa) * volume * g * 2/(rhoa*V*V*area)

    Variables in this file:
         g : gravitational acceleration        [m/s2]
      rhop : particle density                  [kg/m3]
         d : particle diameter                 [m]
      rhoa : fluid density                     [kg/m3]
         V : settling velocity                 [m/s]
        nu : kinematic viscosity of the fluid  [m2/s]
        Re : reynolds number                   [-]

    This class is responsible to produce a pandas DataFrame for three different model experiments.
    """

    def __init__(self):
        pass

    @staticmethod
    def generate_noiseless(samples: int,
                           rhoa_range: tuple,
                           rhop_range: tuple,
                           nu_range: tuple,
                           re_range: tuple,
                           show_progress_bar: bool = True):
        """
        Generates a pandas dataframe of experiments to represent steady free fall measurements.
        """
        rhoa_range_delta = rhoa_range[1] - rhoa_range[0]
        rhop_range_delta = rhop_range[1] - rhop_range[0]
        nu_range_delta = nu_range[1] - nu_range[0]
        re_range_delta = re_range[1] - re_range[0]
        data = []
        # generate samples
        pbar = tqdm(total=samples, desc="Generating baseline data") if show_progress_bar else None
        for sample_index in range(samples):
            # sample data from ranges
            rhoa = np.random.random_sample() * rhoa_range_delta + rhoa_range[0]
            nu = np.random.random_sample() * nu_range_delta + nu_range[0]
            re = np.random.random_sample() * re_range_delta + re_range[0]
            # calc rhop from rhoa with range
            rhop = rhoa + np.random.random_sample() * rhop_range_delta + rhop_range[0]
            # calc Cd from Re acc to known drag to Reynolds relation
            cd = 0.4 + 24.0 / re + 6.0 / (1 + re ** 0.5)
            # calc 'd' from the other parameters
            if rhop == rhoa:  # just to make sure we will not divide by zero
                raise ZeroDivisionError("Rhop can not be equal to rhoa")
            d = np.power((cd * re * re * nu * nu * rhoa) / (13.08 * (rhop - rhoa)), 1 / 3)
            # recalculate 'v'
            v = re * nu / d
            # add the data and alert the user by a progress bar, if needed
            data.append([rhoa, v, d, rhop, nu, cd])
            if show_progress_bar:
                pbar.update(1)

        if show_progress_bar:
            pbar.close()
        # make a Pandas.DataFrame and save it as a CSV file
        df = pd.DataFrame(data=data, columns=["rhoa", "V", "d", "rhop", "nu", "Cd"])
        df.to_csv(os.path.join(DATA_FOLDER,
                               "SFF_noiseless_baseline_{}_samples.csv".format(SFF_N_SAMPLES_STR)),
                  index=False)
        return df

    @staticmethod
    def generate_case_1(samples: int,
                        rhoa_range: tuple,
                        rhop_range: tuple,
                        nu_range: tuple,
                        re_range: tuple,
                        save_path: str,
                        dropped_param: str = SFF_1_DROP_PARAM,
                        force: bool = FORCE_DATA_OVERRIDE_FLAG):
        """
        Generate a pandas dataframe with only 3 our of 4 needed features to calc Cd.
        Saves the dataframe for model experiment.
        """
        baseline_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     DATA_FOLDER,
                                     "SFF_noiseless_baseline_{}_samples.csv".format(SFF_N_SAMPLES_STR))
        if os.path.exists(baseline_path) and not force:
            df = pd.read_csv(baseline_path)
        else:
            df = SFFWDdataGenerator.generate_noiseless(samples=samples,
                                                       rhoa_range=rhoa_range,
                                                       nu_range=nu_range,
                                                       re_range=re_range,
                                                       rhop_range=rhop_range)
        # alert user
        re = df["d"] * df["V"] / df["nu"]
        Logger.print('Generated data with Re: min={:.4} max={:.4}'.format(re.min(), re.max()))
        # build case
        df.drop([dropped_param, 'nu'], axis=1).to_csv(save_path, index=False)
        # alert user that feature selection isn't needed
        feature_indexes_ranges = "Not applicable"
        return feature_indexes_ranges

    @staticmethod
    def generate_case_2(samples: int,
                        rhoa_range: tuple,
                        rhop_range: tuple,
                        nu_range: tuple,
                        re_range: tuple,
                        save_path: str,
                        noise_range: tuple = SFF_CASE_2_NOISE_RANGE,
                        force: bool = FORCE_DATA_OVERRIDE_FLAG):
        """
        Generate a pandas dataframe with rhoa,V,d,rhop,Cd measurements.
        Adds noise to Cd, and saves the dataframe for model experiment.
        """
        baseline_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     DATA_FOLDER,
                                     "SFF_noiseless_baseline_{}_samples.csv".format(SFF_N_SAMPLES_STR))
        if os.path.exists(baseline_path) and not force:
            df = pd.read_csv(baseline_path)
        else:
            df = SFFWDdataGenerator.generate_noiseless(samples=samples,
                                                       rhoa_range=rhoa_range,
                                                       nu_range=nu_range,
                                                       re_range=re_range,
                                                       rhop_range=rhop_range)
        # alert user
        re = df["d"] * df["V"] / df["nu"]
        Logger.print('Generated data with Re: min={:.4} max={:.4}'.format(re.min(), re.max()))
        # add noise to target
        noise_range_delta = noise_range[1] - noise_range[0]
        noise = (np.random.random_sample() * noise_range_delta + noise_range[0]) * np.random.choice((-1, 1))
        df.Cd = [val * (1 + noise) for val in df["Cd"]]
        # build case
        df.drop("nu", axis=1).to_csv(save_path, index=False)
        # alert user that feature selection isn't needed
        feature_indexes_ranges = "Not applicable"
        return feature_indexes_ranges

    @staticmethod
    def generate_case_2_with_guess(samples: int,
                                   rhoa_range: tuple,
                                   rhop_range: tuple,
                                   nu_range: tuple,
                                   re_range: tuple,
                                   save_path: str,
                                   noise_range: tuple = SFF_CASE_2_NOISE_RANGE,
                                   force: bool = FORCE_DATA_OVERRIDE_FLAG):
        """
        Generate a pandas dataframe similar to that of casse 2.
        Adds two educated guesses (rhop-rhoa, and V^2),
        and saves the dataframe for model experiment.
        """
        baseline_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     DATA_FOLDER,
                                     "SFF_noiseless_baseline_{}_samples.csv".format(SFF_N_SAMPLES_STR))
        if os.path.exists(baseline_path) and not force:
            df = pd.read_csv(baseline_path)
        else:
            df = SFFWDdataGenerator.generate_noiseless(samples=samples,
                                                       rhoa_range=rhoa_range,
                                                       nu_range=nu_range,
                                                       re_range=re_range,
                                                       rhop_range=rhop_range)
        # alert user
        re = df["d"] * df["V"] / df["nu"]
        Logger.print('Generated data with Re: min={:.4} max={:.4}'.format(re.min(), re.max()))
        # add noise to target
        noise_range_delta = noise_range[1] - noise_range[0]
        noise = (np.random.random_sample() * noise_range_delta + noise_range[0]) * np.random.choice((-1, 1))
        df.Cd = [val * (1 + noise) for val in df["Cd"]]
        # build case
        delta_rho = df["rhop"] - df["rhoa"]
        df.insert(0, "delta_rho", delta_rho)
        vel_squared = df["V"] * df["V"]
        df.insert(0, "V^2", vel_squared)
        df.drop("nu", axis=1).to_csv(save_path, index=False)
        # alert user that feature selection isn't needed
        feature_indexes_ranges = "Not applicable"
        return feature_indexes_ranges

    @staticmethod
    def generate_case_3(samples: int,
                        rhoa_range: tuple,
                        rhop_range: tuple,
                        nu_range: tuple,
                        re_range: tuple,
                        save_path: str,
                        force: bool = FORCE_DATA_OVERRIDE_FLAG):
        """
        Generate a pandas dataframe with rhoa,V,d,rhop,nu,Cd measurements.
        Uses the dataframe to create a dataframe of dimensionless features,
        and saves it for model experiment.
        """
        baseline_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     DATA_FOLDER,
                                     "SFF_noiseless_baseline_{}_samples.csv".format(SFF_N_SAMPLES_STR))
        if os.path.exists(baseline_path) and not force:
            df = pd.read_csv(baseline_path)
        else:
            df = SFFWDdataGenerator.generate_noiseless(samples=samples,
                                                       rhoa_range=rhoa_range,
                                                       nu_range=nu_range,
                                                       re_range=re_range,
                                                       rhop_range=rhop_range)
        # alert user
        re = df["d"] * df["V"] / df["nu"]
        Logger.print('Generated data with Re: min={:.4} max={:.4}'.format(re.min(), re.max()))
        # add all possible N frequency combinations
        df, n_suffix = NFrequencyGenerator.add_all_combos(df=df,
                                                          g=g_force)
        # create non-dimensional features
        features = pd.DataFrame()
        # density ratio:
        features["rhop/rhoa"] = df["rhop"] / df["rhoa"]
        # density delta ratio:
        features["delta_rho/rhoa"] = (df["rhop"] - df["rhoa"]) / df["rhoa"]
        # Reynolds number:
        features["Re"] = re
        # Unknown number - nu*g/V**3:
        features["nu*g/V**3"] = g_force * df["nu"] / df["V"] ** 3
        # Unknown number - g*d/V**2:
        features["g*d/V**2"] = g_force * df["d"] / df["V"] ** 2
        # add 4 more groups of features
        for suff in n_suffix:
            # Froude number:
            features["Fr{}".format(suff)] = df["V"] / (df["d"] * df["N{}".format(suff)])
            # Froude number from acceleration - g/(V*N_i):
            features["AccFr{}".format(suff)] = g_force / (df["V"] * df["N{}".format(suff)])
            # Unknown number (Num1) - g*d/(nu*N_i)
            features["1Num{}".format(suff)] = g_force * df["d"] / (df["nu"] * df["N{}".format(suff)])
            # Unknown number (Num2) - V*V/(nu*N_i)
            features["2Num{}".format(suff)] = df["V"] * df["V"] / (df["nu"] * df["N{}".format(suff)])
        # reorder column names to be in groups
        features = features[sorted(features.keys(), key=lambda x: x[0])]
        # set index ranges for all 9 feature groups
        feature_indexes_ranges = []
        index = 0
        for i in range(9):
            if i < 4:
                feature_indexes_ranges.append([index, index + len(n_suffix) - 1])
                index += len(n_suffix)
            else:
                feature_indexes_ranges.append([index, index])
                index += 1
        # add target
        features["Cd"] = df["Cd"]
        # save the result to a csv file
        features.to_csv(save_path, index=False)

        return feature_indexes_ranges
