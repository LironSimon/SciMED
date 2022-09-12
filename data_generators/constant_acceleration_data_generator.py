# project imports
import numpy as np
import pandas as pd

# library imports


class ConstantAccelerationDataGenerator:
    """
    This class is responsible for the generation of measurements of motion with
     constant acceleration to test the model.
    """

    # CONSTS #

    # END - CONSTS #

    def __init__(self):
        pass

    # Logic - start #

    @staticmethod
    def generate(samples: int,
                 a_range: tuple,
                 t_range: tuple,
                 v0_range: tuple,
                 noise_range: tuple,
                 save_path: str):
        """
        Generate a pandas dataframe of experiments to represent motion with constant acceleration.
        We assume we sample 3 parameters:
            v0: initial velocity   [m/s]
             a: acceleration      [m/s2]
             t: time pass         [s]
        and calculate with them:
             v: velocity at time t [m/s]
        via v = v0 + a * t.
        """
        a_range_delta = a_range[1] - a_range[0]
        t_range_delta = t_range[1] - t_range[0]
        v0_range_delta = v0_range[1] - v0_range[0]
        noise_range_delta = noise_range[1] - noise_range[0]
        data = []
        # generate samples
        for i in range(samples):
            a = round(np.random.random_sample() * a_range_delta + a_range[0], 2)
            v0 = round(np.random.random_sample() * v0_range_delta + v0_range[0], 2)
            t = round(np.random.random_sample() * t_range_delta + t_range[0], 2)
            noise = round(np.random.random_sample() * noise_range_delta + noise_range[0], 2) * np.random.choice((-1, 1))
            v_sampled = v0 + a * t
            v_sampled = round(v_sampled * (1 + noise), 2)
            data.append([v0, a, t, v_sampled])
        # make a Pandas.DataFrame and save it as a CSV file
        pd.DataFrame(data=data, columns=["v0", "a", "t", "v"]).to_csv(save_path, index=False)

    # Logic - end #
