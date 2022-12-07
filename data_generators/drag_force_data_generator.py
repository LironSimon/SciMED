# project imports
import numpy as np
import pandas as pd

# library imports


class DragForce:
    """
    This class is responsible for the generation of measurements of aerodynamic 
    drag force Fd, exerted on a sphere.
    """

    # CONSTS #

    # END - CONSTS #

    def __init__(self):
        pass

    # Logic - start #

    @staticmethod
    def generate(samples: int,
                 cd_range: tuple,
                 rho_range: tuple,
                 v_range: tuple,
                 d_range: tuple,
                 noise_range: tuple,
                 save_path: str):
        """
        Generate a pandas dataframe of experiments to represent the measurments.
        We assume we sample 4 parameters:
            cd: drag coefficient on a sphere    [-]
            rho: sphere density                 [kg/m3]
            v: momentary velocity of the sphere [m/s]
            d: diameter of the sphere           [m]
        and calculate with them:
            fd: drag exerted on the sphere      [kg*m/s2]
        via fd = pi*cd*rho*(v**2)*(d**2)/8.
        """
        cd_range_delta = cd_range[1] - cd_range[0]
        rho_range_delta = rho_range[1] - rho_range[0]
        v_range_delta = v_range[1] - v_range[0]
        d_range_delta = d_range[1] - d_range[0]
        noise_range_delta = noise_range[1] - noise_range[0]
        data = []
        # generate samples
        for i in range(samples):
            cd = round(np.random.random_sample() * cd_range_delta + cd_range[0], 2)
            rho = round(np.random.random_sample() * rho_range_delta + rho_range[0], 2)
            v = round(np.random.random_sample() * v_range_delta + v_range[0], 2)
            d = round(np.random.random_sample() * d_range_delta + d_range[0], 2)
            noise = round(np.random.random_sample() * noise_range_delta + noise_range[0], 2) * np.random.choice((-1, 1))
            fd_sampled = np.pi*cd*rho*(v**2)*(d**2)/8
            fd_sampled = round(fd_sampled * (1 + noise), 2)
            data.append([cd, rho, v, d, fd_sampled])
        # make a Pandas.DataFrame and save it as a CSV file
        pd.DataFrame(data=data, columns=["Cd", "rho", "v", "d", "Fd"]).to_csv(save_path, index=False)

    # Logic - end #
