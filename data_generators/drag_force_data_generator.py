# project imports
import numpy as np
import pandas as pd

# library imports


class DragForceDataGenerator:
    """
    This class generates measurements of aerodynamic drag force Fd, 
    exerted on a sphere.
    """

    # CONSTS #

    # END - CONSTS #

    def __init__(self):
        pass

    # Logic - start #

    @staticmethod
    def generate(samples: int,
                 cd_range: tuple,
                 rhoa_range: tuple,
                 v_range: tuple,
                 d_range: tuple,
                 noise_range: tuple,
                 save_path: str,
                 rhop_range: tuple=(0.15,0.4),
                 L_range: tuple=(0.15,0.4),
                 p1_range: tuple=(15,35),
                 p2_range: tuple=(40,60),
                 p3_range: tuple=(1e-4,2e-4),
                 p4_range: tuple=(200,300),
                 p5_range: tuple=(1,3),
                 p6_range: tuple=(1000,2500)):
        """
        Generate a pandas dataframe of experiments to represent the measurments.
        We assume we sample 4 parameters:
            cd: drag coefficient on a sphere    [-]
            rhoa: air density                   [kg/m3]
            v: momentary velocity of the sphere [m/s]
            d: diameter of the sphere           [m]
        and calculate with them:
            fd: drag exerted on the sphere      [kg*m/s2]
        via fd = pi*cd*rho*(v**2)*(d**2)/8.
        Measurements include additional 8 parameters
        that are not included within this function.
        """
        cd_range_delta = cd_range[1] - cd_range[0]
        rhoa_range_delta = rhoa_range[1] - rhoa_range[0]
        v_range_delta = v_range[1] - v_range[0]
        d_range_delta = d_range[1] - d_range[0]
        rhop_range_delta = d_range[1] - d_range[0]
        L_range_delta = L_range[1] - L_range[0]
        p1_range_delta = p1_range[1] - p1_range[0]
        p2_range_delta = p2_range[1] - p2_range[0]
        p3_range_delta = p3_range[1] - p3_range[0]
        p4_range_delta = p4_range[1] - p4_range[0]
        p5_range_delta = p5_range[1] - p5_range[0]
        p6_range_delta = p6_range[1] - p6_range[0]
        noise_range_delta = noise_range[1] - noise_range[0]
        data = []
        # generate samples
        for i in range(samples):
            cd = round(np.random.random_sample() * cd_range_delta + cd_range[0], 2)
            rho = round(np.random.random_sample() * rhoa_range_delta + rhoa_range[0], 2)
            v = round(np.random.random_sample() * v_range_delta + v_range[0], 2)
            d = round(np.random.random_sample() * d_range_delta + d_range[0], 2)
            rhop = round(np.random.random_sample() * rhop_range_delta + rhop_range[0], 2)
            L = round(np.random.random_sample() * L_range_delta + L_range[0], 2)
            p1 = round(np.random.random_sample() * p1_range_delta + p1_range[0], 2)
            p2 = round(np.random.random_sample() * p2_range_delta + p2_range[0], 2)
            p3 = round(np.random.random_sample() * p3_range_delta + p3_range[0], 2)
            p4 = round(np.random.random_sample() * p4_range_delta + p4_range[0], 2)
            p5 = round(np.random.random_sample() * p5_range_delta + p5_range[0], 2)
            p6 = round(np.random.random_sample() * p6_range_delta + p6_range[0], 2)
            noise = round(np.random.random_sample() * noise_range_delta + noise_range[0], 2) * np.random.choice((-1, 1))
            fd_sampled = np.pi*cd*rho*(v**2)*(d**2)/8
            fd_sampled = round(fd_sampled * (1 + noise), 2)
            data.append([cd, rho, rhop, v, d, L, p1, p2, p3, p4, p5, p6, fd_sampled])
        # make a Pandas.DataFrame and save it as a CSV file
        pd.DataFrame(data=data, columns=["Cd","rhoa","rhop","v","d","L","p1","p2",
                                         "p3","p4","p5","p6","Fd"]).to_csv(save_path, index=False)
        #return indices of feature groups: rhoa-rhop and d-l form a group. The rest do not have a selection option
        return [[0,0],[1,2],[2,2],[3,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,11]] 

    # Logic - end #
