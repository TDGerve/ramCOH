"""
This module holds some default baseline interpolation regions for different phases
"""

import numpy as np

default_birs = {
    "CO2": np.array(
        [[1000, 1260], [1270, 1275], [1300, 1375], [1396, 1402], [1418, 1500]]
    ),
    "glass": np.array([[200, 300], [640, 655], [800, 810], [1220, 2300], [3750, 4000]]),
    "neon": np.array(
        [
            [1027, 1108],
            [1118, 1213],
            [1221, 1303],
            [1312, 1390],
            [1401, 1435],
            [1450, 1464],
        ]
    ),
    "olivine": np.array(
        [
            [100, 185],
            [260, 272],
            [370, 380],
            [470, 515],
            [660, 740],
            [1050, 4000],
        ]
    ),
}
