import numpy as np
import os

import tpysa

key = "pres"
lag_dict = np.load(
    os.path.join(
        os.path.dirname(__file__),
        "sol_lagged",
        "lagged_source.npz",
    )
)
lag = lag_dict[key]

fix_list, fix_true = tpysa.extract_arrays("fixed_point", 10, key, __file__)
and_list, and_true = tpysa.extract_arrays("anderson", 10, key, __file__)

tpysa.plot_spacetime_convergence(__file__, fix_list, and_list, lag, and_true, "fault")
