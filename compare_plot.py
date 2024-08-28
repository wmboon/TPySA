import os
import numpy as np
import matplotlib.pyplot as plt

plt.figure(0)

case_str = "data_single_phase/SINGLE_PHASE"
dir_name = os.path.dirname(__file__)
opmcase = os.path.join(dir_name, case_str)


dir_lagged = np.load(opmcase + "lagged.npz")
dir_iterative = np.load(opmcase + "iterative.npz")

t_lag = dir_lagged["t"]
p_lag = dir_lagged["p"]
t_ite = dir_iterative["t"]
p_ite = dir_iterative["p"]


plt.plot(t_lag, p_lag, t_ite, p_ite)
plt.show()

pass
