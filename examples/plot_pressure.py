import matplotlib.pyplot as plt
import numpy as np
import os

nx = 10
case_str = "cartgrid/GRID_" + str(nx)
dir_name = os.path.dirname(__file__)
opmcase = os.path.join(dir_name, case_str)

plt.figure(1)

lag_dict = np.load("_".join((opmcase, "0", "lagged.npz")))
p_zero, t_zero = lag_dict.values()
plt.plot(t_zero, p_zero, label="No mechanics")

lag_dict = np.load("_".join((opmcase, "1", "lagged.npz")))
p_lagged, t_lagged = lag_dict.values()
plt.plot(t_lagged, p_lagged, "--", label="Lagged coupling")

lag_dict = np.load("_".join((opmcase, "1", "iterative.npz")))
p_iter, t_iter = lag_dict.values()
plt.plot(t_iter, p_iter, "*", label="Iterative coupling")

plt.xlabel("Days")
plt.ylabel("Pressure (Bar)")
plt.legend()
plt.savefig(opmcase + "well_pressures.svg")

# plt.show()
pass
