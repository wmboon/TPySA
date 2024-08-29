import os
import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)

case_str = "data_single_phase/SINGLE_PHASE"
dir_name = os.path.dirname(__file__)
opmcase = os.path.join(dir_name, case_str)

dir_lag = np.load(opmcase + "lagged.npz")
p_lag = dir_lag["p"]
t_lag = dir_lag["t"]

dir_ite = np.load(opmcase + "iterative.npz")
p_ite = dir_ite["p"]
t_ite = dir_ite["t"]

plt.plot(t_lag, p_lag, t_ite, p_ite)
plt.show()

# plt.figure(1)
# ax = plt.figure(1).add_subplot(projection="3d")
# x, y, z = grid.cell_centers
# u_x, u_y, u_z = displ.reshape((3, -1), order="F")
# ax.quiver(x, y, z, u_x, u_y, u_z)

# plt.show()

pass
