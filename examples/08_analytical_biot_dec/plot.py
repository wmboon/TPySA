import os
import numpy as np
import matplotlib.pyplot as plt

dir_name = os.path.dirname(__file__)

with open(os.path.join(dir_name, "errs.txt")) as f:
    str = f.read()


err_table = np.array(str.split()).astype(float)
err_table = np.reshape(err_table, (-1, 5))
n_x = err_table[:, 0]
errs = err_table[:, 1:]


h_init = np.prod(errs[0]) ** (1 / errs.shape[1])
h_squared = 1 / np.array(n_x) ** 2
h_squared *= h_init / h_squared.max()

plt.loglog(n_x, errs, "*-", n_x, h_squared, "--")
plt.legend(("Displacement", "Rotation", "Solid pressure", "Fluid pressure", "O($h^2$)"))
plt.grid(True, which="both", ls="-", color="0.65")

ax = plt.gca()
ax.set_xlabel("$1 / h$")
ax.set_ylabel("Relative $L^2$ error")

plt.savefig(os.path.join(dir_name, "convergence_biot.svg"))
plt.show()
