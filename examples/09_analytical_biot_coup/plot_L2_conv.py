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
h = 1 / np.array(n_x)
h *= h_init / h.max()

h_squared = 1 / np.array(n_x) ** 2
h_squared *= h_init / h_squared.max()

plt.rcParams.update({"font.size": 13})

plt.loglog(n_x, errs, "o-", n_x, h_squared, "--")
plt.legend(
    (
        "Solid displacement",
        "Solid rotation",
        "Solid pressure",
        "Fluid pressure",
        "O($h^2$)",
    )
)
plt.grid(True, which="both", ls="-", color="0.65")

ax = plt.gca()
ax.set_xlabel("$1 / h$")
ax.set_ylabel("Relative $L^2$ error")
# ax.set_box_aspect(1)

fig = plt.gcf()
fig.set_size_inches(7, 7)
plt.savefig(os.path.join(dir_name, "convergence_TPSA_L2.pdf"), bbox_inches="tight")
plt.show()
