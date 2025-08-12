import numpy as np
import matplotlib.pyplot as plt

n_x = [5, 10, 20, 40]
err_u = [1.73e-01, 6.23e-02, 1.74e-02, 4.49e-03]
err_r = [3.50e-01, 1.18e-01, 3.36e-02, 9.23e-03]
err_p = [5.86e-03, 2.46e-03, 7.32e-04, 1.93e-04]

h_max = np.max([np.max(err_u), np.max(err_r), np.max(err_p)])
h = 1 / np.array(n_x)
h *= h_max / h.max()

h_squared = h**2
h_squared *= h_max / h_squared.max()

plt.loglog(n_x, err_u, n_x, err_r, n_x, err_p, n_x, h_squared, "--")
plt.legend(("displacement", "rotation", "solid pressure", "O($h^2$)"))
plt.grid(True, which="both", ls="-", color="0.65")

ax = plt.gca()
ax.set_xlabel("n_x")
ax.set_ylabel("errors")

plt.savefig("convergence_elasticity.svg")
plt.show()
