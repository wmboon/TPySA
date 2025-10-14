import numpy as np
import matplotlib.pyplot as plt

res = np.array(
    [
        1.00e00,
        1.15e-01,
        2.18e-02,
        4.50e-03,
        9.65e-04,
    ]
)


plt.semilogy(
    np.arange(len(res)) + 1,
    res,
)
ax = plt.gca()
ax.set_xlabel("Iteration")
ax.set_ylabel("Update")

plt.savefig("space_time_cart.svg")
plt.show()
