import numpy as np
import matplotlib.pyplot as plt

res = np.array(
    [
        6.44e-03,
        2.56e-03,
        6.45e-04,
        4.92e-04,
        1.50e-04,
        9.78e-05,
        4.26e-05,
        3.00e-05,
        2.07e-05,
        1.53e-05,
        4.60e-06,
    ]
)


plt.semilogy(
    np.arange(len(res)) + 1,
    res,
)
ax = plt.gca()
ax.set_xlabel("Iteration")
ax.set_ylabel("Relative Residual")

plt.show()


res = np.array(
    [
        7.58e-04,
        1.72e-07,
        1.58e-08,
        1.74e-07,
        7.38e-08,
        1.22e-08,
        6.90e-08,
        7.39e-08,
    ]
)
plt.figure()
plt.semilogy(
    np.arange(len(res)) + 1,
    res,
)
ax = plt.gca()
ax.set_xlabel("Iteration")
ax.set_ylabel("Update")

plt.show()
