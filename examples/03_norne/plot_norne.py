import os
import numpy as np
import matplotlib.pyplot as plt

import tpysa

dir_name = os.path.dirname(__file__)

# res = np.array(
#     [
#         6.44e-03,
#         2.56e-03,
#         6.45e-04,
#         4.92e-04,
#         1.50e-04,
#         9.78e-05,
#         4.26e-05,
#         3.00e-05,
#         2.07e-05,
#         1.53e-05,
#         4.60e-06,
#     ]
# )


# plt.semilogy(
#     np.arange(len(res)) + 1,
#     res,
# )
# ax = plt.gca()
# ax.set_xlabel("Iteration")
# ax.set_ylabel("Relative Residual")

# plt.show()


res_truth = np.array(
    [
        5.69e-01,
        3.23e-01,
        1.83e-01,
        1.34e-01,
        6.00e-02,
        3.38e-02,
        1.93e-02,
        1.09e-02,
        6.22e-03,
        3.54e-03,
        2.03e-03,
    ]
)
res_truth_lagged = 2.04e-01

contract = res_truth[1:] / res_truth[:-1]
mean_cont = np.mean(contract)

plt.rcParams.update({"font.size": 13})

plt.figure()
plt.semilogy(np.arange(len(res_truth)) + 1, res_truth, "o-")
plt.semilogy(
    np.arange(len(res_truth)) + 1,
    np.full_like(res_truth, res_truth_lagged),
)
plt.grid(True, which="both", ls="-", color="0.65")


plt.legend(("Iterative coupling", "Lagged coupling"))
ax = plt.gca()
ax.set_xlabel("Iteration")
ax.set_ylabel("Relative error in source terms")
ax.set_box_aspect(1)

fig = plt.gcf()
fig.set_size_inches(7, 7)
plt.savefig(os.path.join(dir_name, "convergence_spacetime.pdf"), bbox_inches="tight")

plt.show()


# res = np.array(
#     [
#         1.00e00,
#         3.62e-01,
#         1.71e-01,
#         1.24e-01,
#         9.41e-02,
#         2.73e-02,
#         1.48e-02,
#         8.50e-03,
#         4.74e-03,
#         2.71e-03,
#         1.53e-03,
#         8.70e-04,
#         4.97e-04,
#         3.44e-04,
#         1.97e-04,
#         9.15e-05,
#         1.40e-04,
#         6.25e-05,
#         2.11e-04,
#         2.25e-04,
#         2.36e-04,
#     ]
# )
