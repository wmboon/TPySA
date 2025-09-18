import numpy as np
import os


def extract_psi_array(scheme: str, num_psi: int):
    psi_list = []
    psi_out_list = []
    for i in range(num_psi):
        file = os.path.join(
            os.path.dirname(__file__),
            scheme,
            "source_{:}.npz".format(i),
        )
        psi_list.append(np.load(file)["psi"])

        file = os.path.join(
            os.path.dirname(__file__),
            scheme,
            "source_{:}_out.npz".format(i),
        )
        psi_out_list.append(np.load(file)["psi"])

        pass

    true_dict = np.load(
        os.path.join(
            os.path.dirname(__file__),
            scheme,
            "source_true.npz".format(i),
        )
    )
    psi_true = true_dict["psi"].reshape((-1, 44431))

    return np.array(psi_list), np.array(psi_out_list), psi_true


and_psi, and_out, and_true = extract_psi_array("anderson", 8)
and_norm = np.linalg.norm(and_true)

fix_psi, fix_out, fix_true = extract_psi_array("fixed_point", 8)
fix_norm = np.linalg.norm(fix_true)

lagged = np.load(os.path.join(os.path.dirname(__file__), "lagged_source.npz"))["psi"]
lagged = lagged[2:]

and_norms = np.array([np.linalg.norm(psi - and_true) for psi in and_psi]) / and_norm
fix_norms = np.array([np.linalg.norm(psi - and_true) for psi in fix_psi]) / and_norm
# fix_norms = 0.36 ** (np.arange(len(fix_psi)))
lag_norms = np.full(len(fix_norms), np.linalg.norm(lagged - and_true)) / and_norm


fix_rate = fix_norms[1:] / fix_norms[:-1]
and_rate = and_norms[1:] / and_norms[:-1]

import matplotlib.pyplot as plt

plt.figure()
plt.rcParams.update({"font.size": 13})

plt.semilogy(np.arange(len(fix_norms)), lag_norms, "o-")
plt.semilogy(np.arange(len(fix_norms)), fix_norms, "o-")
plt.semilogy(np.arange(len(and_norms)), and_norms, "o-")


plt.grid(True, which="both", ls="-", color="0.65")
plt.legend(("Lagged", "Fixed point", "Anderson"))

ax = plt.gca()
ax.set_xlabel("Iteration")
ax.set_ylabel("Relative error in source terms")
ax.set_box_aspect(1)

fig = plt.gcf()
fig.set_size_inches(7, 7)
plt.savefig(
    os.path.join(os.path.dirname(__file__), "convergence_spacetime.pdf"),
    bbox_inches="tight",
)
plt.show()
pass
