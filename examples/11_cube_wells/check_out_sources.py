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

    # true_dict = np.load(
    #     os.path.join(
    #         os.path.dirname(__file__),
    #         scheme,
    #         "source_true.npz".format(i),
    #     )
    # )
    # psi_true = true_dict["psi"].reshape((-1, 44431))

    return np.array(psi_list), np.array(psi_out_list), psi_list[-1]


and_psi, and_out, and_true = extract_psi_array("anderson", 12)
and_norm = np.linalg.norm(and_true)
and_norms = np.array([np.linalg.norm(psi - and_true) for psi in and_psi]) / and_norm

fix_psi, fix_out, fix_true = extract_psi_array("fixed_point", 3)
fix_norm = np.linalg.norm(fix_true)
fix_norms = np.array([np.linalg.norm(psi - fix_true) for psi in fix_psi]) / fix_norm

import matplotlib.pyplot as plt

plt.semilogy(np.arange(len(and_norms)), and_norms)
plt.semilogy(np.arange(len(fix_norms)), fix_norms)
plt.legend(("Anderson", "Fixed point"))
plt.show()
pass
