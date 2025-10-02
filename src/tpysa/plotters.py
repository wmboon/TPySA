import os
import numpy as np


def extract_arrays(
    scheme: str,
    num_psi: int,
    key: str,
    file_in: str,
):
    out_list = []

    for i in range(num_psi):
        file = os.path.join(
            os.path.dirname(file_in),
            scheme,
            "source_{:}_out.npz".format(i),
        )
        out_dict = np.load(file)
        out_list.append(out_dict[key])

    true_dict = np.load(
        os.path.join(
            os.path.dirname(file_in),
            scheme,
            "source_true.npz".format(i),
        )
    )
    truth = true_dict[key]

    return np.array(out_list), truth


def plot_spacetime_convergence(file, fix_list, and_list, lag, truth, tag):
    true_norm = np.linalg.norm(truth)

    fix_norms = np.array([np.linalg.norm(p_i - truth) for p_i in fix_list]) / true_norm
    and_norms = np.array([np.linalg.norm(p_i - truth) for p_i in and_list]) / true_norm
    lag_norms = np.linalg.norm(lag - truth) / true_norm

    import matplotlib.pyplot as plt

    plt.figure()
    plt.rcParams.update({"font.size": 13})

    plt.semilogy(1 + np.arange(len(fix_norms)), np.full_like(fix_norms, lag_norms), "-")
    plt.semilogy(1 + np.arange(len(fix_norms)), fix_norms, "o-")
    plt.semilogy(1 + np.arange(len(and_norms)), and_norms, "o-")

    plt.grid(True, which="both", ls="-", color="0.65")
    plt.legend(("Lagged", "Fixed stress", "Anderson"))

    ax = plt.gca()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative error in fluid pressure")
    # ax.set_box_aspect(1)

    fig = plt.gcf()
    fig.set_size_inches(7, 7)
    plt.savefig(
        os.path.join(
            os.path.dirname(file), "convergence_spacetime_{:}.pdf".format(tag)
        ),
        bbox_inches="tight",
    )
