import csv
import os
import numpy as np

dir_name = os.path.dirname(__file__)
input = os.path.join(
    dir_name,
    "converged_anderson",
    "NORNE_TPSA.INFO",
)
tag = "Norne"


def plot_bicgstab_residuals(file, info_file, tag):
    res_list = []
    with open(info_file, "r", newline="") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=" ")

        res_i = []
        track = False

        for row in csv_reader:
            if len(row) > 1:
                if row[0] == "BiCGStab":
                    res_list.append(np.array(res_i))
                    res_i = []
                    track = False
                if row[0] == "Iter":
                    track = True
                    continue
            if track:
                res_i.append(float(row[-1]))

    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 13})
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [9, 1]}
    )

    plt.subplots_adjust(hspace=0.0)

    for res in res_list:
        ax1.semilogy(1 + np.arange(len(res)), res, ".-")

    ax1.grid(True, which="both", ls="-", color="0.65")
    ax1.set_xlim(0)
    ax1.set_ylabel("Relative residual")

    lengths = [len(res) for res in res_list[1:]]
    ax2.boxplot(
        lengths,
        orientation="horizontal",
        widths=0.6,
    )
    ax2.set_xlabel("BiCGStab iteration")
    ax2.set_yticks([])

    fig.set_size_inches(7, 7)
    plt.savefig(
        os.path.join(os.path.dirname(file), "convergence_BiCGStab_{:}.pdf".format(tag)),
        bbox_inches="tight",
    )
    return res_list


plot_bicgstab_residuals(__file__, input, tag)
