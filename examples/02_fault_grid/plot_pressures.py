import os
from opm.io.ecl import ESmry
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

smspec_files = [
    "/home/AD.NORCERESEARCH.NO/wibo/TPySA/examples/02_fault_grid/sol_decoupled/FAULT.SMSPEC",
    "/home/AD.NORCERESEARCH.NO/wibo/TPySA/examples/02_fault_grid/sol_fixedpoint/FAULT.SMSPEC",
    "/home/AD.NORCERESEARCH.NO/wibo/TPySA/examples/02_fault_grid/sol_lagged/FAULT.SMSPEC",
]
markers = [".", "x", "o"]
markers = ["", "", "."]
colors = ["blue", "orange", "green"]
zorders = [10, 5, 0]

plt.figure()
plt.rcParams.update({"font.size": 13})

for spec, mark, color, zorder in zip(smspec_files, markers, colors, zorders):
    smry = ESmry(spec)
    time = smry["TIME"]
    pressure = smry["RPR:1"]

    plt.plot(time, pressure, marker=mark, color=color, zorder=zorder)

for spec, mark, color, zorder in zip(smspec_files, markers, colors, zorders):
    smry = ESmry(spec)
    time = smry["TIME"]
    pressure = smry["RPR:2"]

    plt.plot(time, pressure, marker=mark, linestyle="--", color=color, zorder=zorder)

method_legend = plt.legend(
    (
        "No mechanics",
        "Fixed point",
        "Lagged",
    )
)

plt.gca().add_artist(method_legend)

# Legend for instances (linestyle-coded)
instances = ["$\Omega_1$", "$\Omega_2$"]
instance_lines = [Line2D([0], [0], color="black", linestyle=ls) for ls in ["-", "--"]]
plt.legend(instance_lines, instances, loc="right")

ax = plt.gca()
ax.set_xlabel("Time (Days)")
ax.set_ylabel("Average fluid pressure (Bar)")
ax.set_box_aspect(1)

fig = plt.gcf()
fig.set_size_inches(7, 7)
plt.savefig(
    os.path.join(os.path.dirname(__file__), "average_pressure.pdf"), bbox_inches="tight"
)

plt.show()
pass
