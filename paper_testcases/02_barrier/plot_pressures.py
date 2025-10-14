import os
from opm.io.ecl import ESmry
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

smspec_files = [
    "/home/AD.NORCERESEARCH.NO/wibo/TPySA/examples/02_fault_grid/sol_decoupled/FAULT.SMSPEC",
    "/home/AD.NORCERESEARCH.NO/wibo/TPySA/examples/02_fault_grid/sol_fixedpoint/FAULT.SMSPEC",
    "/home/AD.NORCERESEARCH.NO/wibo/TPySA/examples/02_fault_grid/sol_lagged/FAULT.SMSPEC",
]
markers = ["", "", ""]
colors = ["purple", "tab:orange", "tab:blue"]
zorders = [10, 5, 0]

RPRs = ["RPR:1", "RPR:2"]
linestyles = ["-", "--"]

fig = plt.figure()
plt.rcParams.update({"font.size": 13})

for RPR, ls in zip(RPRs, linestyles):
    for spec, mark, color, zorder in zip(smspec_files, markers, colors, zorders):
        smry = ESmry(spec)
        time = smry["TIME"]
        pressure = smry[RPR]

        plt.plot(time, pressure, marker=mark, linestyle=ls, color=color, zorder=zorder)

# Add legend
method_legend = plt.legend(
    (
        "No mechanics",
        "Fixed stress",
        "Lagged",
    )
)

plt.gca().add_artist(method_legend)

# Legend for instances (linestyle-coded)
subdomains = ["$\Omega_1$", "$\Omega_2$"]
subdomain_lines = [Line2D([0], [0], color="black", linestyle=ls) for ls in linestyles]
plt.legend(subdomain_lines, subdomains, loc="right")

# Label axes
ax = plt.gca()
ax.set_xlabel("Time (days)")
ax.set_ylabel("Average fluid pressure (bar)")
# ax.set_box_aspect(1)

fig.set_size_inches(7, 7)
plt.savefig(
    os.path.join(os.path.dirname(__file__), "average_pressure_fault.pdf"),
    bbox_inches="tight",
)

# plt.show()
