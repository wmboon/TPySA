import numpy as np
import os
import csv
import matplotlib.pyplot as plt


h_list = [16, 25, 37, 56]

dirname = os.path.dirname(__file__)
dof_list = []
t_list = []

for h in h_list:
    info_file = os.path.join(dirname, "GRID_{:}_TPSA.INFO".format(h))
    with open(info_file, "r", newline="") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=" ")

        time_i = []

        for row in csv_reader:
            if len(row) > 1:
                if row[0] == "Assembled":
                    ndof = int(row[3])
                    dof_list.append(ndof)
                if row[0] == "BiCGStab" and row[1] == "converged":
                    time = float(row[-2][1:])
                    time_i.append(time)

    t_list.append(time_i)

mean_times = np.array([np.mean(time) for time in t_list])
dofs = np.array(dof_list)

plt.rcParams.update({"font.size": 13})
ax = plt.gca()
ax.loglog(dofs, mean_times, ".--")
ax.grid(True, which="both", ls="-", color="0.65")

# ax2 = ax.twinx
# for t, dof in zip(t_list, dofs):
#     plt.boxplot(t, positions=[dof])

plt.show()


table = np.vstack((dofs, mean_times)).T
ratio = table[:, 1] / table[:, 0]


pass
