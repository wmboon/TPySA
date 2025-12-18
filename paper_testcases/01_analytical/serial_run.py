import os
import numpy as np

num_meshes = 7
nx_list = np.floor(5 * 1.5 ** np.arange(num_meshes)).astype(int)

dir_name = os.path.dirname(__file__)
output = os.path.join(dir_name, "errs.txt")
if os.path.exists(output):
    os.remove(output)

for nx in nx_list:
    python = "/home/AD.NORCERESEARCH.NO/wibo/opm/opm-common/.venv/bin/python"
    file = "/home/AD.NORCERESEARCH.NO/wibo/TPySA/paper_testcases/01_analytical/main.py"

    os.system(" ".join((python, file, str(nx))))
