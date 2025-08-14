import os

nx_list = [5, 10, 20, 40]
for nx in nx_list:
    python = "/home/AD.NORCERESEARCH.NO/wibo/opm/opm-common/.venv/bin/python"
    file = (
        "/home/AD.NORCERESEARCH.NO/wibo/TPySA/examples/10_analytical_biot_dec/main.py"
    )

    os.system(" ".join((python, file, str(nx))))
