import subprocess

subprocess.run(["flow", "tests/data/pyaction/PYACTION_TEST.DATA"])

""" 
Shows the following warning:

PYACTION can be used without a run(ecl_state, schedule, report_step, summary_state, actionx_callback) function, 
its arguments are available as attributes of the module opm_embedded, try the following in your python script:
                    
import opm_embedded

help(opm_embedded.current_ecl_state)
help(opm_embedded.current_schedule)
help(opm_embedded.current_report_step)
help(opm_embedded.current_summary_state)
"""
