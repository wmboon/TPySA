# from opm_embedded import SummaryState
import pdb
from opm.opmcommon_python import SummaryState

def run(ecl_state, schedule, report_step, summary_state, actionx_callback):

        # pdb.set_trace()
        print("We are now on {}".format(report_step))
        data = """
        SOURCE
            1 1 1 WATER 100 /
        /
        """
        schedule.insert_keywords(data)
