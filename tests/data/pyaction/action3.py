from opm.opmcommon_python import SummaryState


def run(ecl_state, schedule, report_step, summary_state, actionx_callback):
    print("We are now on {}".format(report_step))
    data = """
        SOURCE
            1 1 1 WATER 100 /
        /
        """
    schedule.insert_keywords(data)
