import scipy.sparse as sps
import scipy.sparse.linalg as spla
import time
import tpysa


class Solver:
    def report_time(self, output, start_time):
        print("TPSA: {} ({:.2f} sec)".format(output, time.time() - start_time))

    def solve(self, rhs):
        pass


class DirectSolver(Solver):
    def __init__(self, system: sps.sparray):
        start_time = time.time()
        self.system_LU = spla.splu(system.system)
        self.report_time("LU-factorization", start_time)

    def solve(self, rhs):
        start_time = time.time()
        sol = self.system_LU.solve(rhs)
        self.report_time("Direct solve", start_time)

        return sol, 0


class ILUSolver(Solver):
    def __init__(self, system: sps.sparray):
        start_time = time.time()

        self.system = system
        P_LU = spla.spilu(self.system, fill_factor=3)
        self.precond = spla.LinearOperator(self.system.shape, P_LU.solve)
        self.report_time("ILU-factorization", start_time)

    def solve(self, rhs):
        start_time = time.time()

        num_it = 0

        def callback(r):
            nonlocal num_it
            num_it += 1
            print(
                "TPSA: Iterate {:3}, Prec. Res. norm: {:.2e}".format(num_it, r),
                end="\r",
            )

        sol, info = spla.gmres(
            self.system,
            rhs,
            # rtol=1e-5,
            M=self.precond,
            callback=callback,
            callback_type="pr_norm",
        )
        self.report_time("GMRes converged in {} iterations".format(num_it), start_time)

        return sol, info
