import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import time
import pyamg


class Solver:
    def report_time(self, report: str, start_time: float):
        print("TPSA: {} ({:.2f} sec)".format(report, time.time() - start_time))

    def solve(self, rhs: np.ndarray) -> tuple:
        return rhs, 1


class DirectSolver(Solver):
    def __init__(self, system: sps.sparray):
        start_time = time.time()
        self.system_LU = spla.splu(system)
        self.report_time("LU-factorization", start_time)

    def solve(self, rhs: np.ndarray) -> tuple:
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

    def solve(self, rhs: np.ndarray) -> tuple:
        start_time = time.time()

        num_it = 0

        def callback(r):
            nonlocal num_it
            num_it += 1
            print(
                "GMRes: Iterate {:3}, Prec. Res. norm: {:.2e}".format(num_it, r),
                end="\r",
            )

        sol, info = spla.gmres(
            self.system,
            rhs,
            M=self.precond,
            callback=callback,
            callback_type="pr_norm",
        )
        self.report_time("GMRes converged in {} iterations".format(num_it), start_time)

        return sol, info


class AMGSolver(Solver):
    def __init__(self, system: sps.sparray):
        start_time = time.time()

        self.system = system

        ndof = self.system.shape[0]
        num_cells = ndof // 7
        self.ndofs = [3 * num_cells, 3 * num_cells]

        # Displacement preconditioner
        restriction_u = sps.eye_array(num_cells, ndof)
        laplace = sps.csr_matrix(restriction_u @ self.system @ restriction_u.T)

        laplace.indices = laplace.indices.astype(np.int32)
        laplace.indptr = laplace.indptr.astype(np.int32)

        amg_u = pyamg.smoothed_aggregation_solver(laplace)
        precond_u = amg_u.aspreconditioner(cycle="V")

        # Rotation preconditioner
        restriction_r = sps.diags_array(
            np.ones(num_cells), offsets=self.ndofs[0], shape=(restriction_u.shape)
        )
        mass_r = restriction_r @ self.system @ restriction_r.T
        precond_r = mass_r.data

        # Pressure preconditioner
        restriction_p = sps.diags_array(
            np.ones(num_cells), offsets=2 * self.ndofs[0], shape=(restriction_u.shape)
        )
        laplace_p = sps.csr_matrix(restriction_p @ self.system @ restriction_p.T)

        laplace_p.indices = laplace_p.indices.astype(np.int32)
        laplace_p.indptr = laplace_p.indptr.astype(np.int32)

        amg_p = pyamg.smoothed_aggregation_solver(laplace_p)
        precond_p = amg_p.aspreconditioner(cycle="V")

        # Combining the preconditioners
        def precond_func(res: np.ndarray):
            u, r, p = np.split(res, np.cumsum(self.ndofs))

            u = res[: self.ndofs[0]].reshape((-1, 3), order="F")
            u = precond_u.matmat(u).ravel(order="F")

            r = r.reshape((-1, 3), order="F") / precond_r[:, None]
            r = r.ravel(order="F")
            p = precond_p.matvec(p)

            return np.hstack((u, r, p))

        self.precond = spla.LinearOperator(self.system.shape, precond_func)

        self.report_time("AMG-initialization", start_time)

    def solve(self, rhs: np.ndarray) -> tuple:
        start_time = time.time()

        num_it = 0

        def callback(_):
            nonlocal num_it
            num_it += 1
            print("BiCGStab: Iterate {:3}".format(num_it), end="\r")

        sol, info = spla.bicgstab(
            self.system,
            rhs,
            rtol=1e-3,
            M=self.precond,
            callback=callback,
        )
        self.report_time(
            "BiCGStab converged in {} iterations".format(num_it), start_time
        )

        return sol, info
