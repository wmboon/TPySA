import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import time
import pyamg


class Solver:
    def report_time(self, report: str, start_time: float):
        print("TPSA: {} ({:.2f} sec)".format(report, time.time() - start_time))

    def solve(self, rhs: np.ndarray, **kwargs) -> tuple:
        return rhs, 1


class DirectSolver(Solver):
    def __init__(self, system: sps.sparray):
        start_time = time.time()
        self.system_LU = spla.splu(system)
        self.report_time("LU-factorization", start_time)

    def solve(self, rhs: np.ndarray, **kwargs) -> tuple:
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

    def solve(self, rhs: np.ndarray, rtol: float = 1e-5) -> tuple:
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
            rtol=rtol,
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
        u_x_slice = slice(0, num_cells)
        precond_u = self.assemble_AMG_preconditioner(u_x_slice)

        # Rotation preconditioner
        r_x_slice = slice(self.ndofs[0], self.ndofs[0] + num_cells)
        restriction_r = self.create_restriction(r_x_slice)
        mass_r = restriction_r @ self.system @ restriction_r.T
        precond_r = mass_r.data

        # Pressure preconditioner
        p_slice = slice(6 * num_cells, None)
        precond_p = self.assemble_AMG_preconditioner(p_slice)

        # Off-diagonal blocks
        col_restriction = self.create_restriction(slice(0, self.ndofs[0]))
        row_restriction = self.create_restriction(slice(self.ndofs[0], None))
        A_10 = row_restriction @ self.system @ col_restriction.T

        # Combining the preconditioners
        def precond_func(res: np.ndarray):
            u, r, p = np.split(res.astype(float), np.cumsum(self.ndofs))

            # Apply AMG to the different components of u
            u = u.reshape((-1, 3), order="F")
            u = precond_u.matmat(u).ravel(order="F")

            # Compute the update to the residuals of r and p
            r_delta, p_delta = np.split(A_10 @ u, [len(r)])

            # Update the rotation residual and solve using the diagonal
            r -= r_delta
            r = r.reshape((-1, 3), order="F") / precond_r[:, None]
            r = r.ravel(order="F")

            # Update the pressure residual and solve using AMG
            p -= p_delta
            p = precond_p.matvec(p)

            return np.hstack((u, r, p))

        self.precond = spla.LinearOperator(self.system.shape, precond_func)

        self.report_time("AMG-initialization", start_time)

    def assemble_AMG_preconditioner(self, indices: slice):
        restriction = self.create_restriction(indices)
        laplace = sps.csr_matrix(restriction @ self.system @ restriction.T)

        laplace.indices = laplace.indices.astype(np.int32)
        laplace.indptr = laplace.indptr.astype(np.int32)

        amg_u = pyamg.smoothed_aggregation_solver(laplace)
        return amg_u.aspreconditioner()

    def create_restriction(self, indices: slice):
        eye = sps.eye_array(*self.system.shape, format="csr")
        return eye[indices]

    def solve(self, rhs: np.ndarray, rtol: float = 1e-5) -> tuple:
        start_time = time.time()

        num_it = 0
        res = 0.0
        norm_rhs = np.linalg.norm(rhs)

        def callback(x):
            nonlocal num_it, res
            num_it += 1
            res = np.linalg.norm(rhs - self.system @ x) / norm_rhs
            print(
                "BiCGStab: Iterate {:3}, Residual: {:.2e}".format(num_it, res), end="\r"
            )

        sol, info = spla.bicgstab(
            self.system,
            rhs,
            rtol=rtol,
            M=self.precond,
            callback=callback,
        )
        self.report_time(
            "BiCGStab converged in {} iterations to an accuracy of {:.0e}".format(
                num_it, res
            ),
            start_time,
        )

        return sol, info
