## Rescaling TPSA

The TPSA discretization of linearized elasticity discretizes the system:

$$
\begin{align}
    \left(
    \nabla \cdot 
    \begin{bmatrix}
        2 \mu \nabla & S^* & I \\\
        S^* \\\
        I 
    \end{bmatrix}
    - 
    \begin{bmatrix}
        0 \\\ & \mu^{-1} \\\ & & \lambda^{-1}
    \end{bmatrix}
    \right)
    \begin{bmatrix}
        u \\\ r \\\ p_s
    \end{bmatrix}
    =
    \begin{bmatrix}
        f_u \\\ f_r \\\ f_p
    \end{bmatrix}
\end{align}
$$

The Lamé parameters are typically on the order of $10^{10}$ in SI units, which yields a terribly scaled matrix. We therefore look into a rescaling of the equations and variables:
$$
\begin{align}
    u^* &= \tilde \mu^{\frac12} u, &
    r^* &= \tilde \mu^{-\frac12} r, &
    p^* &= \tilde \mu^{-\frac12} p
\end{align}
$$
where $\tilde \mu$ is a representative value for $\mu$ (we take the mean). Multiplying left and right with appropriate scaling matrices, we obtain
$$
\begin{align}
    \begin{bmatrix}
        \tilde \mu^{-\frac12} \\\
        & \tilde \mu^{\frac12} \\\
        & & \tilde \mu^{\frac12} 
    \end{bmatrix}
    \left(
    \nabla \cdot 
    \begin{bmatrix}
        2 \mu \nabla & S^* & I \\\
        S^* \\\
        I 
    \end{bmatrix}
    - 
    \begin{bmatrix}
        0 \\\ & \mu^{-1} \\\ & & \lambda^{-1}
    \end{bmatrix}
    \right)
    \begin{bmatrix}
        \tilde \mu^{-\frac12} \\\
        & \tilde \mu^{\frac12} \\\
        & & \tilde \mu^{\frac12} 
    \end{bmatrix}
    \begin{bmatrix}
        u^* \\\ r^* \\\ p_s^*
    \end{bmatrix}
    &=
    \begin{bmatrix}
        \tilde \mu^{-\frac12} f_u \\\ 
        \tilde \mu^{\frac12} f_r \\\ 
        \tilde \mu^{\frac12}f_p
    \end{bmatrix}
 \\\
    \left(
    \nabla \cdot 
    \begin{bmatrix}
        2 (\mu \tilde \mu^{-1}) \nabla & S^* & I \\\
        S^* \\\
        I 
    \end{bmatrix}
    - 
    \begin{bmatrix}
        0 \\\ & \tilde \mu \mu^{-1} \\\ & & \tilde \mu \lambda^{-1}
    \end{bmatrix}
    \right)
    \begin{bmatrix}
        u^* \\\ r^* \\\ p_s^*
    \end{bmatrix}
    &=
    \begin{bmatrix}
        \tilde \mu^{-\frac12} f_u \\\ 
        \tilde \mu^{\frac12} f_r \\\ 
        \tilde \mu^{\frac12}f_p
    \end{bmatrix}
\end{align}
$$

This is the matrix that is implemented in the `TPySA` code. Note that this implies that the solution needs to be rescaled as:
$$
\begin{bmatrix}
    u \\\ r \\\ p_s
\end{bmatrix}
= 
\begin{bmatrix}
    \tilde \mu^{-\frac12} \\\
    & \tilde \mu^{\frac12} \\\
    & & \tilde \mu^{\frac12} 
\end{bmatrix}
    \begin{bmatrix}
        u^* \\\ r^* \\\ p_s^*
    \end{bmatrix}
$$