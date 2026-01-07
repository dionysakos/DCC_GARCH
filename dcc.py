import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple

def dcc_recursion(z: np.ndarray, a: float, b: float) -> np.ndarray:
    if a < 0 or b < 0 or (a + b) >= 1:
        raise ValueError("DCC constraints violated: a>=0, b>=0, a+b<1")

    T, N = z.shape
    Qbar = np.cov(z, rowvar=False) # Unconditional covariance of z
    Q = Qbar.copy()
    R = np.zeros((T, N, N)) # Dynamic conditional correlation

    for t in range(T):
        if t > 0:
            zprev = z[t - 1].reshape(N, 1)
            Q = (1 - a - b) * Qbar + a * (zprev @ zprev.T) + b * Q # Update Q with new information

        d = np.sqrt(np.diag(Q))
        Dinv = np.diag(1.0 / d)
        Rt = Dinv @ Q @ Dinv

        # Numeric stability
        Rt = 0.5 * (Rt + Rt.T)
        np.fill_diagonal(Rt, 1.0)
        R[t] = Rt

    return R

def dcc_negloglik(params: np.ndarray, z: np.ndarray) -> float:
    a, b = float(params[0]), float(params[1])
    if a < 0 or b < 0 or (a + b) >= 1:
        return 1e12

    R = dcc_recursion(z, a, b)
    T, N = z.shape
    nll = 0.0

    for t in range(T):
        Rt = R[t]
        sign, logdet = np.linalg.slogdet(Rt)
        if sign <= 0 or not np.isfinite(logdet):
            return 1e12

        zt = z[t].reshape(N, 1)
        x = np.linalg.solve(Rt, zt)
        quad = (zt.T @ x).item()



        nll += 0.5 * (logdet + quad)

    return float(nll)

def fit_dcc_mle(z_df: pd.DataFrame, a0: float = 0.02, b0: float = 0.97) -> Tuple[float, float, np.ndarray, object]:
    z = z_df.values

    cons = (
        {"type": "ineq", "fun": lambda x: x[0]},                     # a >= 0
        {"type": "ineq", "fun": lambda x: x[1]},                     # b >= 0
        {"type": "ineq", "fun": lambda x: 0.999999 - (x[0] + x[1])}  # a+b < 1
    )
    bounds = [(0.0, 0.999999), (0.0, 0.999999)]

    out = minimize(
        dcc_negloglik,
        x0=np.array([a0, b0], dtype=float),
        args=(z,),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 2000, "ftol": 1e-9}
    )

    if not out.success:
        raise RuntimeError(f"DCC MLE failed: {out.message}")

    a_hat, b_hat = float(out.x[0]), float(out.x[1])
    R_hat = dcc_recursion(z, a_hat, b_hat)

    return a_hat, b_hat, R_hat, out

def extract_pair_corr(R: np.ndarray, tickers: list[str], pair: tuple[str, str]) -> np.ndarray:
    t1, t2 = pair
    i, j = tickers.index(t1), tickers.index(t2)
    return R[:, i, j]
