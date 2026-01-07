import pandas as pd
from arch import arch_model
from typing import Dict, Tuple

def fit_garch_t(returns_scaled: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    sigmas = pd.DataFrame(index=returns_scaled.index, columns=returns_scaled.columns, dtype=float)
    z = pd.DataFrame(index=returns_scaled.index, columns=returns_scaled.columns, dtype=float)
    resu = {}

    for ticker in returns_scaled.columns:
        x = returns_scaled[ticker].dropna()

        am = arch_model(x, mean="Constant", vol="Garch", p=1, q=1, dist="t")
        res = am.fit(disp="off")

        sigmas.loc[x.index, ticker] = res.conditional_volatility
        z.loc[x.index, ticker] = res.std_resid  #take standardized residuals
        resu[ticker] = res

    sigmas = sigmas.dropna()
    z = z.dropna()

    return sigmas, z, resu
