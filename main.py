from config import Config
from data import fetch_prices, log_returns
from garch import fit_garch_t
from dcc import fit_dcc_mle, extract_pair_corr
from plotting import plot_dcc_dashboard
import pandas as pd

def run(cfg: Config):
    prices = fetch_prices(list(cfg.tickers), cfg.start_date, cfg.end_date, auto_adjust=True)
    returns_scaled = log_returns(prices, scale=cfg.scale)

    sigmas, z_df, _ = fit_garch_t(returns_scaled)

    a0, b0 = cfg.dcc_init
    a_hat, b_hat, R_hat, _ = fit_dcc_mle(z_df, a0=a0, b0=b0)

    tickers = list(z_df.columns)
    corr = extract_pair_corr(R_hat, tickers, (tickers[0], tickers[1]))
    corr_series = pd.Series(corr, index=z_df.index, name=f"DCC Corr ({tickers[0]},{tickers[1]})")

    print(f"DCC MLE params: a={a_hat:.6f}, b={b_hat:.6f}, a+b={a_hat+b_hat:.6f}")

    # Plot
    plot_dcc_dashboard(
        returns_scaled=returns_scaled.loc[z_df.index],
        sigmas=sigmas,
        corr_series=corr_series,
        title="DCC-GARCH(1,1)-t (MLE a,b)"
    )

if __name__ == "__main__":
    cfg = Config(
        tickers=("AAPL", "NVDA"),
        start_date="2020-01-01",
        end_date="2024-06-01",
        scale=100.0,
        dcc_init=(0.02, 0.97),
    )
    run(cfg)
