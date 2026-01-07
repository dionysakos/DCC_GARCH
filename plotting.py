import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_dcc_dashboard(
    returns_scaled: pd.DataFrame,
    sigmas: pd.DataFrame,
    corr_series: pd.Series,
    title: str = "DCC-GARCH(1,1)-t Analysis"
):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Returns
    for tkr in returns_scaled.columns:
        fig.add_trace(
            go.Scatter(
                x=returns_scaled.index,
                y=returns_scaled[tkr],
                mode="lines",
                name=f"{tkr} Returns",
                opacity=0.7
            ),
            secondary_y=False
        )

    # GARCH vol
    for tkr in sigmas.columns:
        fig.add_trace(
            go.Scatter(
                x=sigmas.index,
                y=sigmas[tkr],
                mode="lines",
                name=f"{tkr} GARCH-t Vol",
                line=dict(dash="dash"),
                opacity=0.9
            ),
            secondary_y=False
        )

    # DCC correlation
    fig.add_trace(
        go.Scatter(
            x=corr_series.index,
            y=corr_series.values,
            mode="lines",
            name=corr_series.name or "DCC Corr",
            line=dict(width=3),
            opacity=0.65
        ),
        secondary_y=True
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        height=650,
        width=1100,
        hovermode="x unified",
        legend=dict(orientation="v")
    )

    fig.update_yaxes(title_text="Returns / Volatility", secondary_y=False)
    fig.update_yaxes(title_text="DCC Correlation", secondary_y=True, range=[-1, 1])

    fig.show()
