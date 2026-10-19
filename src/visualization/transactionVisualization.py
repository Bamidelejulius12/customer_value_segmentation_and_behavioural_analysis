import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def channel_activity(type_mix, wallet_seg):
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=(
            "Transaction Type Mix per Segment",
            "Digital / Crypto Activity per Segment (%)",
        ),
        horizontal_spacing=0.12,
    )

    # 1. Transaction type mix (stacked bar)
    for i, t in enumerate(type_mix["TransactionType"].unique()):
        sub = type_mix[type_mix["TransactionType"] == t]
        fig.add_trace(
            go.Bar(
                x=sub["Segment"], y=sub["Count"], name=t,
                legendgroup="type",
                marker_color=px.colors.qualitative.Bold[i % 10],
            ),
            row=1, col=1,
        )

    # 2. Wallet share (bar)
    fig.add_trace(
        go.Bar(
            x=wallet_seg["Segment"], y=wallet_seg["WalletShare%"],
            marker_color=px.colors.qualitative.Bold[2],
            text=wallet_seg["WalletShare%"].round(1),
            texttemplate="%{text}%", textposition="outside",
            showlegend=False,
        ),
        row=1, col=2,
    )

    fig.update_xaxes(tickangle=-25)
    fig.update_layout(
        height=520,
        barmode="stack",
        title_text="Channel Mix and Digital Activity by Segment",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                    xanchor="right", x=1, font=dict(size=10)),
    )
    return fig

def currency_status(cur_mix, status_mix):
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=(
            "Currency Mix per Segment",
            "Status Mix per Segment (%)",
        ),
        horizontal_spacing=0.12,
    )

    # 3. Currency mix (stacked bar)
    for i, c in enumerate(cur_mix["TransactionCurrency"].unique()):
        sub = cur_mix[cur_mix["TransactionCurrency"] == c]
        fig.add_trace(
            go.Bar(
                x=sub["Segment"], y=sub["Count"], name=c,
                legendgroup="currency",
                marker_color=px.colors.qualitative.Bold[i % 10],
            ),
            row=1, col=1,
        )

    # 4. Status mix (stacked bar, as %)
    for i, s in enumerate(status_mix["TransactionStatus"].unique()):
        sub = status_mix[status_mix["TransactionStatus"] == s]
        fig.add_trace(
            go.Bar(
                x=sub["Segment"], y=sub["Pct"], name=s,
                legendgroup="status",
                marker_color=px.colors.qualitative.Bold[i % 10],
            ),
            row=1, col=2,
        )

    fig.update_xaxes(tickangle=-25)
    fig.update_layout(
        height=520,
        barmode="stack",
        title_text="Currency and Status Behaviour by Segment",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                    xanchor="right", x=1, font=dict(size=10)),
    )
    return fig 

