import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def customer_value_geography(modelling_data):
    top10 = modelling_data.sort_values("Monetary", ascending=False).head(10)

    loc_counts = modelling_data["CustLocation"].value_counts().head(5).reset_index()
    loc_counts.columns = ["Location", "Customers"]

    bal_seg = (modelling_data
            .groupby("Segment")["LastAccountBalance"]
            .mean().reset_index()
            .sort_values("LastAccountBalance", ascending=False))

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy"},     {"type": "domain"}],
            [{"type": "xy"},     {"type": "xy"}]],
        subplot_titles=("Top 10 Customers by Revenue",
                        "Top 5 Customer Locations",
                        "Cumulative Revenue Within Each Segment",
                        "Avg Account Balance per Segment"),
        horizontal_spacing=0.12, vertical_spacing=0.20,
    )

    # 3a — Bar: top 10 customers
    fig.add_trace(
        go.Bar(
            x=top10["CustomerID"], y=top10["Monetary"],
            marker_color=[px.colors.qualitative.Bold[
                list(modelling_data["Segment"].unique()).index(s) % 10]
                for s in top10["Segment"]],
            text=top10["Monetary"].apply(lambda v: f"£{v/1e3:,.0f}k"),
            textposition="outside",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # 3b — Donut: locations
    fig.add_trace(
        go.Pie(
            labels=loc_counts["Location"], values=loc_counts["Customers"],
            hole=0.5, textinfo="label+percent",
            marker=dict(colors=px.colors.qualitative.Bold),
            showlegend=False,
        ),
        row=1, col=2,
    )

    # 3c — Line: cumulative revenue within each segment
    for i, seg in enumerate(modelling_data["Segment"].unique()):
        seg_data = (modelling_data[modelling_data["Segment"] == seg]
                    .sort_values("Monetary", ascending=False))
        if len(seg_data) == 0:
            continue
        cumulative = seg_data["Monetary"].cumsum() / seg_data["Monetary"].sum() * 100
        cust_pct = np.arange(1, len(seg_data) + 1) / len(seg_data) * 100
        fig.add_trace(
            go.Scatter(
                x=cust_pct, y=cumulative, mode="lines", name=seg,
                line=dict(width=3, color=px.colors.qualitative.Bold[i % 10]),
            ),
            row=2, col=1,
        )

    # 3d — Bar: avg balance per segment
    fig.add_trace(
        go.Bar(
            x=bal_seg["Segment"], y=bal_seg["LastAccountBalance"],
            marker_color=px.colors.qualitative.Bold[:len(bal_seg)],
            text=bal_seg["LastAccountBalance"].apply(lambda v: f"£{v:,.0f}"),
            textposition="outside", showlegend=False,
        ),
        row=2, col=2,
    )

    fig.update_layout(
        height=850, title_text="Value Concentration, Geography, and Balance",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    fig.update_xaxes(tickangle=-25)

    return fig
