import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def segment_overview(modelling_data):
  
    # Data
    seg_counts = modelling_data["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Customers"]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "xy"}, {"type": "domain"}]],
        subplot_titles=("Customer Count per Segment", "Segment Share of Customers"),
        horizontal_spacing=0.12,
    )

    # 1a — Bar
    fig.add_trace(
        go.Bar(
            x=seg_counts["Segment"], y=seg_counts["Customers"],
            marker_color=px.colors.qualitative.Bold[:len(seg_counts)],
            text=seg_counts["Customers"], textposition="outside",
            showlegend=False, name="Customers",
        ),
        row=1, col=1,
    )

    # 1b — Pie (donut)
    fig.add_trace(
        go.Pie(
            labels=seg_counts["Segment"], values=seg_counts["Customers"],
            hole=0.5, textinfo="label+percent",
            marker=dict(colors=px.colors.qualitative.Bold),
            showlegend=False,
        ),
        row=1, col=2,
    )

    fig.update_xaxes(tickangle=-25, row=1, col=1)
    fig.update_layout(height=480, title_text="Segment Overview")
    return fig

def segment_behaviour_demographics(modelling_data):
    seg_profile = (modelling_data
               .groupby("Segment")
               .agg(Recency=("Recency", "mean"),
                    Frequency=("Frequency", "mean"),
                    Monetary=("Monetary", "mean"),
                    Age=("Age", "mean"))
               .reset_index())

    rev_by_seg = (modelling_data
                .groupby("Segment")["Monetary"].sum()
                .reset_index())

    gender_seg = (modelling_data
                .groupby(["Segment", "CustGender"])
                .size().reset_index(name="Count"))

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "domain"}, {"type": "xy"}],
            [{"type": "xy"},     {"type": "xy"}]],
        subplot_titles=("Revenue Share per Segment",
                        "Avg Recency / Frequency per Segment",
                        "Gender Composition per Segment",
                        "Average Age per Segment"),
        horizontal_spacing=0.12, vertical_spacing=0.18,
    )

    # 2a — Donut: revenue share
    fig.add_trace(
        go.Pie(
            labels=rev_by_seg["Segment"], values=rev_by_seg["Monetary"],
            hole=0.55, textinfo="label+percent",
            marker=dict(colors=px.colors.qualitative.Bold),
            showlegend=False,
        ),
        row=1, col=1,
    )

    # 2b — Grouped bar: Recency + Frequency (single plot, two traces)
    fig.add_trace(
        go.Bar(
            x=seg_profile["Segment"], y=seg_profile["Recency"],
            name="Avg Recency (days)",
            marker_color=px.colors.qualitative.Bold[0],
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(
            x=seg_profile["Segment"], y=seg_profile["Frequency"],
            name="Avg Frequency",
            marker_color=px.colors.qualitative.Bold[1],
            yaxis="y2",
        ),
        row=1, col=2,
    )

    # 2c — Grouped bar: gender
    for i, gender in enumerate(gender_seg["CustGender"].unique()):
        sub = gender_seg[gender_seg["CustGender"] == gender]
        fig.add_trace(
            go.Bar(
                x=sub["Segment"], y=sub["Count"], name=gender,
                marker_color=px.colors.qualitative.Bold[i],
            ),
            row=2, col=1,
        )

    # 2d — Bar: average age
    fig.add_trace(
        go.Bar(
            x=seg_profile["Segment"], y=seg_profile["Age"],
            marker_color=px.colors.qualitative.Bold[2],
            text=seg_profile["Age"].round(1), textposition="outside",
            showlegend=False, name="Avg Age",
        ),
        row=2, col=2,
    )

    fig.update_layout(
        height=800, barmode="group",
        title_text="Segment Behaviour and Demographics",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    fig.update_xaxes(tickangle=-25)
    return fig
