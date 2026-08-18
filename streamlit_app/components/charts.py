import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, List

def render_live_trend_chart(history_df: pd.DataFrame, selected_metrics: List[str]) -> go.Figure:
    """
    Renders a multi-line chart of historical sensor telemetry trends.
    
    Args:
        history_df: DataFrame of historical simulation logs.
        selected_metrics: List of metrics to plot.
        
    Returns:
        go.Figure: Plotly Express line chart.
    """
    if history_df.empty or not selected_metrics:
        # Return an empty placeholder figure
        fig = go.Figure()
        fig.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{
                "text": "NO TELEMETRY RECORDED YET",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 16, "color": "#64748b"}
            }],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=280
        )
        return fig
        
    # Standardize column naming for legend labels
    melted_df = history_df.melt(
        id_vars=["ticks"],
        value_vars=selected_metrics,
        var_name="Sensor Variable",
        value_name="Value"
    )
    
    # Capitalize and format titles
    melted_df["Sensor Variable"] = melted_df["Sensor Variable"].apply(lambda x: x.replace("_", " ").title())
    
    fig = px.line(
        melted_df,
        x="ticks",
        y="Value",
        color="Sensor Variable",
        title="REAL-TIME TELEMETRY STREAM",
        color_discrete_sequence=px.colors.qualitative.G10
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.5)',
        margin=dict(l=40, r=20, t=40, b=30),
        height=280,
        font={'color': "#e2e8f0", 'family': "Outfit"},
        xaxis={
            'title': "Simulation Tick (Time)", 
            'gridcolor': '#1e293b', 
            'linecolor': '#1e293b'
        },
        yaxis={
            'title': "Sensor Output Value", 
            'gridcolor': '#1e293b', 
            'linecolor': '#1e293b'
        },
        legend={
            'title': None, 
            'orientation': 'h', 
            'yanchor': 'bottom', 
            'y': 1.02, 
            'xanchor': 'right', 
            'x': 1.0
        }
    )
    
    # Make lines smooth
    fig.update_traces(line_shape="spline", line_width=2.5)
    return fig

def render_probability_breakdown_chart(probabilities: Dict[str, float]) -> go.Figure:
    """
    Renders a horizontal bar chart showing predicted health state likelihoods.
    
    Args:
        probabilities: Dictionary of state -> probability.
        
    Returns:
        go.Figure: Plotly horizontal bar chart.
    """
    states = ["Normal", "Warning", "Critical"]
    values = [probabilities.get(s, 0.0) * 100.0 for s in states]
    
    colors = ["#10b981", "#f59e0b", "#ef4444"]  # emerald, amber, rose
    
    fig = go.Figure(go.Bar(
        x=values,
        y=states,
        orientation='h',
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition='outside',
        textfont={'color': '#e2e8f0', 'family': 'Share Tech Mono'}
    ))
    
    fig.update_layout(
        title="RISK LIKELIHOOD BREAKDOWN",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=70, r=50, t=40, b=10),
        height=180,
        font={'color': "#e2e8f0", 'family': "Outfit"},
        xaxis={
            'title': "Confidence (%)", 
            'range': [0, 115],
            'gridcolor': '#1e293b', 
            'linecolor': '#1e293b',
            'visible': False
        },
        yaxis={'linecolor': '#1e293b'}
    )
    
    return fig

def render_rul_history_chart(history_df: pd.DataFrame) -> go.Figure:
    """
    Renders a trend chart showing predictions of Remaining Useful Life over time.
    
    Args:
        history_df: DataFrame of historical logs containing predicted_days_until_failure.
        
    Returns:
        go.Figure: RUL trend chart.
    """
    if history_df.empty or "predicted_days_until_failure" not in history_df.columns:
        fig = go.Figure()
        fig.update_layout(
            xaxis={"visible": False}, yaxis={"visible": False},
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=220
        )
        return fig
        
    fig = px.area(
        history_df,
        x="ticks",
        y="predicted_days_until_failure",
        title="PREDICTED DAYS UNTIL FAILURE (RUL) TREND"
    )
    
    # Style area fill
    fig.update_traces(
        line_color="#38bdf8", 
        fillcolor="rgba(56, 189, 248, 0.15)", 
        line_width=2.5,
        line_shape="spline"
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.5)',
        margin=dict(l=40, r=20, t=45, b=30),
        height=220,
        font={'color': "#e2e8f0", 'family': "Outfit"},
        xaxis={'title': "Simulation Tick", 'gridcolor': '#1e293b', 'linecolor': '#1e293b'},
        yaxis={'title': "Days Remaining", 'gridcolor': '#1e293b', 'linecolor': '#1e293b', 'range': [0, 370]}
    )
    
    return fig
