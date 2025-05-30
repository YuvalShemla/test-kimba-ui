import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from palette import palette

def create_comparison_plot(df, feature, display_name, unit, min_active_nights=5, fig_width=2400):
    """
    Create an interactive comparison plot using Plotly Express
    """
    # Filter for active users
    active_nights = df[df['diffuser_category'] == 'active'].groupby('user_id').size()
    valid_users = active_nights[active_nights >= min_active_nights].index
    filtered_df = df[df['user_id'].isin(valid_users)]
    
    # Calculate statistics
    user_stats = filtered_df.groupby(['user_id', 'diffuser_category'])[feature].agg(['mean', 'count']).reset_index()
    pivot_stats = user_stats.pivot(index='user_id', columns='diffuser_category', values=['mean', 'count'])
    
    if ('mean', 'active') not in pivot_stats or ('mean', 'control') not in pivot_stats:
        return None
    
    # Calculate percentage difference
    pivot_stats['diff_percent'] = ((pivot_stats[('mean', 'active')] - pivot_stats[('mean', 'control')]) /
                                   pivot_stats[('mean', 'control')] * 100)
    
    # Convert user IDs to strings for robust annotation alignment
    user_ids = [str(uid) for uid in pivot_stats.index]
    y_control = list(pivot_stats[('mean', 'control')])
    y_active = list(pivot_stats[('mean', 'active')])
    diffs = list(pivot_stats['diff_percent'])
    n_control = list(pivot_stats[('count', 'control')])
    n_active = list(pivot_stats[('count', 'active')])
    
    # Build custom x-tick labels
    def format_diff(diff):
        if np.isfinite(diff):
            return f"{diff:.1f}%"
        else:
            return "N/A"
    custom_labels = [f"{uid}<br>({format_diff(diff)})" for uid, diff in zip(user_ids, diffs)]

    # Calculate mean of percentage differences (finite only)
    finite_diffs = [d for d in diffs if np.isfinite(d)]
    mean_diff = np.nanmean(finite_diffs) if finite_diffs else float('nan')
    mean_diff_text = f"Mean % diff: {mean_diff:.2f}%" if np.isfinite(mean_diff) else "Mean % diff: N/A"
    
    # Create the figure
    fig = go.Figure()
    
    # Add control bars
    fig.add_trace(go.Bar(
        x=custom_labels,
        y=y_control,
        name='Control',
        marker_color=palette["control"],
        text=[f"n={int(n)}" for n in n_control],
        textposition='outside',
        hovertemplate='User: %{x}<br>Mean: %{y:.2f}<br>n: %{text}',
        width=0.3
    ))
    
    # Add active bars
    fig.add_trace(go.Bar(
        x=custom_labels,
        y=y_active,
        name='Active',
        marker_color=palette["active"],
        text=[f"n={int(n)}" for n in n_active],
        textposition='outside',
        hovertemplate='User: %{x}<br>Mean: %{y:.2f}<br>n: %{text}',
        width=0.3
    ))
    
    # Add mean diff as an annotation in the legend area
    fig.add_annotation(
        text=mean_diff_text,
        xref="paper", yref="paper",
        x=1, y=1.13,  # above the legend
        showarrow=False,
        font=dict(size=16, color="black", family="Arial"),
        align="right",
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="black",
        borderwidth=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"{display_name} Comparison (Control vs Active)",
        xaxis_title="User ID (and % difference below)",
        yaxis_title=f"{display_name} ({unit})",
        barmode='group',
        showlegend=True,
        height=700,
        width=fig_width,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, tickangle=-45, type='category'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey'),
        margin=dict(l=40, r=40, t=80, b=180)
    )
    
    return fig

# Define the features to plot with their display names and units
FEATURES_TO_PLOT = {
    'sleep_efficiency': ('Sleep Efficiency', '%'),
    'total_sleep_time': ('Total Sleep Time', 'minutes'),
    'waso_duration': ('WASO Duration', 'minutes'),
    'waso_count': ('WASO Count', 'count'),
    'sleep_latency': ('Sleep Latency', 'minutes'),
    'heart_rate_mean': ('Heart Rate Mean', 'bpm'),
    'hrv_mean_hr_mean': ('HRV Heart Rate Mean', 'ms'),
    'bbi_mean': ('BBI HRV Mean', 'ms'),
    'hrv_std_hr_mean': ('RMSSD', 'ms'),
    'acc_magnitude_mean': ('Movement Vector', 'm/s²'),
    'respiration_rate_mean': ('Respiration Rate', 'breaths/min'),
    'oxygen_level_mean': ('SpO2', '%'),
    'deep_duration': ('Deep Sleep', 'minutes'),
    'rem_duration': ('REM Sleep', 'minutes')
} 