import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from palette import palette

def create_comparison_plot(df, feature, display_name, unit, min_active_nights=5, fig_width=2400, positive_directions=None, selected_stats=None):
    """
    Create an interactive comparison plot using Plotly Express
    """
    if selected_stats is None:
        selected_stats = ['mean_diff', 'delta_mean']
    
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
    
    # Calculate positive effect percentage if positive_directions is provided
    positive_effect_text = ""
    if positive_directions and feature in positive_directions:
        # Calculate deltas for positive effect
        deltas = np.array(y_active) - np.array(y_control)
        direction = positive_directions[feature]
        
        if direction == 'increase':
            positive_users = (deltas > 0).sum()
        elif direction == 'decrease':
            positive_users = (deltas < 0).sum()
        else:  # neutral
            positive_users = 0
        
        total_users = len(deltas)
        if direction != 'neutral' and total_users > 0:
            percentage = (positive_users / total_users) * 100
            direction_symbol = "↑" if direction == 'increase' else "↓"
            positive_effect_text = f"Positive Effect {direction_symbol}: {positive_users}/{total_users} ({percentage:.1f}%)"
        else:
            positive_effect_text = "Positive Effect: N/A (neutral)"
    
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
    
    # Calculate mean values for reference lines
    control_mean = np.nanmean(y_control)
    active_mean = np.nanmean(y_active)
    
    # Conditionally add stats to legend based on selection
    if 'control_mean' in selected_stats:
        # Add control mean line as a trace (appears in legend)
        fig.add_trace(go.Scatter(
            x=[custom_labels[0], custom_labels[-1]],
            y=[control_mean, control_mean],
            mode='lines',
            line=dict(color=palette["control"], dash='dash', width=3),
            name='Control Mean',
            hoverinfo='skip',
            showlegend=True
        ))
    
    if 'active_mean' in selected_stats:
        # Add active mean line as a trace (appears in legend)  
        fig.add_trace(go.Scatter(
            x=[custom_labels[0], custom_labels[-1]],
            y=[active_mean, active_mean],
            mode='lines',
            line=dict(color=palette["active"], dash='dash', width=3),
            name='Active Mean',
            hoverinfo='skip',
            showlegend=True
        ))

    if 'mean_diff' in selected_stats:
        # Add mean percentage difference as a legend entry (invisible trace)
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=0),
            name=mean_diff_text,
            showlegend=True,
            hoverinfo='skip'
        ))

    if 'delta_mean' in selected_stats:
        # Add delta mean (absolute difference) as a legend entry
        delta_mean = active_mean - control_mean
        delta_mean_text = f"Delta Mean: {delta_mean:+.2f} {unit}"
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=0),
            name=delta_mean_text,
            showlegend=True,
            hoverinfo='skip'
        ))

    if 'positive_effect' in selected_stats and positive_effect_text:
        # Add positive effect as a legend entry
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=0),
            name=positive_effect_text,
            showlegend=True,
            hoverinfo='skip'
        ))

    # Add horizontal dashed lines for mean values (always show the lines, regardless of legend)
    # Control mean line
    fig.add_shape(
        type='line',
        x0=-0.5, x1=len(user_ids)-0.5,
        y0=control_mean, y1=control_mean,
        line=dict(color=palette["control"], dash='dash', width=2),
        name="Control Mean"
    )
    
    # Active mean line  
    fig.add_shape(
        type='line',
        x0=-0.5, x1=len(user_ids)-0.5,
        y0=active_mean, y1=active_mean,
        line=dict(color=palette["active"], dash='dash', width=2),
        name=" Active Mean"
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
        legend=dict(
            orientation='h', 
            yanchor='bottom', 
            y=1.02, 
            xanchor='center', 
            x=0.5,
            font=dict(size=14)
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, tickangle=-45, type='category'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey'),
        margin=dict(l=40, r=40, t=100, b=180)
    )
    
    # Set y-axis range to start from below the minimum value
    min_value = min(min(y_control), min(y_active))
    max_value = max(max(y_control), max(y_active))
    value_range = max_value - min_value
    bottom_padding = value_range * 0.1  # 10% padding below
    top_padding = value_range * 0.15    # 15% padding above
    
    fig.update_yaxes(range=[min_value - bottom_padding, max_value + top_padding])
    
    return fig

# Define the features to plot with their display names and units
FEATURES_TO_PLOT = {
    'heart_rate_mean': ('Heart Rate Mean', 'bpm'),
    'bbi_mean': ('BBI Mean', 'ms'),
    'hrv_std_hr_mean': ('RMSSD (HRV)', 'ms'),
    'stress_level_mean': ('Stress Level Mean', 'level'),
    'sleep_efficiency': ('Sleep Efficiency', '%'),
    'total_sleep_time': ('Total Sleep Time', 'minutes'),
    'waso_duration': ('WASO Duration', 'minutes'),
    'waso_count': ('WASO Count', 'count'),
    'sleep_latency': ('Sleep Latency', 'minutes'),
    'acc_magnitude_mean': ('Movement Vector', 'm/s²'),
    'respiration_rate_mean': ('Respiration Rate', 'breaths/min'),
    'oxygen_level_mean': ('SpO2', '%'),
    'deep_duration': ('Deep Sleep', 'minutes'),
    'rem_duration': ('REM Sleep', 'minutes'),
} 