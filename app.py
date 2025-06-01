import streamlit as st
import pandas as pd
from load_data import load_night_data, get_active_users
from analysis import create_comparison_plot, FEATURES_TO_PLOT
import io
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from palette import palette
from scent_map import scent_map
from datetime import datetime


# Set page config
st.set_page_config(
    page_title="Kimba Data Analysis",
    page_icon="🌙",
    layout="wide"
)

# Define which direction is positive for each feature
POSITIVE_DIRECTIONS = {
    'heart_rate_mean': 'decrease',  # Lower heart rate is better
    'bbi_mean': 'increase',  # Higher BBI (heart rate variability) is better
    'hrv_std_hr_mean': 'increase',  # Higher HRV is better
    'stress_level_mean': 'decrease',  # Lower stress is better
    'sleep_efficiency': 'increase',  # Higher efficiency is better
    'total_sleep_time': 'increase',  # More sleep is better
    'waso_duration': 'decrease',  # Less wake after sleep onset is better
    'waso_count': 'decrease',  # Fewer wake episodes is better
    'sleep_latency': 'decrease',  # Faster sleep onset is better
    'acc_magnitude_mean': 'decrease',  # Less movement is better
    'respiration_rate_mean': 'neutral',  # No clear direction
    'oxygen_level_mean': 'increase',  # Higher SpO2 is better
    'deep_duration': 'increase',  # More deep sleep is better
    'rem_duration': 'increase',  # More REM sleep is better
}

def calculate_positive_effect_percentage(filtered_df, selected_feature, min_active_nights):
    """
    Calculate the percentage of users where the effect was positive
    """
    # Filter for valid users (same as main analysis)
    active_nights = filtered_df[filtered_df['diffuser_category'] == 'active'].groupby('user_id').size()
    valid_users = active_nights[active_nights >= min_active_nights].index
    user_filtered_df = filtered_df[filtered_df['user_id'].isin(valid_users)]
    
    # Calculate user means for control and active
    user_means = (
        user_filtered_df
        .groupby(['user_id', 'diffuser_category'])[selected_feature]
        .mean()
        .reset_index()
    )
    
    # Pivot to get control and active columns
    slopes = (
        user_means
        .pivot(index='user_id', columns='diffuser_category', values=selected_feature)
        .reset_index()
    )
    
    # Only include users who have both control and active data
    slopes = slopes.dropna(subset=['control', 'active'])
    
    if len(slopes) == 0:
        return 0, 0, 0
    
    # Calculate delta (active - control)
    slopes['delta'] = slopes['active'] - slopes['control']
    
    # Determine positive effects based on feature direction
    direction = POSITIVE_DIRECTIONS.get(selected_feature, 'neutral')
    
    if direction == 'increase':
        # Positive effect means delta > 0 (active > control)
        positive_users = (slopes['delta'] > 0).sum()
    elif direction == 'decrease':
        # Positive effect means delta < 0 (active < control)
        positive_users = (slopes['delta'] < 0).sum()
    else:  # neutral
        # For neutral features, we can't determine positive direction
        return len(slopes), 0, 0
    
    total_users = len(slopes)
    percentage = (positive_users / total_users * 100) if total_users > 0 else 0
    
    return total_users, positive_users, percentage

def identify_siren_nights(df):
    """
    Identify nights that overlap with siren events
    """
    # Define siren events (date and time when sirens occurred)
    siren_events = [
        ("14/01/2025", "03:00"),
        ("20/03/2025", "04:00"),
        ("21/03/2025", "22:30"),
        ("23/03/2025", "07:23"),
        ("18/04/2025", "06:36"),
        ("23/04/2025", "03:58"),
        ("26/04/2025", "02:45"),
        ("27/04/2025", "04:50"),
        ("02/05/2025", "05:30"),
        ("03/05/2025", "06:23"),
        ("15/05/2025", "21:10"),
        ("18/05/2025", "02:00"),
        ("22/05/2025", "03:00"),
        ("23/05/2025", "04:12")
    ]
    
    # Convert siren events to datetime objects
    siren_datetimes = []
    for date_str, time_str in siren_events:
        try:
            # Parse DD/MM/YYYY format
            dt = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M")
            siren_datetimes.append(dt)
        except ValueError:
            continue
    
    # Add siren flag column
    df['is_siren_night'] = 0
    
    # Check each night against siren events
    for idx, row in df.iterrows():
        try:
            # Convert start_date and end_date to datetime
            start_dt = pd.to_datetime(row['start_date'])
            end_dt = pd.to_datetime(row['end_date'])
            
            # Check if any siren event falls within this sleep period
            for siren_dt in siren_datetimes:
                if start_dt <= siren_dt <= end_dt:
                    df.at[idx, 'is_siren_night'] = 1
                    break
                    
        except (ValueError, TypeError):
            # Skip rows with invalid dates
            continue
    
    return df

# Title
st.title("Kimba Sleep Analysis")

# Add cache refresh button
if st.button("🔄 Refresh Data (Clear Cache)"):
    st.cache_data.clear()
    st.rerun()
    
# Load data
def load_data():
    """Load and process the night data with siren detection"""
    df = load_night_data()
    if df is not None:
        # Identify siren nights
        df = identify_siren_nights(df)
        
        # Remove specific users from the dataset
        excluded_users = ['1.0', '3.0', '5.0', '39.0']  # Match the float format in the dataset
        df = df[~df['user_id'].isin(excluded_users)]
        
    return df

# Cache the data loading
@st.cache_data
def get_cached_data():
    return load_data()

df = get_cached_data()

if df is not None:
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Feature selection
    selected_feature = st.sidebar.selectbox(
        "Select Feature to Analyze",
        options=list(FEATURES_TO_PLOT.keys()),
        format_func=lambda x: FEATURES_TO_PLOT[x][0]
    )
    
    # Minimum active nights filter
    min_active_nights = st.sidebar.slider(
        "Minimum Active Nights",
        min_value=1,
        max_value=20,
        value=5
    )

    # Sidebar filters
    exclude_alcohol = st.sidebar.checkbox("Exclude nights with alcohol", value=True)
    exclude_sick = st.sidebar.checkbox("Exclude nights when sick", value=True)
    exclude_siren = st.sidebar.checkbox("Exclude siren nights", value=True)

    # Get all unique scent serials from active nights only, filter out -1 and 21
    all_scent_serials = sorted(
        s for s in df[df['diffuser_category'] == 'active']['dominant_scent'].dropna().unique()
        if s not in [-1, 21]
    )
    # Map serials to names for display
    scent_options = {serial: scent_map.get(int(serial), str(serial)) for serial in all_scent_serials}
    selected_scent_serials = st.sidebar.multiselect(
        "Filter by Scent",
        options=list(scent_options.keys()),
        default=[],
        format_func=lambda x: scent_options[x]
    )

    # User ID filter - get all unique user IDs and start with all selected
    all_user_ids = sorted(df['user_id'].unique())
    
    # Initialize session state if it doesn't exist
    if 'user_selection' not in st.session_state:
        st.session_state.user_selection = []
    
    # The multiselect dropdown
    selected_user_ids = st.sidebar.multiselect(
        "Filter by User ID",
        options=all_user_ids,
        default=st.session_state.user_selection,
        help="Select/deselect specific users. Use buttons below for quick select all/none."
    )
    
    # Add buttons for select all / deselect all
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Select All", key="select_all_users"):
            st.session_state.user_selection = all_user_ids
            st.rerun()
    with col2:
        if st.button("Remove All", key="deselect_all_users"):
            st.session_state.user_selection = []
            st.rerun()
    
    # Only update session state if no button was clicked
    if not st.session_state.get('button_clicked', False):
        st.session_state.user_selection = selected_user_ids

    # Start with the full dataframe
    filtered_df = df.copy()

    # Filter by selected user IDs
    if selected_user_ids:  # Only filter if some users are selected
        filtered_df = filtered_df[filtered_df['user_id'].isin(selected_user_ids)]

    if exclude_alcohol:
        filtered_df = filtered_df[filtered_df['did_drink_alcohol'] != 1]
    if exclude_sick:
        filtered_df = filtered_df[filtered_df['is_sick'] != 1]
    if exclude_siren:
        filtered_df = filtered_df[filtered_df['is_siren_night'] != 1]

    # Only filter active nights by scent if any are selected
    if selected_scent_serials:
        is_active = filtered_df['diffuser_category'] == 'active'
        scent_mask = filtered_df['dominant_scent'].isin(selected_scent_serials)
        filtered_df = pd.concat([
            filtered_df[~is_active],
            filtered_df[is_active & scent_mask]
        ])

    # Now apply the active nights threshold filter
    active_nights = filtered_df[filtered_df['diffuser_category'] == 'active'].groupby('user_id').size()
    valid_users = active_nights[active_nights >= min_active_nights].index
    filtered_df = filtered_df[filtered_df['user_id'].isin(valid_users)]

    # Create and display the plot (very wide)
    display_name, unit = FEATURES_TO_PLOT[selected_feature]
    fig = create_comparison_plot(filtered_df, selected_feature, display_name, unit, min_active_nights, fig_width=2400)
    
    if fig is not None:
        # Calculate positive effect percentage
        total_users_with_data, positive_users, positive_percentage = calculate_positive_effect_percentage(
            filtered_df, selected_feature, min_active_nights
        )
        
        # Display some statistics (move this block above the first plot)
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Users", len(filtered_df['user_id'].unique()))
        with col2:
            st.metric("Total Nights", len(filtered_df))
        with col3:
            active_users = len(get_active_users(filtered_df, min_active_nights))
            st.metric("Valid Users (≥{} nights)".format(min_active_nights), active_users)
        with col4:
            direction = POSITIVE_DIRECTIONS.get(selected_feature, 'neutral')
            if direction != 'neutral':
                direction_text = "↑" if direction == 'increase' else "↓"
                st.metric(
                    f"Positive Effect {direction_text}", 
                    f"{positive_users}/{total_users_with_data} ({positive_percentage:.1f}%)"
                )
            else:
                st.metric("Positive Effect", "N/A (neutral)")

        st.plotly_chart(fig, use_container_width=False)
        
        # ⬇️ START NEW FIGURE BLOCK  ----------------------------------------------------
        # Helper ─ filter the DataFrame the same way the main bar-chart does
        active_nights  = filtered_df[filtered_df['diffuser_category'] == 'active'].groupby('user_id').size()
        valid_users    = active_nights[active_nights >= min_active_nights].index
        filtered_df    = filtered_df[filtered_df['user_id'].isin(valid_users)]

        # --------------------------------------------------------------------------- #
        # Build a tidy table: one mean value per (user, condition)
        user_means = (
            filtered_df
            .groupby(['user_id', 'diffuser_category'])[selected_feature]
            .mean()
            .reset_index()
        )

        # Pivot so we have "control" and "active" columns
        slopes = (
            user_means
              .pivot(index='user_id', columns='diffuser_category', values=selected_feature)
              .reset_index()
              .assign(delta=lambda d: d['active'] - d['control'])
              .astype({'user_id': str})
        )

        # Sort by absolute Δ so longest bars go to the top
        user_order = (slopes
                      .assign(abs_delta=lambda d: d['delta'].abs())
                      .sort_values('abs_delta', ascending=False)['user_id']
                      .tolist())

        # --------------------------------------------------------------------------- #
        # 3️⃣  Δ-scatter: one dot per user, dashed zero line
        fig_delta = go.Figure()

        # Colour by sign of the delta (light-blue = improvement, dark-blue = decline)
        colours = np.where(slopes['delta'] > 0,
                           palette["delta_pos"],
                           palette["delta_neg"])

        fig_delta.add_trace(go.Scatter(
            x=slopes['user_id'], y=slopes['delta'],
            mode='markers', marker=dict(color=colours, size=10),
            hovertemplate='User %{x}<br>Δ = %{y:.2f}<extra></extra>'
        ))

        # horizontal dashed zero line
        fig_delta.add_shape(
            type='line', x0=-0.5, x1=len(slopes['user_id'])-0.5,
            y0=0, y1=0,
            line=dict(color='black', dash='dash')
        )

        # ─── Mean-Δ horizontal line ──────────────────────────────────────────────
        mean_delta = float(slopes['delta'].mean(skipna=True))
        mean_color = palette["delta_pos"] if mean_delta > 0 else palette["delta_neg"]

        fig_delta.add_shape(
            type='line',
            x0=-0.5,                # full width of category axis
            x1=len(slopes['user_id']) - 0.5,
            y0=mean_delta,
            y1=mean_delta,
            line=dict(color=mean_color, dash='dot', width=2)
        )

        fig_delta.add_annotation(
            x=len(slopes['user_id']) - 1,
            y=mean_delta,
            xref='x', yref='y',
            text=f"<b>mean Δ = {mean_delta:+.2f}</b>",
            showarrow=False,
            font=dict(color=mean_color, size=18, family='Arial')
        )
        # ─────────────────────────────────────────────────────────────────────────

        # --- Make the y-axis symmetric -------------------------------------------------
        max_abs = float(np.nanmax(np.abs(slopes['delta'])))  # largest |Δ|
        pad     = max_abs * 0.05                            # 5 % breathing room
        fig_delta.update_yaxes(
            range=[-max_abs - pad, max_abs + pad],          # same distance above & below 0
            zeroline=True, zerolinewidth=1, zerolinecolor='black'
        )
        ## ----------------------------------------------------------------------------- 

        fig_delta.update_layout(
            title=f"{selected_feature}: Per-user delta (Active – Control)",
            xaxis=dict(title='User ID', type='category', tickangle=-45),
            yaxis_title='Δ',
            template='plotly_white',
            height=400,
            margin=dict(l=60, r=40, t=60, b=120)
        )
        st.plotly_chart(fig_delta, use_container_width=True)

        # ───────────────────────────────────────────────────────────────────────────── #
        # 2️⃣  Box-and-scatter distribution per condition
        fig_box = px.box(
            filtered_df, x='diffuser_category', y=selected_feature, points='all',
            color='diffuser_category',
            category_orders={'diffuser_category': ['control', 'active', 'placebo']},
            color_discrete_map={
               "control" : palette["control"],
               "active"  : palette["active"],
               "placebo" : palette["placebo"]
            },
            title=f"{selected_feature}: Distribution"
        )
        
        # Calculate medians for each condition and add connecting line
        medians = []
        x_positions = []
        conditions = ['control', 'active', 'placebo']
        
        for condition in conditions:
            condition_data = filtered_df[filtered_df['diffuser_category'] == condition][selected_feature]
            if len(condition_data) > 0:
                median_val = condition_data.median()
                medians.append(median_val)
                x_positions.append(condition)
        
        # Add median connecting line
        if len(medians) > 1:
            fig_box.add_trace(go.Scatter(
                x=x_positions,
                y=medians,
                mode='lines+markers',
                line=dict(color='red', width=2, dash='dot'),
                marker=dict(color='red', size=8),
                name='Median Trend',
                hovertemplate='%{x}: %{y:.2f}<extra></extra>'
            ))
        
        st.plotly_chart(fig_box, use_container_width=True)
        # ⬆️ END NEW FIGURE BLOCK  ------------------------------------------------------
    else:
        st.error("Not enough data to create the plot for the selected feature")
else:
    st.error("Failed to load data. Please check if the CSV file exists and is properly formatted.")

