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


# Set page config
st.set_page_config(
    page_title="Sleep Analysis Dashboard",
    page_icon="🌙",
    layout="wide"
)

# Title
st.title("Kimba Sleep Analysis")

# Load data
@st.cache_data
def load_data():
    return load_night_data()

df = load_data()

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

    # Start with the full dataframe
    filtered_df = df.copy()

    if exclude_alcohol:
        filtered_df = filtered_df[filtered_df['did_drink_alcohol'] != 1]
    if exclude_sick:
        filtered_df = filtered_df[filtered_df['is_sick'] != 1]

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
        # Display some statistics (move this block above the first plot)
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Users", len(filtered_df['user_id'].unique()))
        with col2:
            st.metric("Total Nights", len(filtered_df))
        with col3:
            active_users = len(get_active_users(filtered_df, min_active_nights))
            st.metric("Valid Users (≥{} nights)".format(min_active_nights), active_users)

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
            title=f"{selected_feature}: Per-user Δ (Active – Control)",
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
        st.plotly_chart(fig_box, use_container_width=True)
        # ⬆️ END NEW FIGURE BLOCK  ------------------------------------------------------
    else:
        st.error("Not enough data to create the plot for the selected feature")
else:
    st.error("Failed to load data. Please check if the CSV file exists and is properly formatted.")

