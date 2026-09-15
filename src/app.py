import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import os
from pathlib import Path

st.set_page_config(
    page_title="SkyCity Restaurant Growth & Strategic Classification",
    page_icon="🍽️",
    layout="wide"
)

# --- Load Data ---
@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "final_processed_restaurant_data.csv"
    if not data_path.exists():
        data_path = Path("data/final_processed_restaurant_data.csv")
    return pd.read_csv(data_path)

df = load_data()

st.title("🍽️ SkyCity Restaurant Growth Modeling & Classification System")
st.markdown("Unsupervised Machine Learning & Strategic Decision Support Platform")

st.sidebar.header("Filter Criteria")

archetypes = st.sidebar.multiselect(
    "Select Archetype",
    options=df['Archetype'].unique(),
    default=df['Archetype'].unique()
)

if 'Subregion' in df.columns:
    subregions = st.sidebar.multiselect(
        "Select Subregion",
        options=df['Subregion'].unique(),
        default=df['Subregion'].unique()
    )
    filtered_df = df[(df['Archetype'].isin(archetypes)) & (df['Subregion'].isin(subregions))]
else:
    filtered_df = df[df['Archetype'].isin(archetypes)]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Restaurants", len(filtered_df))
col2.metric("Average GPI Score", f"{filtered_df['GPI'].mean():.1f}")
col3.metric("High-Growth Outliers", len(filtered_df[filtered_df['GPI'] >= 70]))
col4.metric("At-Risk Restaurants", len(filtered_df[filtered_df['COGSRate'] > 0.35]) if 'COGSRate' in filtered_df.columns else 0)

st.divider()

tab1, tab2, tab3 = st.tabs(["🗺️ Cluster Mapping", "📊 Archetype Benchmarks", "🔍 Restaurant Inspector"])

with tab1:
    st.subheader("2D Latent Feature Mapping (UMAP Dimensionality Reduction)")
    if 'UMAP1' in filtered_df.columns and 'UMAP2' in filtered_df.columns:
        fig_scatter = px.scatter(
            filtered_df,
            x='UMAP1',
            y='UMAP2',
            color='Archetype',
            size='GPI',
            hover_name='RestaurantName' if 'RestaurantName' in filtered_df.columns else None,
            title="Cluster Distribution across Strategic Space",
            template="plotly_white",
            height=550
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("UMAP coordinates not found in dataset.")

with tab2:
    st.subheader("Archetype Performance Comparison")
    
    fig_box = px.box(
        filtered_df,
        x='Archetype',
        y='GPI',
        color='Archetype',
        title="GPI Distribution per Archetype",
        template="plotly_white"
    )
    st.plotly_chart(fig_box, use_container_width=True)

with tab3:
    st.subheader("Individual Restaurant Deep-Dive")
    
    rest_names = filtered_df['RestaurantName'].unique() if 'RestaurantName' in filtered_df.columns else filtered_df.index
    selected_rest = st.selectbox("Select Restaurant", options=rest_names)
    
    if 'RestaurantName' in filtered_df.columns:
        rest_info = filtered_df[filtered_df['RestaurantName'] == selected_rest].iloc[0]
    else:
        rest_info = filtered_df.loc[selected_rest]
        
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.write(f"**Archetype:** {rest_info['Archetype']}")
        st.write(f"**Growth Potential Index (GPI):** {rest_info['GPI']} / 100")
        if 'GrowthFactor' in rest_info:
            st.write(f"**Growth Factor:** {rest_info['GrowthFactor']}")
        if 'COGSRate' in rest_info:
            st.write(f"**COGS Rate:** {rest_info['COGSRate']:.2%}")
            
        st.success(f"**Recommended Strategy:**\n{rest_info['Strategic_Recommendation']}")

    with c2:
        categories = ['Norm_Growth', 'Norm_CostResilience', 'Norm_ChannelBalance', 'Norm_Logistics']
        valid_cats = [cat for cat in categories if cat in rest_info]
        
        if valid_cats:
            values = [rest_info[cat] for cat in valid_cats]
            
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=[c.replace('Norm_', '') for c in valid_cats],
                fill='toself',
                name=str(selected_rest)
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                title="Operational Profile Radar"
            )
            st.plotly_chart(fig_radar, use_container_width=True)

with st.expander("📄 View Raw Processed Dataset"):
    st.dataframe(filtered_df)