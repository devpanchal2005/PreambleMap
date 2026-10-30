"""
src/map_view.py
Modular map building and country card rendering functions for PreambleMap.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

COLOR_MAP = {
    "Generic Rights & Freedoms (catch-all)": "#F0E442",
    "African Charter & Unity": "#009E73",
    "Statehood & History": "#0072B2",
    "Constituent Assembly & God": "#D55E00",
    "Commonwealth 'Whereas'": "#CC79A7",
    "Islamic & Arab Identity": "#E69F00",
    "Struggle & Socialism (mostly Asia)": "#56B4E9",
    "No preamble in dataset": "#B0B0B0",
    "Formula only (too short)": "#D9D9D9"
}

CATEGORY_ORDER = [
    "Generic Rights & Freedoms (catch-all)",
    "African Charter & Unity",
    "Statehood & History",
    "Constituent Assembly & God",
    "Commonwealth 'Whereas'",
    "Islamic & Arab Identity",
    "Struggle & Socialism (mostly Asia)",
    "No preamble in dataset",
    "Formula only (too short)"
]

ALIASES = {
    "Antigua and Barbuda": "Antigua and Barb.",
    "Saint Kitts and Nevis": "St. Kitts and Nevis",
    "Marshall Islands": "Marshall Is."
}

REVERSE_ALIASES = {v: k for k, v in ALIASES.items()}


def build_hover_text(row):
    conf = f"{row['conf_sub']:.3f}" if pd.notna(row['conf_sub']) else "N/A"
    ru = str(row['runnerup_label']) if pd.notna(row['runnerup_label']) else "None"
    b_val = "Yes" if row['boundary_flag'] is True else ("No" if row['boundary_flag'] is False else "N/A")
    return (
        f"<b>{row['Country']}</b><br>"
        f"Label: {row['map_category']}<br>"
        f"Confidence: {conf}<br>"
        f"Boundary: {b_val}<br>"
        f"Runner-up: {ru}"
    )


def prepare_map_data(world_gdf, countries_df):
    """
    Prepares merged geographic data excluding Antarctica with hover text.
    """
    # Drop Antarctica
    world = world_gdf[world_gdf["NAME"] != "Antarctica"].copy()

    # Assign map category
    def get_map_category(r):
        if r["category"] == "no_preamble_in_dataset":
            return "No preamble in dataset"
        elif r["category"] == "formula_only":
            return "Formula only (too short)"
        else:
            return str(r["cluster_label"])

    cdf = countries_df.copy()
    if "map_category" not in cdf.columns:
        cdf["map_category"] = cdf.apply(get_map_category, axis=1)

    cdf["ne_name"] = cdf["Country"].replace(ALIASES)
    merged = world.merge(cdf, left_on="NAME", right_on="ne_name", how="inner")
    merged["hover_text"] = merged.apply(build_hover_text, axis=1)

    return world, merged


def build_map(
    world_gdf,
    countries_df,
    focus_category="All categories",
    selected_country=None,
    mark_boundary=False,
    show_small_states=True
):
    """
    Constructs the Plotly Choropleth Figure with per-category traces and interactive styling.
    """
    world_filtered, merged = prepare_map_data(world_gdf, countries_df)
    world_geojson = world_filtered.__geo_interface__

    fig = go.Figure()

    # Add 9 choropleth traces in exact CATEGORY_ORDER
    for cat in CATEGORY_ORDER:
        cat_df = merged[merged["map_category"] == cat]

        if focus_category == "All categories" or focus_category is None:
            op = 1.0
        elif cat == focus_category:
            op = 1.0
        else:
            op = 0.15

        color = COLOR_MAP[cat]

        fig.add_trace(go.Choropleth(
            name=cat,
            locations=cat_df["NAME"],
            geojson=world_geojson,
            featureidkey="properties.NAME",
            z=[1] * len(cat_df),
            colorscale=[[0, color], [1, color]],
            showscale=False,
            marker=dict(
                line=dict(color="#444444", width=0.5),
                opacity=op
            ),
            customdata=cat_df["hover_text"],
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=True
        ))

    # Small states markers (< 10,000 sq km)
    if show_small_states:
        small_states = merged[merged["area_sqkm"] < 10000]
        for cat in CATEGORY_ORDER:
            cat_small = small_states[small_states["map_category"] == cat]
            if len(cat_small) > 0:
                op = 1.0 if (focus_category == "All categories" or cat == focus_category) else 0.15
                fig.add_trace(go.Scattergeo(
                    lon=cat_small["rep_lon"],
                    lat=cat_small["rep_lat"],
                    text=cat_small["hover_text"],
                    hoverinfo="text",
                    showlegend=False,
                    mode="markers",
                    marker=dict(
                        size=7,
                        color=COLOR_MAP[cat],
                        opacity=op,
                        line=dict(width=1, color="#222222")
                    )
                ))

    # Boundary markers
    if mark_boundary:
        boundary_countries = merged[merged["boundary_flag"] == True]
        if len(boundary_countries) > 0:
            fig.add_trace(go.Scattergeo(
                lon=boundary_countries["rep_lon"],
                lat=boundary_countries["rep_lat"],
                text=boundary_countries["hover_text"],
                hoverinfo="text",
                showlegend=False,
                mode="markers",
                marker=dict(
                    size=11,
                    symbol="circle-open",
                    line=dict(width=2, color="#111111")
                )
            ))

    # Selected country highlight trace
    if selected_country is not None:
        sel_row = merged[merged["Country"] == selected_country]
        if len(sel_row) > 0:
            sel_ne_name = sel_row["NAME"].iloc[0]
            sel_cat = sel_row["map_category"].iloc[0]
            sel_color = COLOR_MAP[sel_cat]
            fig.add_trace(go.Choropleth(
                name=f"Selected: {selected_country}",
                locations=[sel_ne_name],
                geojson=world_geojson,
                featureidkey="properties.NAME",
                z=[1],
                colorscale=[[0, sel_color], [1, sel_color]],
                showscale=False,
                marker=dict(
                    line=dict(color="white", width=3),
                    opacity=1.0
                ),
                customdata=sel_row["hover_text"],
                hovertemplate="%{customdata}<extra></extra>",
                showlegend=False
            ))

    fig.update_geos(
        projection_type="natural earth",
        showcountries=True,
        countrycolor="#bbbbbb",
        showcoastlines=True,
        coastlinecolor="#cccccc",
        showland=True,
        landcolor="#EDEDED",
        showocean=True,
        oceancolor="#DCE9F5",
        lataxis_range=[-58, 85]
    )

    fig.update_layout(
        height=600,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(
            title=dict(text="Categories"),
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5
        )
    )

    return fig


def render_country_card(row, countries_df):
    """
    Renders structured country card extracted from former Country tab.
    """
    selected_country = row["Country"]
    cat = row["category"]

    if pd.notna(row["note"]) and str(row["note"]).strip():
        st.info(f"**Note:** {row['note']}")
    if pd.notna(row["source_note"]) and str(row["source_note"]).strip():
        st.warning(f"**Source Note:** {row['source_note']}")

    if cat == "clustered":
        clab = row["cluster_label"]
        color = COLOR_MAP.get(clab, "#888888")
        chip = f"<span style='display:inline-block; width:14px; height:14px; background-color:{color}; border-radius:3px; vertical-align:middle; margin-right:6px;'></span>"

        st.markdown(f"### {chip}{selected_country} - {clab}", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Category:** {cat}")
        c1.markdown(f"**Continent:** {row['continent']}")
        c1.markdown(f"**Word Count:** {row['extracted_words']}")

        c2.markdown(f"**Confidence (conf_sub):** {row['conf_sub']:.4f}")
        c2.markdown(f"**Boundary Case:** {'Yes' if row['boundary_flag'] else 'No'}")
        c2.markdown(f"**Opening Type:** {row['opening_type']}")

        ru_score_str = f"{row['runnerup_score']:.4f}" if pd.notna(row['runnerup_score']) else "N/A"
        c3.markdown(f"**Runner-up Cluster:** {row['runnerup_label']}")
        c3.markdown(f"**Runner-up Score:** {ru_score_str}")

        with st.expander("Constitutional Preamble Text"):
            st.write(row["preamble_text"])

        st.markdown("#### 5 Nearest Preambles (Cosine Similarity on C_c)")
        neighbors = []
        for i in range(1, 6):
            nbr = row[f"neighbor_{i}"]
            sim = row[f"sim_{i}"]
            nbr_row = countries_df[countries_df["Country"] == nbr]
            nbr_label = nbr_row["cluster_label"].iloc[0] if len(nbr_row) > 0 and pd.notna(nbr_row["cluster_label"].iloc[0]) else "Unclustered"
            neighbors.append({
                "Rank": i,
                "Country": nbr,
                "Cluster": nbr_label,
                "Cosine Similarity": f"{sim:.4f}"
            })
        st.dataframe(pd.DataFrame(neighbors), use_container_width=True, hide_index=True)

    else:
        st.markdown(f"### {selected_country}")
        st.markdown(f"**Category:** {cat}")
        st.markdown(f"**Continent:** {row['continent']}")
        st.markdown(f"**Word Count:** {row['extracted_words']}")
        st.markdown(f"**Opening Type:** {row['opening_type']}")

        with st.expander("Constitutional Text in Dataset"):
            if cat == "no_preamble_in_dataset":
                st.write("")
            else:
                st.write(row["preamble_text"] if pd.notna(row["preamble_text"]) else "")
