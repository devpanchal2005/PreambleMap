"""
src/map_view.py
Modular map building and country card rendering functions for PreambleMap.
"""

import html
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
    "No preamble in dataset": "#9A9A9A",
    "Formula only (too short)": "#CFCFCF"
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


def fit_label(row):
    """
    Computes qualitative fit label from boundary_flag and conf_sub.
    """
    if row.get("boundary_flag") is True:
        return "Borderline"
    conf = row.get("conf_sub")
    if pd.notna(conf) and conf >= 0.75:
        return "Clear"
    return "Moderate"


def build_hover_text(row):
    """
    Builds clean tooltip without internal variable names.
    """
    country = row["Country"]
    cat = row["category"]
    if cat == "clustered":
        label = row["cluster_label"]
        fit = fit_label(row)
        return f"<b>{country}</b><br>Group: {label}<br>Fit: {fit}"
    else:
        label = row["map_category"]
        return f"<b>{country}</b><br>Group: {label}"


def prepare_map_data(world_gdf, countries_df):
    """
    Prepares merged geographic data excluding Antarctica with hover text.
    """
    world = world_gdf[world_gdf["NAME"] != "Antarctica"].copy()

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


def country_from_map_state(map_state, ne_to_country):
    """
    Parses Plotly chart selection state to extract (country, token) or (None, None).
    """
    if not map_state or not isinstance(map_state, dict):
        return None, None
    sel = map_state.get("selection")
    if not sel or not isinstance(sel, dict):
        return None, None
    points = sel.get("points")
    if not points or not isinstance(points, (list, tuple)) or len(points) == 0:
        return None, None

    pt = points[0]
    if not isinstance(pt, dict):
        return None, None

    curve_number = pt.get("curve_number")
    point_index = pt.get("point_index")
    loc = pt.get("location")
    cdata = pt.get("customdata")

    # If location is present, take location if in ne_to_country
    if loc is not None:
        if loc in ne_to_country:
            country = ne_to_country[loc]
            token = (curve_number, point_index, loc)
            return country, token
        else:
            return None, None

    # Otherwise, check customdata
    cdata_str = None
    if isinstance(cdata, str):
        cdata_str = cdata
    elif isinstance(cdata, (list, tuple)) and len(cdata) == 1 and isinstance(cdata[0], str):
        cdata_str = cdata[0]

    if cdata_str is not None:
        if cdata_str in ne_to_country.values():
            token = (curve_number, point_index, cdata_str)
            return cdata_str, token

    return None, None


def map_chart_key(session):
    """
    Returns rotating Plotly chart key based on map_gen.
    """
    return f"map_{session.get('map_gen', 0)}"


def sync_map_click(session, ne_to_country):
    """
    Synchronizes Plotly map point selection into session state.
    Never raises.
    """
    try:
        current_key = map_chart_key(session)
        state = session.get(current_key)
        country, token = country_from_map_state(state, ne_to_country)
        if country is not None and token != session.get("map_last_token"):
            session["map_last_token"] = token
            session["find_country"] = country
    except Exception:
        pass


def reset_map_selection(session):
    """
    Rotates chart generation key and resets last click token on user dropdown change.
    """
    session["map_gen"] = session.get("map_gen", 0) + 1
    session["map_last_token"] = None


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

        if cat == "Formula only (too short)":
            line_color = "#6E6E6E"
            line_width = 1.2
        else:
            line_color = "#5A5346"
            line_width = 0.4

        fig.add_trace(go.Choropleth(
            name=cat,
            locations=cat_df["NAME"],
            geojson=world_geojson,
            featureidkey="properties.NAME",
            z=[1] * len(cat_df),
            colorscale=[[0, color], [1, color]],
            showscale=False,
            marker=dict(
                line=dict(color=line_color, width=line_width),
                opacity=op
            ),
            selected=dict(marker=dict(opacity=op)),
            unselected=dict(marker=dict(opacity=op)),
            customdata=cat_df["hover_text"],
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False
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
                    customdata=cat_small["Country"],
                    hoverinfo="text",
                    showlegend=False,
                    mode="markers",
                    marker=dict(
                        size=6,
                        color=COLOR_MAP[cat],
                        opacity=op,
                        line=dict(width=0.5, color="#1F1B16")
                    ),
                    selected=dict(marker=dict(opacity=op)),
                    unselected=dict(marker=dict(opacity=op))
                ))

    # Boundary markers
    if mark_boundary:
        boundary_countries = merged[merged["boundary_flag"] == True]
        if len(boundary_countries) > 0:
            fig.add_trace(go.Scattergeo(
                lon=boundary_countries["rep_lon"],
                lat=boundary_countries["rep_lat"],
                text=boundary_countries["hover_text"],
                customdata=boundary_countries["Country"],
                hoverinfo="text",
                showlegend=False,
                mode="markers",
                marker=dict(
                    size=11,
                    symbol="circle-open",
                    line=dict(width=2, color="#111111")
                ),
                selected=dict(marker=dict(opacity=1.0)),
                unselected=dict(marker=dict(opacity=1.0))
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
                    line=dict(color="#1F1B16", width=3),
                    opacity=1.0
                ),
                selected=dict(marker=dict(opacity=1.0)),
                unselected=dict(marker=dict(opacity=1.0)),
                customdata=sel_row["hover_text"],
                hovertemplate="%{customdata}<extra></extra>",
                showlegend=False
            ))

    fig.update_geos(
        projection_type="natural earth",
        showcountries=True,
        countrycolor="#5A5346",
        showcoastlines=True,
        coastlinecolor="#5A5346",
        showland=True,
        landcolor="#F1ECDD",
        showocean=True,
        oceancolor="#DDE6EC",
        bgcolor="rgba(0,0,0,0)",
        showframe=False,
        lataxis_range=[-58, 85]
    )

    fig.update_layout(
        height=560,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )

    return fig


def render_legend(countries_df):
    """
    Renders custom HTML legend for the right panel in Map tab.
    """
    lines = []
    lines.append('<div style="line-height:1.9; font-size:14px;">')
    lines.append('<div style="font-weight:bold; font-size:16px; margin-bottom:6px;">Groups</div>')
    for c in range(7):
        subset = countries_df[countries_df["final_cluster"] == c]
        label = subset["cluster_label"].iloc[0]
        count = len(subset)
        color = COLOR_MAP[label]
        escaped_label = html.escape(label)
        lines.append(
            f'<div><span style="display:inline-block; width:14px; height:14px; background-color:{color}; border:1px solid #1F1B16; margin-right:8px; vertical-align:middle;"></span>{escaped_label} ({count})</div>'
        )
    lines.append('<div style="font-size:12px; color:#666; margin-top:8px; margin-bottom:2px;">Not clustered</div>')
    for cat, label_str in [("no_preamble_in_dataset", "No preamble in dataset"), ("formula_only", "Formula only (too short)")]:
        count = int((countries_df["category"] == cat).sum())
        color = COLOR_MAP[label_str]
        escaped_label = html.escape(label_str)
        lines.append(
            f'<div><span style="display:inline-block; width:14px; height:14px; background-color:{color}; border:1px solid #1F1B16; margin-right:8px; vertical-align:middle;"></span>{escaped_label} ({count})</div>'
        )
    lines.append('</div>')
    legend_html = "".join(lines)
    st.markdown(legend_html, unsafe_allow_html=True)
    return legend_html


def render_country_card(row, countries_df):
    """
    Renders compact country card without internal variable names.
    """
    selected_country = row["Country"]
    cat = row["category"]

    if cat == "clustered":
        clab = row["cluster_label"]
        color = COLOR_MAP.get(clab, "#9A9A9A")
    else:
        color = "#9A9A9A"

    chip = f"<span style='display:inline-block; width:14px; height:14px; background-color:{color}; border-radius:3px; vertical-align:middle; margin-right:8px;'></span>"
    st.markdown(f"### {chip}{selected_country}", unsafe_allow_html=True)

    if pd.notna(row.get("note")) and str(row["note"]).strip():
        st.info(str(row["note"]))
    if pd.notna(row.get("source_note")) and str(row["source_note"]).strip():
        st.warning(str(row["source_note"]))

    continent = str(row.get("continent", "")).strip() if pd.notna(row.get("continent")) else ""

    if cat == "clustered":
        clab = row["cluster_label"]
        fit = fit_label(row)
        conf = float(row["conf_sub"]) if pd.notna(row.get("conf_sub")) else 0.0
        ru_label = str(row["runnerup_label"]) if pd.notna(row.get("runnerup_label")) else "None"

        raw_op = str(row.get("opening_type", ""))
        if raw_op == "we_the_people":
            op_display = "We the people..."
        elif raw_op == "whereas":
            op_display = "Whereas..."
        elif raw_op == "in_the_name_of":
            op_display = "In the name of..."
        else:
            op_display = "Other opening"

        words = row.get("extracted_words")

        c_left, c_right = st.columns([1, 1.3], vertical_alignment="top")
        with c_left:
            st.markdown(f"Group: {clab}")
            st.markdown(
                f"Fit: {fit} ({conf:.2f})",
                help="How consistently this country lands in this group across repeated clustering runs. Borderline means it sits close to another group."
            )
            st.markdown(f"Next closest group: {ru_label}")
            st.markdown(f"Opens with: {op_display}")
            st.markdown(f"Length: {words} words")
            if continent:
                st.markdown(f"Region: {continent}")

        with c_right:
            st.markdown(
                "**Most similar preambles**",
                help="Similarity of wording after each country's own name is masked. 1.00 would be identical."
            )
            neighbors = []
            for i in range(1, 6):
                nbr = row[f"neighbor_{i}"]
                sim = float(row[f"sim_{i}"])
                nbr_row = countries_df[countries_df["Country"] == nbr]
                if len(nbr_row) > 0 and pd.notna(nbr_row["cluster_label"].iloc[0]):
                    nbr_group = str(nbr_row["cluster_label"].iloc[0])
                else:
                    nbr_group = "Not clustered"
                neighbors.append({
                    "Country": nbr,
                    "Group": nbr_group,
                    "Similarity": sim
                })
            st.dataframe(
                pd.DataFrame(neighbors),
                column_config={
                    "Similarity": st.column_config.ProgressColumn(
                        min_value=0,
                        max_value=1,
                        format="%.2f"
                    )
                },
                hide_index=True,
                width="stretch"
            )

        with st.expander("Read the preamble"):
            st.write(row["preamble_text"] if pd.notna(row.get("preamble_text")) else "")

    else:
        st.markdown("Group: Not clustered")
        if continent:
            st.markdown(f"Region: {continent}")
        if cat == "formula_only":
            words = row.get("extracted_words")
            st.markdown(f"Length: {words} words")

        if cat != "no_preamble_in_dataset":
            with st.expander("Read the preamble"):
                st.write(row["preamble_text"] if pd.notna(row.get("preamble_text")) else "")
