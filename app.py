"""
PreambleMap: Clustering Constitutional Preambles by Their Wording
Production Streamlit Application
"""

import json
import os
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page configuration
st.set_page_config(page_title="PreambleMap", layout="wide")

# Exact color mapping
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

US_SAMPLE = (
    "We the People of the United States, in Order to form a more perfect Union, "
    "establish Justice, insure domestic Tranquility, provide for the common defence, "
    "promote the general Welfare, and secure the Blessings of Liberty to ourselves and our Posterity, "
    "do ordain and establish this Constitution for the United States of America."
)

ISLAMIC_SAMPLE = (
    "In the name of God, the Merciful, the Compassionate, we, the people of this republic, "
    "proclaim our faith in Islam, social justice and Arab unity."
)


@st.cache_data
def load_app_data():
    base_dir = Path(__file__).resolve().parent
    countries_path = base_dir / "data" / "app" / "countries.csv"
    terms_path = base_dir / "data" / "app" / "terms.csv"
    clusters_path = base_dir / "data" / "app" / "clusters.csv"
    assets_path = base_dir / "data" / "app" / "model_assets.npz"
    val_path = base_dir / "data" / "app" / "validation.json"

    countries_df = pd.read_csv(countries_path)
    countries_df["final_cluster"] = countries_df["final_cluster"].astype("Int64")

    terms_df = pd.read_csv(terms_path)
    clusters_df = pd.read_csv(clusters_path)

    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    with np.load(assets_path) as npz:
        assets = {k: npz[k] for k in npz.files}

    def get_map_category(r):
        if r["category"] == "no_preamble_in_dataset":
            return "No preamble in dataset"
        elif r["category"] == "formula_only":
            return "Formula only (too short)"
        else:
            return str(r["cluster_label"])

    countries_df["map_category"] = countries_df.apply(get_map_category, axis=1)

    return countries_df, terms_df, clusters_df, val_data, assets


@st.cache_data
def load_world_geometry():
    base_dir = Path(__file__).resolve().parent
    shp_path = base_dir / "data" / "external" / "ne_50m_admin_0_countries" / "ne_50m_admin_0_countries.shp"
    world = gpd.read_file(shp_path)

    # Compute area in EPSG:6933 for small states threshold (< 10,000 sq km)
    world_proj = world.to_crs("EPSG:6933")
    world["area_sqkm"] = world_proj.geometry.area / 1e6

    # Geographic representative points (lon/lat)
    rep_pts = world.geometry.representative_point()
    world["rep_lon"] = rep_pts.x
    world["rep_lat"] = rep_pts.y

    # Simplify geometry with tolerance 0.02 degrees, preserve_topology=True
    world["geometry"] = world.geometry.simplify(tolerance=0.02, preserve_topology=True)

    return world


@st.cache_resource
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-mpnet-base-v2")


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


# Load data
countries_df, terms_df, clusters_df, val_data, assets = load_app_data()
world_gdf = load_world_geometry()

# Prepare merged dataset for map
countries_for_map = countries_df.copy()
countries_for_map["ne_name"] = countries_for_map["Country"].replace(ALIASES)
merged = world_gdf.merge(countries_for_map, left_on="NAME", right_on="ne_name", how="inner")
merged["hover_text"] = merged.apply(build_hover_text, axis=1)

# Application Header
st.title("PreambleMap: Clustering Constitutional Preambles by Their Wording")
st.markdown("Labels describe the wording and drafting tradition of preambles. They are not claims about a country's politics or values.")

# Sidebar controls
with st.sidebar:
    st.header("Map Controls")
    selected_categories = st.multiselect(
        "Show Categories",
        options=CATEGORY_ORDER,
        default=CATEGORY_ORDER,
        key="map_cat_filter"
    )
    mark_boundary = st.checkbox(
        "Mark boundary countries",
        value=False,
        key="map_mark_boundary"
    )
    show_small_states = st.checkbox(
        "Show markers for small states",
        value=True,
        key="map_show_small"
    )

# Main navigation tabs
tab_map, tab_clusters, tab_country, tab_predict, tab_method = st.tabs([
    "Map",
    "Clusters",
    "Country",
    "Try your own text",
    "Method and limits"
])

# -------------------------------------------------------------
# TAB 1: MAP
# -------------------------------------------------------------
with tab_map:
    selected_set = set(selected_categories)
    merged_filtered = merged[merged["map_category"].isin(selected_set)]

    if len(merged_filtered) == 0:
        st.warning("No categories selected. Please select at least one category in the sidebar.")
    else:
        fig = px.choropleth(
            merged_filtered,
            geojson=world_gdf.__geo_interface__,
            locations="NAME",
            featureidkey="properties.NAME",
            color="map_category",
            category_orders={"map_category": CATEGORY_ORDER},
            color_discrete_map=COLOR_MAP,
            custom_data=["hover_text"],
            projection="natural earth"
        )
        fig.update_traces(
            hovertemplate="%{customdata[0]}<extra></extra>",
            marker_line_width=0.5,
            marker_line_color="#444444"
        )
        fig.update_geos(
            showcountries=True,
            countrycolor="#bbbbbb",
            showcoastlines=True,
            coastlinecolor="#cccccc",
            showland=True,
            landcolor="#f5f5f5",
            showocean=True,
            oceancolor="#eef3f8"
        )
        fig.update_layout(
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

        # Small states markers (< 10,000 sq km)
        if show_small_states:
            small_states = merged_filtered[merged_filtered["area_sqkm"] < 10000]
            for cat in CATEGORY_ORDER:
                cat_small = small_states[small_states["map_category"] == cat]
                if len(cat_small) > 0:
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
                            line=dict(width=1, color="#222222")
                        )
                    ))

        # Boundary markers
        if mark_boundary:
            boundary_countries = merged_filtered[merged_filtered["boundary_flag"] == True]
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

        st.plotly_chart(fig, use_container_width=True)

    # Category counts table
    st.subheader("Category Distribution")
    counts_series = countries_df["map_category"].value_counts()
    table_rows = []
    for cat in CATEGORY_ORDER:
        cnt = int(counts_series.get(cat, 0))
        pct = (cnt / len(countries_df)) * 100.0
        table_rows.append({
            "Category": cat,
            "Count": cnt,
            "Share": f"{pct:.1f}%"
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


# -------------------------------------------------------------
# TAB 2: CLUSTERS
# -------------------------------------------------------------
with tab_clusters:
    st.subheader("Constitutional Preamble Clusters")

    for c in range(7):
        c_row = clusters_df[clusters_df["cluster"] == c].iloc[0]
        label = c_row["label"]
        color = COLOR_MAP[label]
        n_c = int(c_row["n"])

        expander_title = f"Cluster {c}: {label} (n = {n_c})"
        with st.expander(expander_title):
            col_chip = f"<span style='display:inline-block; width:14px; height:14px; background-color:{color}; border-radius:3px; vertical-align:middle; margin-right:8px;'></span>"
            rule_badge = f"<span style='background-color:#e0e0e0; color:#333; padding:2px 8px; border-radius:4px; font-size:12px; font-weight:bold; margin-right:6px;'>Rule Rating: {c_row['rule_rating']}</span>"
            tier_badge = f"<span style='background-color:#d0e2ff; color:#0043ce; padding:2px 8px; border-radius:4px; font-size:12px; font-weight:bold;'>Cohesion Tier: {c_row['cohesion_tier']}</span>"

            st.markdown(f"### {col_chip}{label}", unsafe_allow_html=True)
            st.markdown(f"{rule_badge} {tier_badge}", unsafe_allow_html=True)

            st.markdown(f"**Description:** {c_row['description']}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Members (n)", n_c)
            c2.metric("Median Words", f"{c_row['median_words']:.0f}")
            c3.metric("Mean Silhouette", f"{c_row['mean_silhouette']:.4f}")
            c4.metric("Mean Subsample Conf", f"{c_row['mean_conf_sub']:.4f}")

            contrast_val = c_row["top_contrast_terms"]
            if pd.isna(contrast_val) or str(contrast_val).strip() == "":
                contrast_str = "No distinctive terms"
            else:
                contrast_str = str(contrast_val)

            st.markdown(f"**Top Contrast Terms:** {contrast_str}")
            st.markdown(f"**Top Characteristic Bigrams:** {c_row['top_bigrams']}")
            st.markdown(f"**Central Preambles (closest to centroid):** {c_row['central_countries']}")

            b_val = c_row["boundary_countries"]
            b_str = str(b_val) if pd.notna(b_val) and str(b_val).strip() != "" else "None"
            st.markdown(f"**Boundary Countries:** {b_str}")
            st.markdown(f"**Geographic Distribution:** {c_row['continents']}")

            # Members sorted by conf_sub
            members = countries_df[countries_df["final_cluster"] == c].sort_values(by="conf_sub", ascending=False)
            member_display = members[["Country", "conf_sub", "boundary_flag", "runnerup_label", "extracted_words"]].copy()
            member_display.columns = ["Country", "Confidence (conf_sub)", "Boundary Flag", "Runner-up Cluster", "Word Count"]
            st.markdown("**Member Preambles (sorted by confidence):**")
            st.dataframe(member_display, use_container_width=True, hide_index=True)

    strength_note = clusters_df["strength_note"].iloc[0]
    st.caption(f"**Methodology Note:** {strength_note}")


# -------------------------------------------------------------
# TAB 3: COUNTRY
# -------------------------------------------------------------
with tab_country:
    st.subheader("Country Preamble Profile")

    country_options = sorted(countries_df["Country"].unique())
    selected_country = st.selectbox(
        "Select a Country",
        options=country_options,
        index=None,
        placeholder="Type or select a country...",
        key="country_select"
    )

    if selected_country is not None:
        row = countries_df[countries_df["Country"] == selected_country].iloc[0]
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
    else:
        st.info("Please select a country from the dropdown above to view its details.")


# -------------------------------------------------------------
# TAB 4: TRY YOUR OWN TEXT
# -------------------------------------------------------------
with tab_predict:
    st.subheader("Classify a Preamble Text")

    acc_pct = f"{val_data['loo_accuracy'] * 100:.1f}"
    top2_pct = f"{val_data['loo_top2_accuracy'] * 100:.1f}"

    st.caption(f"Indicative only. On leave-one-out tests this method picks a country's own cluster {acc_pct}% of the time ({top2_pct}% within the top two).")
    st.caption("Short, generic or non-constitutional text will land in the catch-all cluster more often.")

    if "preamble_input" not in st.session_state:
        st.session_state["preamble_input"] = ""
    if "mask_name_input" not in st.session_state:
        st.session_state["mask_name_input"] = ""

    col_btn1, col_btn2 = st.columns(2)
    if col_btn1.button("Load US Constitution Preamble"):
        st.session_state["preamble_input"] = US_SAMPLE
        st.session_state["mask_name_input"] = "United States of America"
    if col_btn2.button("Load Islamic Republic Preamble Sample"):
        st.session_state["preamble_input"] = ISLAMIC_SAMPLE
        st.session_state["mask_name_input"] = ""

    user_text = st.text_area(
        "Enter preamble text to analyze:",
        key="preamble_input",
        height=160
    )
    mask_input = st.text_input(
        "Country name to mask (optional):",
        key="mask_name_input"
    )

    if st.button("Classify", type="primary", key="classify_btn"):
        word_count = len(user_text.strip().split())
        if word_count < 30:
            st.warning(f"Text contains only {word_count} words. Please enter at least 30 words for reliable classification.")
        else:
            try:
                model = get_embedding_model()
                import src.preamble_model as pm
                res = pm.classify_text(
                    model,
                    assets,
                    user_text,
                    mask_name=mask_input.strip() if mask_input.strip() else None
                )

                st.markdown("### Classification Results")

                top1_id, top1_label, top1_sim = res["top_clusters"][0]
                top2_id, top2_label, top2_sim = res["top_clusters"][1]

                c1, c2 = st.columns(2)
                c1.metric(f"Top 1: {top1_label}", f"{top1_sim:.4f}")
                c2.metric(f"Top 2: {top2_label}", f"{top2_sim:.4f}")

                if res["close_gap_flag"]:
                    st.warning(f"Close call: The similarity gap between top two clusters ({res['gap']:.4f}) is below 0.03.")

                st.markdown("#### 5 Nearest Preambles in Dataset")
                near_data = []
                for rank, (c_name, c_sim) in enumerate(res["top_countries"], start=1):
                    c_row = countries_df[countries_df["Country"] == c_name]
                    c_label = c_row["cluster_label"].iloc[0] if len(c_row) > 0 and pd.notna(c_row["cluster_label"].iloc[0]) else "Unclustered"
                    near_data.append({
                        "Rank": rank,
                        "Country": c_name,
                        "Cluster": c_label,
                        "Cosine Similarity": f"{c_sim:.4f}"
                    })
                st.dataframe(pd.DataFrame(near_data), use_container_width=True, hide_index=True)

            except Exception:
                st.error("This feature could not load the model on this machine")


# -------------------------------------------------------------
# TAB 5: METHOD AND LIMITS
# -------------------------------------------------------------
with tab_method:
    st.subheader("Methodology, Validation and Limitations")

    st.markdown("""
### Data
This dataset covers constitutional texts for 193 sovereign countries:
- **158 clustered preambles**: Texts containing sufficient substantive wording for comparative sentence embedding.
- **29 countries with no preamble in dataset**: The dataset record contains only the substantive articles of the constitution with no opening preamble.
- **6 formula only**: Preamble consists strictly of a brief enacting formula (too short to cluster reliably).

**Historical and Source Notes:**
- **United Kingdom**: Dataset text is Magna Carta, not a constitutional preamble.
- **Russia**: Dataset text refers to the Third All-Russian Congress of Soviets (an early Soviet-era text), not the current preamble.

**Data Sources:**
- Constitutional texts compiled by the Comparative Constitutions Project (Elkins, Ginsburg, and Melton).
- Cultural boundaries from Natural Earth (50m Admin 0 Countries).

---

### Method
1. **Sentence Embeddings**: Preambles are embedded using the `all-mpnet-base-v2` transformer model.
2. **Chunking**: Texts longer than 300 tokens are split into chunks of at most 300 tokens and averaged into a single document vector.
3. **Country Name Masking**: The country's own name and its capitalized components (at least 4 characters) are masked as `[COUNTRY]` to avoid trivial self-identification clustering.
4. **Mean-Centering**: Document vectors are centered by subtracting the column-wise training mean and re-normalized to unit L2 length ($C_c$).
5. **Clustering & Consensus**: Spherical K-Means ($k = 7$) executed over 200 bootstrapped runs to form a co-association matrix and consensus partition.
6. **Refinement**: 3 boundary countries were reassigned during consensus refinement to maximize silhouette and cluster cohesion.

---

### How to Read the Labels
Labels describe the wording and drafting tradition of preambles. They are not claims about a country's politics or values.

**Strength Ratings and Cohesion Tiers:**
Rule rating: strong if coverage >= 60% and precision >= 50%, weak if coverage < 40%, otherwise moderate. Loosely defined means the cluster is among the three with the lowest mean silhouette and lowest mean subsample confidence.

---

### Validation and Limitations

**Statistical Properties:**
- **Overall Silhouette**: Overall cosine silhouette score is 0.16 (indicating weak global geometric separation).
- **Consensus Reproducibility**: Single KMeans runs agree with each other at ARI 0.56 to 0.71 but the consensus is reproducible across split halves at ARI 0.96.
- **Continental Alignment**: Against continent the clusters have ARI 0.30 and leave-one-out accuracy 46% versus 19% for always guessing the largest cluster.
- **Opening Formulas**:
  - 12 of 18 "Whereas" preambles are in the Commonwealth cluster.
  - 10 of 16 "In the name of" preambles are in the Islamic and Arab cluster.
  - Only 22 of 62 "We the people" preambles are in the catch-all cluster (40 sit in other clusters).

**Limitations:**
- **Wording vs Governance**: Clusters reflect preamble wording, not governance.
- **Translations**: Many texts are English translations.
- **Regional Vocabulary**: Words such as "African" and "Arab" and demonyms are not masked, so region partly shows through.
- **Missing Dataset Texts**: 29 countries have no preamble in the dataset though the real constitution may have one.
- **Cohesion Variance**: Some clusters are loosely defined.
- **Inference Scope**: "Try your own text" is indicative.

**Disclaimer:**
The presence or absence of any country on the map is not an ideological or political statement.

**Project Repository:**
Code, notebooks, and dataset assets are available at [https://github.com/devpanchal2005/PreambleMap](https://github.com/devpanchal2005/PreambleMap).
""")
