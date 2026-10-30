"""
PreambleMap: Clustering Constitutional Preambles by Their Wording
Production Streamlit Application
"""

import json
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st

from src.map_view import (
    CATEGORY_ORDER,
    COLOR_MAP,
    ALIASES,
    REVERSE_ALIASES,
    build_map,
    render_country_card
)

# Page configuration with wide layout and collapsed sidebar
st.set_page_config(
    page_title="PreambleMap",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

    # Drop Antarctica polygon
    world = world[world["NAME"] != "Antarctica"].copy()

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


# Load application data
countries_df, terms_df, clusters_df, val_data, assets = load_app_data()
world_gdf = load_world_geometry()

# Mapping between Natural Earth shapefile NAME and Country column
ne_to_country = {}
for c in countries_df["Country"]:
    ne_name = ALIASES.get(c, c)
    ne_to_country[ne_name] = c

# Application Header
st.title("PreambleMap: Clustering Constitutional Preambles by Their Wording")
st.markdown("Labels describe the wording and drafting tradition of preambles. They are not claims about a country's politics or values.")

# Navigation tabs (Country tab removed, country card embedded in Map tab)
tab_map, tab_clusters, tab_predict, tab_method = st.tabs([
    "Map",
    "Clusters",
    "Try your own text",
    "Method and limits"
])

# -------------------------------------------------------------
# TAB 1: MAP
# -------------------------------------------------------------
with tab_map:
    # Synchronize click-to-select from map before rendering selectbox
    if "map" in st.session_state and st.session_state["map"]:
        sel_points = st.session_state["map"].get("selection", {}).get("points", [])
        if sel_points:
            clicked_loc = sel_points[0].get("location")
            if clicked_loc in ne_to_country:
                matched_c = ne_to_country[clicked_loc]
                if st.session_state.get("find_country") != matched_c:
                    st.session_state["find_country"] = matched_c

    # Row of three controls above the map
    sorted_country_options = sorted(countries_df["Country"].unique())
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2.5, 2.5, 1.2])

    with ctrl_col1:
        find_country = st.selectbox(
            "Find a country",
            options=sorted_country_options,
            index=None,
            placeholder="Type a country name",
            key="find_country"
        )

    with ctrl_col2:
        focus_options = ["All categories"] + CATEGORY_ORDER
        focus_category = st.selectbox(
            "Focus on",
            options=focus_options,
            index=0,
            key="focus_category"
        )

    with ctrl_col3:
        st.write("")
        st.write("")
        with st.popover("Display options"):
            mark_boundary = st.checkbox(
                "Mark boundary countries",
                value=False,
                key="mark_boundary"
            )
            show_small_states = st.checkbox(
                "Show markers for small states",
                value=True,
                key="show_small_states"
            )

    # Build and render the map figure
    fig = build_map(
        world_gdf=world_gdf,
        countries_df=countries_df,
        focus_category=focus_category,
        selected_country=find_country,
        mark_boundary=mark_boundary,
        show_small_states=show_small_states
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        theme="streamlit",
        on_select="rerun",
        selection_mode="points",
        key="map"
    )

    # Caption under the map
    st.caption("Blank areas are not in the dataset (for example Greenland). Click a legend entry to hide or show a category.")

    # Show country card if a country is selected
    if find_country is not None:
        row = countries_df[countries_df["Country"] == find_country].iloc[0]
        render_country_card(row, countries_df)

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
# TAB 3: TRY YOUR OWN TEXT
# -------------------------------------------------------------
with tab_predict:
    st.subheader("Classify a Preamble Text")

    st.caption(
        "This shows which cluster centre your text is closest to and which preambles it resembles. "
        "It describes wording and drafting tradition, not politics. "
        "Clusters overlap (overall silhouette 0.16), so treat the result as a guide, not a verdict. "
        "Short, generic or non-constitutional text will land in the catch-all cluster more often."
    )

    if "preamble_input" not in st.session_state:
        st.session_state["preamble_input"] = ""
    if "mask_name_input" not in st.session_state:
        st.session_state["mask_name_input"] = ""

    col_btn1, col_btn2 = st.columns(2)
    if col_btn1.button("Load US Constitution Preamble", key="btn_us"):
        st.session_state["preamble_input"] = US_SAMPLE
        st.session_state["mask_name_input"] = "United States of America"
    if col_btn2.button("Load Islamic Republic Preamble Sample", key="btn_islamic"):
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
        words = user_text.strip().split()
        word_count = len(words)
        if word_count == 0:
            st.warning("Please enter some text to classify.")
        else:
            if word_count < 30:
                st.warning(f"Short text ({word_count} words). Minimum 30 words recommended for reliable classification.")
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
# TAB 4: METHOD AND LIMITS
# -------------------------------------------------------------
with tab_method:
    st.subheader("Methodology, Validation and Limitations")

    n_loo_correct = int(round(val_data["loo_accuracy"] * val_data["n"]))
    consistency_note = (
        f"Consistency check: when each country is left out of its own cluster centre, "
        f"nearest-centre assignment reproduces the stored label for {n_loo_correct} of 158 countries. "
        f"This is expected, because the clusters were found in the same space; "
        f"it does not show that the groups are natural categories."
    )

    st.markdown(f"""
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
- **Consistency Check**: {consistency_note}
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
