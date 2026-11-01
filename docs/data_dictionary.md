# Data dictionary

Files the app reads are in `data/app/`. `countries.csv`, `clusters.csv`, `terms.csv` and `model_assets.npz` are produced by `notebooks/08_app_data.ipynb` from the earlier notebooks' outputs. `validation.json` was written by a separate script during the app rebuild and is not produced by a notebook. All other files are intermediate results; the last section lists what each one is.

## data/app/countries.csv

One row per country. 193 rows, 25 columns.

| Column | Type | Meaning |
|---|---|---|
| Country | text | Country name as used in the dataset. A few names are mapped to Natural Earth names when drawing the map (for example Marshall Islands to "Marshall Is."). |
| category | text | `clustered` (158), `no_preamble_in_dataset` (29) or `formula_only` (6). |
| final_cluster | number | Final group number, 0 to 6. Empty for countries that are not clustered. |
| cluster_label | text | Name of the final group. Empty for countries that are not clustered. |
| conf_sub | number | Subsample confidence, 0 to 1: how consistently the country stays in its final group when the clustering is repeated on random subsamples. Shown in the app as "Fit". |
| boundary_flag | true or false | True for the 14 borderline countries (subsample confidence below 0.5, or a runner-up group scoring above the country's own). |
| runnerup_cluster | number | Number of the next closest group. |
| runnerup_label | text | Name of the next closest group. Shown in the app as "Next closest group". |
| runnerup_score | number | Subsample co-occurrence score with the next closest group's members (mean affinity in P_sub). |
| extracted_words | number | Word count of the extracted preamble text. |
| opening_type | text | How the preamble opens: `we_the_people`, `whereas`, `in_the_name_of` or `other`. Empty for countries that are not clustered. |
| continent | text | Continent from Natural Earth. Seychelles is listed as "Seven seas (open ocean)". |
| preamble_text | text | The extracted preamble. Empty for `no_preamble_in_dataset`. |
| note | text | A caveat shown in the app, for example that the dataset text has no preamble or is only an enacting formula. |
| source_note | text | A provenance warning. Only the United Kingdom (Magna Carta) and Russia (an early Soviet-era text) have one. |
| neighbor_1 to neighbor_5 | text | The five most similar preambles (country names), most similar first. Clustered countries only. |
| sim_1 to sim_5 | number | Cosine similarity to each neighbour, in the same order. |

The app adds one column when it loads the file: `map_category`, which combines the group label with the two not-clustered categories to give the nine map colours.

## data/app/clusters.csv

One row per group. 7 rows, 17 columns.

| Column | Meaning |
|---|---|
| cluster | Group number, 0 to 6 (numbered by size, largest first). |
| label | Group name. |
| n | Number of countries in the group. |
| description | One-sentence summary shown in the app. |
| rule_rating | `strong`, `moderate` or `weak`, from the rule fixed before testing: strong if coverage is at least 60% and precision at least 50%, weak if coverage is below 40%, otherwise moderate. |
| cohesion_tier | `well-defined` or `loosely defined`. The three groups with the lowest silhouette and the lowest subsample confidence are loosely defined. |
| coverage | Share of group members that match the label's keyword pattern. Partly circular, because the patterns came from the groups' own keywords. |
| precision | Share of all countries matching the pattern that are in this group. |
| mean_silhouette | Mean cosine silhouette of the group's members. |
| mean_conf_sub | Mean subsample confidence of the group's members. |
| top_contrast_terms | Words that appear in many more members than non-members. |
| top_bigrams | Most characteristic two-word phrases. |
| central_countries | Members closest to the group centre. |
| boundary_countries | Members flagged as borderline. |
| median_words | Median length of the group's preambles, in words. |
| continents | Count of members per continent. |
| strength_note | Method note shown on the Clusters tab. |

## data/app/terms.csv

Keyword results for each group. 240 rows, 8 columns.

| Column | Meaning |
|---|---|
| cluster | Group number. |
| kind | `unigram_ctfidf`, `bigram_ctfidf` or `contrast`. |
| rank | Rank of the term within its group and kind. |
| term | The word or phrase. |
| score | Score for the kind: class-based TF-IDF, or the contrast score. |
| share_in | Share of group members containing the term. |
| share_out | Share of non-members containing the term. |
| generic | Marks terms that are common in constitutional language in general. |

## data/app/model_assets.npz

NumPy archive used by the classifier. It covers the 158 clustered countries.

| Key | Shape | Meaning |
|---|---|---|
| country_names | 158 | Country names, in the same order as the vectors. |
| vectors | 158 by 768 | Masked, centred, normalised embeddings. |
| column_mean | 768 | The mean subtracted from each embedding. New text must be centred with this same mean. |
| centroids | 7 by 768 | Group centres. |
| cluster_ids | 158 | Final group number of each country. |

## data/app/validation.json

| Key | Meaning |
|---|---|
| loo_accuracy | Share of countries whose nearest group centre, with the country left out of its own centre, is its stored group (155 of 158, 0.981). This is a consistency check, not accuracy: the groups were found in the same space. |
| loo_top2_accuracy | Same, counting a hit if the stored group is one of the two nearest. |
| n | Number of countries tested (158). |
| per_cluster_accuracy | The consistency check for each group. |

## Other files in data/

| File | What it is |
|---|---|
| data/raw/constitution_dataset.csv | The original constitution text per country. |
| data/processed/constitution_dataset_cleaned.csv | Cleaned version from the first notebook. |
| data/processed/constitution_preamble_clustered.csv | Output of the original (first) clustering. Kept for comparison only. |
| data/processed/doc_type_audit.csv | The audit that classified each text as preamble, body text or other. |
| data/processed/preamble_extraction_proposal.csv, preamble_extraction_v2.csv, preamble_extraction_v3.csv | Successive drafts of the preamble extraction. |
| data/processed/preamble_final.csv | The final extraction (193 rows) with its status for each country. |
| data/processed/final_country_order.csv | Country order shared by the saved embeddings and results. |
| data/processed/cluster_runs.csv | KMeans labels for each of the six text versions, one column per version (labels_A and so on), used in the stability analysis. |
| data/processed/cluster_consensus.csv | Consensus groups before refinement. |
| data/processed/cluster_final.csv | Final groups after refinement, with confidence scores. |
| data/processed/cluster_keywords.csv | Keywords per group. |
| data/processed/label_validation.csv | Coverage and precision results behind the label ratings. |
| data/processed/emb_*.npy | Saved embeddings. Not tracked in git; regenerate them with notebook 04 (raw and cleaned) and notebook 05 (masked). |
| data/external/ne_50m_admin_0_countries/ | Natural Earth country shapes (public domain). |
