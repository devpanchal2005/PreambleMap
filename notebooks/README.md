# Notebooks

Run them in order. The saved outputs already contain the results, so you only need to re-run them to reproduce the analysis.

```bash
pip install -r requirements-dev.txt
```

Open each notebook with the project's virtual environment as the kernel, from the repository root or the `notebooks/` folder.

| Notebook | Purpose | Main outputs |
|---|---|---|
| `01_data_preprocessing.ipynb` | The original first-pass analysis: cleaning, embedding the "Preamble" column and clustering it. It has no markdown cells and is kept unchanged for history. The data audit later showed that its text column mixed document types, so its clusters are not used in the final results. | `data/processed/constitution_dataset_cleaned.csv`, `constitution_preamble_clustered.csv` |
| `02_embedding_comparison.ipynb` | Diagnostics on the original data: text lengths, truncation by the embedding model, document types. This is where the data problems were found. | `data/processed/doc_type_audit.csv` |
| `03_preamble_extraction.ipynb` | Extracts the real preamble from each text. Goes from a proposal, through two revisions, to the validated final dataset (clustered, no preamble in dataset, formula only). | `preamble_extraction_proposal.csv`, `preamble_extraction_v2.csv`, `preamble_extraction_v3.csv`, `preamble_final.csv` |
| `04_final_embeddings.ipynb` | Embeds the 158 clusterable preambles with `all-mpnet-base-v2` in chunks, with country-name masking and mean-centring. | `emb_final_raw.npy` and `emb_final_clean.npy` (not tracked in git), `final_country_order.csv` |
| `05_clustering_analysis.ipynb` | Quality metrics, a Gaussian null baseline, and seed and subsample stability for six text representations and several values of k. Supports the choice of representation and k = 7. | `emb_final_masked.npy` (not tracked in git), `cluster_runs.csv`, metrics and stability results |
| `06_consensus_and_baselines.ipynb` | Consensus clustering over 200 KMeans runs. Compares other algorithms (Gaussian mixtures, Ward, average linkage, HDBSCAN) and other embeddings (TF-IDF, MiniLM). Checks the effect of text length and chunking. | `cluster_consensus.csv` |
| `07_cluster_labels.ipynb` | Refines the consensus (3 countries move), flags boundary cases, extracts keywords and compares the groups with continent and opening formula. | `cluster_final.csv`, `cluster_keywords.csv` |
| `07b_label_validation.ipynb` | Tests each label's keyword pattern (coverage and precision), applies the fixed rating rule, and compares with a region baseline. | `label_validation.csv` |
| `08_app_data.ipynb` | Builds the data package the app reads and the prediction functions' assets. Also measures the app's memory use. | `countries.csv`, `clusters.csv`, `terms.csv` and `model_assets.npz` in `data/app/` (see `docs/data_dictionary.md`) |

## Notes

- Embeddings are not stored in git (`data/processed/*.npy` is ignored). Notebook 04 recreates the raw and cleaned embeddings and notebook 05 the masked ones.
- Notebook 04 and later download `all-mpnet-base-v2` (about 420 MB) the first time they run.
- The final results depend on notebooks 03 to 08. Notebooks 01 and 02 explain why the project was rebuilt.
