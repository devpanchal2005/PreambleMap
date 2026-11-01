# Viva notes: PreambleMap

One line to open with: "I cluster the wording of 158 constitutional preambles into 7 soft groups. The groups track drafting tradition and region more than ideology, and I say so in the app."

## The eight points

1. **The first clustering mixed document types.** The original "Preamble" column held full constitutions for many countries (33 of 193 texts over 5,000 words, median 255). All 27 body-text countries sat in three of the seven old clusters. A data audit found it and I rebuilt on extracted preambles: 158 clustered, 29 with no preamble in the dataset, 6 formula only.
2. **Why mask the country name and centre the embeddings.** Masking stops a country being grouped because it names itself. Centring removes the offset shared by all formal legal prose. Centring raised seed stability by 0.10 to 0.12 ARI and subsample stability by 0.06 to 0.18 at k = 7. Masked and unmasked versions agree only at ARI 0.40 to 0.52, so the choice matters.
3. **Why a consensus of 200 runs.** One KMeans run is unstable: agreement between seeds is ARI 0.56 to 0.71, and with 50 restarts no other seed reproduced seed 0's partition. The consensus has split-half ARI 0.955.
4. **The structure is weak, and I say so.** Silhouette is at most 0.16, but it beats a Gaussian null for every k from 3 to 12. HDBSCAN finds only 2 or 3 clusters with 31 to 74 unassigned countries. So these are soft groups, not natural categories.
5. **The labels were validated, with a rule fixed beforehand.** Coverage and precision of keyword patterns: one label strong (Islamic & Arab Identity, coverage 94%, precision 89%), six moderate, none weak. Coverage was partly circular, so precision is the number to trust. Silhouette and subsample confidence agree that three groups (catch-all, Struggle & Socialism, Commonwealth 'Whereas') are loosely defined.
6. **Groups follow drafting tradition and region more than ideology.** 12 of 18 "Whereas" openers are in one group, 10 of 16 "In the name of" openers in another, but only 22 of 62 "We the people" openers share a group. Cramer's V between group and opening formula is 0.50. Continent explains less than half: ARI 0.30, and 45.8% leave-one-out accuracy against 19.4% for always guessing the largest group.
7. **What the 98.1% nearest-centre figure does and does not show.** 155 of 158 countries land in their stored group when assigned to the nearest centre. That is expected, because the groups were found in the same space. It is a consistency check, not accuracy, and the app labels it that way.
8. **Limitations.** English translations, region words and demonyms not masked, 35 countries not clustered, the United Kingdom text is Magna Carta and the Russia text is an early Soviet-era text, and the classifier is indicative only.

## Questions to expect

- **Why k = 7?** The silhouette peaks there for the chosen representation and it is the most subsample-stable of 5, 7 and 9. A Gaussian mixture's BIC is lowest at 8 and flat from 7 to 10. So 7 is a defensible resolution, not a discovery.
- **Why this embedding model?** I compared two others. TF-IDF with SVD had silhouette 0.10 and seed stability 0.36. all-MiniLM-L6-v2 had 0.12 and 0.55. `all-mpnet-base-v2` was the best behaved.
- **Are you just clustering boilerplate?** Partly, and I have not tested it. The opening-formula results suggest formulas drive some groups. A formula-stripped re-run is listed as future work. One partial check: a first-chunk-only variant keeps 14 of 16 members of the longest-text group together, so that group is not only a chunk-averaging effect.
- **What does "strong" mean?** Only that the rule was met: coverage at least 60% and precision at least 50% for the label's keyword pattern. It does not mean the group is correct or natural.
- **Why is the United Kingdom's text Magna Carta?** That is what the dataset holds. The app shows a source note instead of hiding it.
- **Could the colonial-heritage story be true?** The African countries in the catch-all group look mostly English-speaking, but there is no legal-origin variable in the data, so this is an untested hypothesis.

## Do not say

- That the groups are ideologies, or that a colour says anything about a country's politics.
- That 98.1% is accuracy.
- That k = 7 is the true number of groups.
- That grey or blank on the map means a country has no constitution or no preamble. It means the dataset text has none or the country is not in the dataset.
