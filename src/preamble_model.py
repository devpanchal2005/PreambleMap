import os
import re
import numpy as np

LABELS = {
    0: "Generic Rights & Freedoms (catch-all)",
    1: "African Charter & Unity",
    2: "Statehood & History",
    3: "Constituent Assembly & God",
    4: "Commonwealth 'Whereas'",
    5: "Islamic & Arab Identity",
    6: "Struggle & Socialism (mostly Asia)"
}


def mask_country_name(country, text):
    """Mask country name and its capitalized components (>= 4 chars) with [COUNTRY]."""
    words = re.findall(r'[a-zA-Z]+', country)
    cap_words = [w for w in words if w and w[0].isupper() and len(w) >= 4]
    terms = set([country] + cap_words)
    sorted_terms = sorted(terms, key=len, reverse=True)
    pattern = r'\b(?:' + '|'.join(re.escape(t) for t in sorted_terms) + r')\b'
    masked_text, count = re.subn(pattern, '[COUNTRY]', text, flags=re.IGNORECASE)
    return masked_text, count


def chunk_variant_a(text, tokenizer, max_tokens=300):
    """Sentence-packing chunker (max 300 tokens) identical to notebook 04."""
    def count_tokens(s):
        return len(tokenizer.encode(str(s), add_special_tokens=False))

    text = ' '.join(str(text).split())
    tot = count_tokens(text)
    if tot <= max_tokens:
        return [text], [tot]

    raw_sents = [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if s.strip()]
    units = []
    for s in raw_sents:
        s_toks = tokenizer.encode(s, add_special_tokens=False)
        if len(s_toks) <= max_tokens:
            units.append(s)
        else:
            for j in range(0, len(s_toks), max_tokens):
                sub_ids = s_toks[j:j + max_tokens]
                sub_t = tokenizer.decode(sub_ids, skip_special_tokens=True).strip()
                if sub_t:
                    units.append(sub_t)

    chunks, curr = [], []
    for u in units:
        cand = ' '.join(curr + [u]) if curr else u
        if count_tokens(cand) <= max_tokens:
            curr.append(u)
        else:
            if curr:
                chunks.append(' '.join(curr))
            curr = [u]
    if curr:
        chunks.append(' '.join(curr))

    chunk_tokens = [count_tokens(c) for c in chunks]
    return chunks, chunk_tokens


def load_assets(path=None):
    """Load model assets from npz file."""
    if path is None:
        # Default relative to repository root or module directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base_dir, 'data', 'app', 'model_assets.npz')

    npz = np.load(path, allow_pickle=True)
    assets = {
        'country_names': npz['country_names'],
        'vectors': npz['vectors'],
        'column_mean': npz['column_mean'],
        'centroids': npz['centroids'],
        'cluster_ids': npz['cluster_ids']
    }
    return assets


def embed_text(model, text, mask_name=None, column_mean=None):
    """
    Embed text using all-mpnet-base-v2 with optional masking and 300-token chunking,
    then subtract column mean and L2-normalize.
    """
    if column_mean is None:
        assets = load_assets()
        column_mean = assets['column_mean']

    text = ' '.join(str(text).split())
    if mask_name:
        text, _ = mask_country_name(mask_name, text)

    tokenizer = model.tokenizer
    chunks, chunk_tokens = chunk_variant_a(text, tokenizer, max_tokens=300)

    embs = model.encode(chunks, batch_size=32, show_progress_bar=False, normalize_embeddings=False)
    weights = np.array(chunk_tokens, dtype=np.float32)

    weighted_mean = np.average(embs, axis=0, weights=weights)
    weighted_mean = weighted_mean / np.linalg.norm(weighted_mean)

    # Center using training column mean and row L2-normalize
    vec = weighted_mean - column_mean
    vec = vec / np.linalg.norm(vec)
    return vec.astype(np.float32)


def classify_text(model, assets, text, mask_name=None, top_n=5):
    """
    Classify input preamble text against cluster centroids and country vectors.

    Returns:
        dict with:
            'top_clusters': list of tuples (cluster_id, label, similarity) for top 2 clusters
            'top_countries': list of tuples (country_name, similarity) for top_n nearest countries
            'close_gap_flag': bool indicating whether the top 2 cluster gap < 0.03
            'gap': similarity difference between top 2 clusters
    """
    vec = embed_text(model, text, mask_name=mask_name, column_mean=assets['column_mean'])

    # Cosine similarities against 7 centroids
    sims_centroids = assets['centroids'] @ vec
    sorted_cluster_indices = np.argsort(sims_centroids)[::-1]

    c1 = int(sorted_cluster_indices[0])
    c2 = int(sorted_cluster_indices[1])
    s1 = float(sims_centroids[c1])
    s2 = float(sims_centroids[c2])
    gap = s1 - s2
    close_gap_flag = bool(gap < 0.03)

    top_clusters = [
        (c1, LABELS[c1], s1),
        (c2, LABELS[c2], s2)
    ]

    # Cosine similarities against 158 country vectors
    sims_countries = assets['vectors'] @ vec
    sorted_country_indices = np.argsort(sims_countries)[::-1][:top_n]

    top_countries = [
        (str(assets['country_names'][idx]), float(sims_countries[idx]))
        for idx in sorted_country_indices
    ]

    return {
        'top_clusters': top_clusters,
        'top_countries': top_countries,
        'close_gap_flag': close_gap_flag,
        'gap': gap
    }
