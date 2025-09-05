import os
import sys
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# --- Load shared train/test split produced by data_split.py ---
try:
    train_df = pd.read_csv('MovieLens/train_df.csv')
    test_df = pd.read_csv('MovieLens/test_df.csv')
    print("Loaded shared train_df/test_df")
except Exception:
    raise FileNotFoundError("MovieLens/train_df.csv and MovieLens/test_df.csv required. Run data_split.py first.")

# --- Load metadata and links and build movieId <-> metadata (safe columns) ---
metadata = pd.read_csv('MovieLens/movies_metadata.csv', low_memory=False)
links = pd.read_csv('MovieLens/links_small.csv')

# ensure numeric metadata ids and tmdbId present
metadata = metadata[metadata['id'].apply(lambda x: str(x).isdigit())].copy()
metadata['id'] = metadata['id'].astype(int)

links = links[links['tmdbId'].notnull()].copy()
links['tmdbId'] = links['tmdbId'].astype(int)

# safe-select metadata columns
cols_needed = ['id', 'title', 'overview', 'genres', 'tagline', 'original_language', 'keywords']
cols_present = [c for c in cols_needed if c in metadata.columns]
meta_part = metadata[cols_present].copy()
for c in cols_needed:
    if c not in meta_part.columns:
        meta_part[c] = ''

# merge to get movieId keyed rows
meta_links = pd.merge(links[['movieId', 'tmdbId']], meta_part, left_on='tmdbId', right_on='id', how='inner')
meta_links = meta_links.drop_duplicates(subset=['movieId']).reset_index(drop=True)

# build metadata_subset and textual content
metadata_subset = meta_links.copy()
for col in ['overview', 'tagline', 'genres', 'keywords', 'original_language']:
    if col not in metadata_subset.columns:
        metadata_subset[col] = ''
metadata_subset['genres_str'] = metadata_subset['genres'].fillna('').astype(str)
metadata_subset['keywords_str'] = metadata_subset['keywords'].fillna('').astype(str)
metadata_subset['content'] = (
    metadata_subset['overview'].fillna('') + ' ' +
    metadata_subset['genres_str'] + ' ' +
    metadata_subset['tagline'].fillna('') + ' ' +
    metadata_subset['original_language'].fillna('') + ' ' +
    metadata_subset['keywords_str']
)

# filter train/test to movies present in metadata_subset
valid_movieids = set(metadata_subset['movieId'].astype(int).tolist())
train_df = train_df[train_df['movieId'].isin(valid_movieids)].copy()
test_df = test_df[test_df['movieId'].isin(valid_movieids)].copy()

# build mappings
movieid_to_idx = dict(zip(metadata_subset['movieId'].astype(int), metadata_subset.index))
idx_to_movieid = dict(zip(metadata_subset.index, metadata_subset['movieId'].astype(int)))

print(f"Content: movies in metadata subset: {len(metadata_subset)}")
print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

# --- TF-IDF and cosine similarity ---
tfidf = TfidfVectorizer(stop_words='english', min_df=2, ngram_range=(1,1))
tfidf_matrix = tfidf.fit_transform(metadata_subset['content'].fillna(''))
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# candidate pool: popular movies in train to reduce noise
min_ratings = 5
pop_counts = train_df['movieId'].value_counts()
popular = pop_counts[pop_counts >= min_ratings].index.tolist()
min_pool = 1000
if len(popular) < min_pool:
    popular = pop_counts.index[:min_pool].tolist()
candidate_movie_ids = [mid for mid in popular if mid in valid_movieids]

def get_user_seen_train(uid):
    return set(train_df[train_df['userId'] == uid]['movieId'].unique())

def content_score_for_query(query_mid, candidates):
    q_idx = movieid_to_idx.get(int(query_mid))
    if q_idx is None:
        return {mid: 0.0 for mid in candidates}
    sims = np.asarray(cosine_sim[q_idx]).flatten()
    out = {}
    for mid in candidates:
        idx = movieid_to_idx.get(int(mid))
        out[mid] = float(sims[idx]) if idx is not None else 0.0
    return out

def rank_content(query_mid, candidates, top_n):
    scores = content_score_for_query(query_mid, candidates)
    ranked = sorted(candidates, key=lambda m: scores.get(m, 0.0), reverse=True)
    return ranked[:top_n]

def content_predict_rating(uid, test_mid):
    # weighted average of user's train items by content similarity to test item
    train_rows = train_df[train_df['userId'] == uid]
    if train_rows.empty:
        return float(train_df['rating'].mean()) if not train_df.empty else float(test_df['rating'].mean())
    weights = 0.0
    weighted_sum = 0.0
    for _, r in train_rows.iterrows():
        tr_mid = int(r['movieId'])
        tr_rating = float(r['rating'])
        if tr_mid in movieid_to_idx and test_mid in movieid_to_idx:
            sim = float(cosine_sim[movieid_to_idx[tr_mid], movieid_to_idx[test_mid]])
        else:
            sim = 0.0
        if sim > 0:
            weights += sim
            weighted_sum += sim * tr_rating
    if weights > 0:
        return weighted_sum / weights
    return float(train_rows['rating'].mean())

# --- Evaluation using shared train/test ---
def evaluate_content(N=10):
    users = sorted(test_df['userId'].unique())
    precisions = []
    aps = []
    recalls = []
    # collect predictions for MAE/RMSE in same order as test_df
    pred_content_all = []
    true_vals = list(test_df['rating'].astype(float))

    # ranking metrics: per-user top-N over candidate pool excluding seen
    users_evaluated = 0
    for uid in users:
        relevant = set(test_df[test_df['userId'] == uid]['movieId'].astype(int).tolist())
        if not relevant:
            continue
        train_rows = train_df[train_df['userId'] == uid]
        if train_rows.empty:
            continue
        # choose query movie (highest-rated in train)
        best_mid = int(train_rows.loc[train_rows['rating'].idxmax()]['movieId'])
        seen = get_user_seen_train(uid)
        candidates = [mid for mid in candidate_movie_ids if mid not in seen]
        if not candidates:
            continue
        top = rank_content(best_mid, candidates, N)
        tp = len(set(top) & relevant)
        precisions.append(tp / N)
        recalls.append(tp / len(relevant))
        # AP@N
        hits = 0; sum_prec = 0.0
        for i, mid in enumerate(top):
            if mid in relevant:
                hits += 1
                sum_prec += hits / (i + 1)
        ap = sum_prec / min(len(relevant), N) if relevant else 0.0
        aps.append(ap)
        users_evaluated += 1

    # MAE/RMSE: predict for all test rows in same order
    for _, trow in test_df.iterrows():
        uid = int(trow['userId'])
        test_mid = int(trow['movieId'])
        pred = content_predict_rating(uid, test_mid)
        pred_content_all.append(float(pred))

    out = {}
    out['users_evaluated'] = users_evaluated
    out['precision_at_10'] = float(np.mean(precisions)) if precisions else float('nan')
    out['recall_at_10'] = float(np.mean(recalls)) if recalls else float('nan')
    out['map10'] = float(np.mean(aps)) if aps else float('nan')
    out['mae'] = float(mean_absolute_error(true_vals, pred_content_all)) if pred_content_all else float('nan')
    out['rmse'] = float(np.sqrt(mean_squared_error(true_vals, pred_content_all))) if pred_content_all else float('nan')
    return out

# run and print
metrics = evaluate_content(N=10)
print("\nContent-only evaluation (shared split):")
print(f"Evaluated users (approx): {metrics['users_evaluated']}")
print(f"Precision@10: {metrics['precision_at_10']:.4f}")
print(f"Recall@10: {metrics['recall_at_10']:.4f}")
print(f"MAP@10: {metrics['map10']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")

# optional: plot distribution of test-item counts per user (diagnostic)
os.makedirs("plots", exist_ok=True)
counts = test_df.groupby('userId').size()
plt.figure(figsize=(8,4))
bins = range(1, int(counts.max())+2)
plt.hist(counts, bins=bins, edgecolor='black')
plt.xlabel('Broj test-itema po korisniku'); plt.ylabel('Broj korisnika')
plt.title('Distribucija broja test-itema po korisniku (test_df)')
plt.tight_layout()
plt.savefig("plots/content_test_items_histogram.png", dpi=150)
plt.show()