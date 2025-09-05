# Refaktorirana skripta za sadržajno filtriranje (GoodReads) koristeći globalni train/test split
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = "GoodReads"
TRAIN_CSV = os.path.join(DATA_DIR, "train_df.csv")
TEST_CSV = os.path.join(DATA_DIR, "test_df.csv")
BOOKS_SUBSET_CSV = os.path.join(DATA_DIR, "books_subset.csv")
MAPPING_CSV = os.path.join(DATA_DIR, "bookid_to_title.csv")

# Učitaj train/test (split iz data_split_goodreads.py) i metapodatke
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
books = pd.read_csv(BOOKS_SUBSET_CSV)

print(f"Loaded: train {len(train_df)} rows, test {len(test_df)} rows, books {len(books)}")

# Osiguraj tipove
train_df['book_id'] = train_df['book_id'].astype(int)
test_df['book_id'] = test_df['book_id'].astype(int)
train_df['user_id'] = train_df['user_id'].astype(int)
test_df['user_id'] = test_df['user_id'].astype(int)
if 'rating' in train_df.columns:
    train_df['rating'] = pd.to_numeric(train_df['rating'], errors='coerce')
if 'rating' in test_df.columns:
    test_df['rating'] = pd.to_numeric(test_df['rating'], errors='coerce')

# Pripremi sadrzaj: title + authors + description (ako postoji) + tag_name (ako postoji)
books['title'] = books['title'].fillna('').astype(str)
books['authors'] = books.get('authors', '').fillna('').astype(str).str.lower()
books['tag_name'] = books.get('tag_name', '').fillna('').astype(str)
if 'description' in books.columns:
    books['description'] = books['description'].fillna('').astype(str)
    books['content'] = books['title'] + ' ' + books['authors'] + ' ' + books['description'] + ' ' + books['tag_name']
else:
    books['content'] = books['title'] + ' ' + books['authors'] + ' ' + books['tag_name']

# map book_id -> index in books dataframe
books['id'] = books['id'].astype(int)
bookid_to_idx = dict(zip(books['id'], books.index))
idx_to_bookid = dict(zip(books.index, books['id']))

# TF-IDF + cosine similarity (use linear_kernel for speed)
tfidf = TfidfVectorizer(stop_words='english', min_df=2, ngram_range=(1,1))
tfidf_matrix = tfidf.fit_transform(books['content'].fillna(''))
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)  # dense enough for moderate sizes

# Candidate pool: popular books in train to reduce noise (min_ratings fallback)
min_ratings = 5
pop_counts = train_df['book_id'].value_counts()
popular = pop_counts[pop_counts >= min_ratings].index.tolist()
min_pool = 1000
if len(popular) < min_pool:
    popular = pop_counts.index[:min_pool].tolist()
valid_books = set(books['id'].tolist())
candidate_book_ids = [int(b) for b in popular if int(b) in valid_books]

def get_user_seen_train(uid):
    return set(train_df[train_df['user_id'] == uid]['book_id'].unique())

def content_score_for_query(query_bid, candidates):
    q_idx = bookid_to_idx.get(int(query_bid))
    if q_idx is None:
        return {mid: 0.0 for mid in candidates}
    sims = np.asarray(cosine_sim[q_idx]).flatten()
    out = {}
    for mid in candidates:
        idx = bookid_to_idx.get(int(mid))
        out[mid] = float(sims[idx]) if idx is not None else 0.0
    return out

def rank_content(query_bid, candidates, top_n):
    scores = content_score_for_query(query_bid, candidates)
    ranked = sorted(candidates, key=lambda m: scores.get(m, 0.0), reverse=True)
    return ranked[:top_n]

def content_predict_rating(uid, test_bid):
    # weighted average of user's train items by content similarity to test item
    train_rows = train_df[train_df['user_id'] == uid]
    if train_rows.empty:
        # fallback global mean from train
        return float(train_df['rating'].mean()) if not train_df.empty else 3.0
    weights = 0.0
    weighted_sum = 0.0
    test_idx = bookid_to_idx.get(int(test_bid))
    for _, r in train_rows.iterrows():
        tr_mid = int(r['book_id'])
        tr_rating = float(r['rating'])
        tr_idx = bookid_to_idx.get(tr_mid)
        if tr_idx is None or test_idx is None:
            sim = 0.0
        else:
            sim = float(cosine_sim[tr_idx, test_idx])
        if sim > 0:
            weights += sim
            weighted_sum += sim * tr_rating
    if weights > 0:
        return weighted_sum / weights
    # fallback to user's train mean then global train mean
    if not train_rows['rating'].dropna().empty:
        return float(train_rows['rating'].mean())
    return float(train_df['rating'].mean()) if not train_df.empty else 3.0

# --- Evaluation (ranking + rating) ---
N = 10
# select eval users: only users with >=1 train and >=1 test item, sample up to 200 for speed
all_test_users = sorted(test_df['user_id'].unique())
eligible = []
for uid in all_test_users:
    train_items = set(train_df[train_df['user_id'] == uid]['book_id'].astype(int).tolist())
    test_items = set(test_df[test_df['user_id'] == uid]['book_id'].astype(int).tolist())
    if len(train_items) > 0 and len(test_items) > 0:
        eligible.append(uid)

MAX_EVAL_USERS = 1000
rng = np.random.RandomState(42)
if len(eligible) > MAX_EVAL_USERS:
    eval_users = list(rng.choice(eligible, size=MAX_EVAL_USERS, replace=False))
else:
    eval_users = eligible
users = sorted(eval_users)
print(f"Evaluating on {len(users)} users (have >=1 train and >=1 test, max={MAX_EVAL_USERS}).")

precisions = []
recalls = []
aps = []
preds = []
trues = []
users_evaluated = 0

# ranking metrics: per-user top-N over candidate pool excluding seen train items
for uid in users:
    relevant = set(test_df[test_df['user_id'] == uid]['book_id'].astype(int).tolist())
    if not relevant:
        continue
    train_rows = train_df[train_df['user_id'] == uid]
    if train_rows.empty:
        continue
    # choose query item (use highest-rated in train as query, like MovieLens example)
    best_row = train_rows.loc[train_rows['rating'].idxmax()]
    best_bid = int(best_row['book_id'])
    seen = get_user_seen_train(uid)
    candidates = [mid for mid in candidate_book_ids if mid not in seen]
    if not candidates:
        continue
    top = rank_content(best_bid, candidates, N)
    tp = len(set(top) & relevant)
    precisions.append(tp / N)
    recalls.append(tp / len(relevant))
    # AP@N
    hits = 0
    sum_prec = 0.0
    for i, mid in enumerate(top):
        if mid in relevant:
            hits += 1
            sum_prec += hits / (i + 1)
    ap = sum_prec / min(len(relevant), N) if relevant else 0.0
    aps.append(ap)
    users_evaluated += 1

# MAE/RMSE: predict for all test rows
for _, trow in test_df.iterrows():
    uid = int(trow['user_id'])
    bid = int(trow['book_id'])
    true_rating = float(trow['rating'])
    pred_rating = content_predict_rating(uid, bid)
    preds.append(pred_rating)
    trues.append(true_rating)

# compute metrics
precision_at_10 = float(np.mean(precisions)) if precisions else float('nan')
recall_at_10 = float(np.mean(recalls)) if recalls else float('nan')
map10 = float(np.mean(aps)) if aps else float('nan')
mae = float(mean_absolute_error(trues, preds)) if preds else float('nan')
rmse = float(np.sqrt(mean_squared_error(trues, preds))) if preds else float('nan')

# print summary
print("\nContent-only evaluation (GoodReads shared split):")
print(f"Evaluated users (ranking): {users_evaluated}")
print(f"Precision@10: {precision_at_10:.4f}")
print(f"Recall@10:    {recall_at_10:.4f}")
print(f"MAP@10:       {map10:.4f}")
print(f"MAE:          {mae:.4f}")
print(f"RMSE:         {rmse:.4f}")

# optional diagnostic plot: metric bar
out_dir = "plots"
os.makedirs(out_dir, exist_ok=True)
plt.figure(figsize=(8,4))
labels = ['Precision@10', 'Recall@10', 'MAP@10']
values = [precision_at_10, recall_at_10, map10]
plt.bar(labels, values, color=['C0','C1','C2'])
plt.ylim(0,1)
plt.title('Ranking metrics (Content-only)')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "content_ranking_metrics.png"), dpi=150)
plt.close()

# --- Dodatne vizualizacije: distribucije broja ocjena po korisniku i broj test-itema po korisniku ---
out_dir = "plots"
os.makedirs(out_dir, exist_ok=True)

# broj ocjena u train i test po korisniku
train_counts = train_df.groupby('user_id').size()
test_counts = test_df.groupby('user_id').size()

# Histogram: distribucija broja ocjena u train skupu (log-scaled y zbog duga repa)
plt.figure(figsize=(8,4))
plt.hist(train_counts.values, bins=50, color='C0', edgecolor='k', alpha=0.7)
plt.yscale('log')
plt.xlabel('Broj ocjena u trainu po korisniku')
plt.ylabel('Broj korisnika (log skalirano)')
plt.title('Distribucija broja ocjena po korisniku (train)')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "train_ratings_per_user_hist.png"), dpi=150)
plt.close()

# Histogram: distribucija broja test-itema po korisniku
plt.figure(figsize=(8,4))
plt.hist(test_counts.values, bins=30, color='C1', edgecolor='k', alpha=0.7)
plt.xlabel('Broj test-itema po korisniku')
plt.ylabel('Broj korisnika')
plt.title('Distribucija broja test-itema po korisniku')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "test_items_per_user_hist.png"), dpi=150)
plt.close()

# Bar: frekvencija broja test-itema (useful when counts are small integers)
test_counts_freq = test_counts.value_counts().sort_index()
plt.figure(figsize=(8,4))
plt.bar(test_counts_freq.index.astype(str), test_counts_freq.values, color='C2', edgecolor='k', alpha=0.8)
plt.xlabel('Broj test-itema')
plt.ylabel('Broj korisnika')
plt.title('Frekvencija broja test-itema po korisniku')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "test_items_per_user_bar.png"), dpi=150)
plt.close()

# Scatter: odnos između broja ocjena u trainu i broja test-itema (za korisnike prisutne u oba skupa)
common_users = sorted(list(set(train_counts.index) & set(test_counts.index)))
if common_users:
    x = train_counts.reindex(common_users).values
    y = test_counts.reindex(common_users).values
    plt.figure(figsize=(6,6))
    plt.scatter(x, y, alpha=0.4, s=10, color='C3')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Broj ocjena u trainu (log)')
    plt.ylabel('Broj test-itema (log)')
    plt.title('Train vs Test: broj ocjena po korisniku (log-log)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_vs_test_scatter.png"), dpi=150)
    plt.close()