import os
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
import matplotlib.pyplot as plt

DATA_DIR = "GoodReads"
TRAIN_CSV = os.path.join(DATA_DIR, "train_df.csv")
TEST_CSV = os.path.join(DATA_DIR, "test_df.csv")
BOOKS_SUBSET_CSV = os.path.join(DATA_DIR, "books_subset.csv")
OUT_PLOTS = "plots"
os.makedirs(OUT_PLOTS, exist_ok=True)

# Učitaj split i metapodatke
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
books = pd.read_csv(BOOKS_SUBSET_CSV)

print(f"Loaded: train {len(train_df)} rows, test {len(test_df)} rows, books {len(books)}")

# tipovi i sanity
train_df['user_id'] = train_df['user_id'].astype(int)
train_df['book_id'] = train_df['book_id'].astype(int)
test_df['user_id'] = test_df['user_id'].astype(int)
test_df['book_id'] = test_df['book_id'].astype(int)
if 'rating' in train_df.columns:
    train_df['rating'] = pd.to_numeric(train_df['rating'], errors='coerce')
if 'rating' in test_df.columns:
    test_df['rating'] = pd.to_numeric(test_df['rating'], errors='coerce')

# nakon učitavanja train_df/test_df/books
# osiguraj tipove (postoje u datoteci) i očisti duplicate u train/test ako je potrebno
train_df = train_df.drop_duplicates(subset=['user_id','book_id']).reset_index(drop=True)
test_df = test_df.drop_duplicates(subset=['user_id','book_id']).reset_index(drop=True)

# Surprise trening (kao prije)
reader = Reader(rating_scale=(train_df['rating'].min(), train_df['rating'].max()))
data = Dataset.load_from_df(train_df[['user_id', 'book_id', 'rating']], reader)
trainset = data.build_full_trainset()

svd = SVD(n_factors=50, random_state=42)
svd.fit(trainset)

# candidate pool: koristi popular + sve test items (union) da ne izgubiš relevantne stavke
test_items = set(test_df['book_id'].astype(int).unique())
valid_books = set(books['id'].astype(int).tolist())
pop_counts = train_df['book_id'].value_counts()
popular = pop_counts[pop_counts >= 5].index.tolist()
min_pool = 1000
if len(popular) < min_pool:
    popular = pop_counts.index[:min_pool].tolist()
candidate_book_ids = sorted(set(popular) & valid_books | test_items)
print(f"Candidate pool size: {len(candidate_book_ids)}; test items covered: {len(test_items & set(candidate_book_ids))}/{len(test_items)}")

# odabir korisnika za evaluaciju (slično kao u ContentFiltering)
def select_eval_users(train_df, test_df, candidate_book_ids, min_train_items=3, max_users=10000, seed=42):
    cand_set = set(candidate_book_ids)
    users = sorted(test_df['user_id'].unique())
    eligible = []
    for uid in users:
        train_items = set(train_df[train_df['user_id'] == uid]['book_id'].astype(int).tolist())
        test_items = set(test_df[test_df['user_id'] == uid]['book_id'].astype(int).tolist())
        # require min history and at least one test item covered in candidate pool
        if len(train_items) >= min_train_items and len(test_items & cand_set) > 0:
            eligible.append(uid)
    rng = np.random.RandomState(seed)
    if len(eligible) > max_users:
        eligible = list(rng.choice(eligible, size=max_users, replace=False))
    return sorted(eligible)

# <-- CHANGED: ensure users have >=1 train item and >=1 test item; limit to at most 200 users -->
min_train_items = 1
max_eval_users = 1000
eval_users = select_eval_users(train_df, test_df, candidate_book_ids,
                               min_train_items=min_train_items, max_users=max_eval_users, seed=42)
print(f"Evaluating on {len(eval_users)} users (min_train_items={min_train_items}, max={max_eval_users}).")

# Evaluacija na podskupu korisnika (MAE, RMSE)
test_subset_df = test_df[test_df['user_id'].isin(eval_users)].reset_index(drop=True)
test_tuples_subset = list(zip(test_subset_df['user_id'].astype(str),
                              test_subset_df['book_id'].astype(str),
                              test_subset_df['rating'].astype(float)))
preds_subset = svd.test(test_tuples_subset)
mae_subset = accuracy.mae(preds_subset, verbose=False)
rmse_subset = accuracy.rmse(preds_subset, verbose=False)

# Top-N preporuke i evaluacija (Precision, Recall, MAP)
N = 10
total_tp = total_pred = total_rel = 0
per_user_prec = []
per_user_rec = []
per_user_ap = []

cand_set = set(candidate_book_ids)
for uid in eval_users:
    relevant = set(test_df[test_df['user_id'] == uid]['book_id'].astype(int).tolist())
    seen = set(train_df[train_df['user_id'] == uid]['book_id'].astype(int).tolist())
    # restrict candidates to pool and exclude seen
    candidates = [b for b in candidate_book_ids if b not in seen]
    if not candidates:
        continue
    # score candidates (may be slow; consider parallelizing)
    scores = [(b, svd.predict(str(uid), str(b)).est) for b in candidates]
    scores.sort(key=lambda x: x[1], reverse=True)
    topn = [b for b,_ in scores[:N]]
    tp = len(set(topn) & relevant)
    per_user_prec.append(tp / N)
    per_user_rec.append(tp / len(relevant) if len(relevant)>0 else 0.0)
    # AP@N
    hits = 0; sum_prec = 0.0
    for i,b in enumerate(topn):
        if b in relevant:
            hits += 1
            sum_prec += hits / (i + 1)
    per_user_ap.append(sum_prec / min(len(relevant), N) if len(relevant)>0 else 0.0)
    total_tp += tp
    total_pred += len(topn)
    total_rel += len(relevant)

precision_macro = float(np.mean(per_user_prec)) if per_user_prec else np.nan
recall_macro = float(np.mean(per_user_rec)) if per_user_rec else np.nan
map_macro = float(np.mean(per_user_ap)) if per_user_ap else np.nan
precision_micro = total_tp / total_pred if total_pred>0 else np.nan
recall_micro = total_tp / total_rel if total_rel>0 else np.nan

print("\nSubset evaluation (selected users):")
print(f" MAE={mae_subset:.4f}, RMSE={rmse_subset:.4f}")
print(f" Precision@{N} macro={precision_macro:.6f}, micro={precision_micro:.6f}")
print(f" Recall@{N}    macro={recall_macro:.6f}, micro={recall_micro:.6f}")
print(f" MAP@{N} (macro)={map_macro:.6f}")

# --- Quick baselines and item-kNN experiment (evaluate on same eval_users) ---
from surprise import KNNBasic

def evaluate_ranklist_for_users(get_top_n_for_user, users_list):
    N = 10
    total_tp = total_pred = total_rel = 0
    per_prec = []; per_rec = []; per_ap = []
    for uid in users_list:
        relevant = set(test_df[test_df['user_id'] == uid]['book_id'].astype(int).tolist())
        if not relevant:
            continue
        seen = set(train_df[train_df['user_id'] == uid]['book_id'].astype(int).tolist())
        recs = get_top_n_for_user(uid, seen, N)
        if not recs:
            continue
        tp = len(set(recs) & relevant)
        per_prec.append(tp / N)
        per_rec.append(tp / len(relevant))
        # AP@N
        hits = 0; sum_prec = 0.0
        for i, b in enumerate(recs):
            if b in relevant:
                hits += 1
                sum_prec += hits / (i + 1)
        per_ap.append(sum_prec / min(len(relevant), N) if len(relevant) > 0 else 0.0)
        total_tp += tp
        total_pred += len(recs)
        total_rel += len(relevant)
    return {
        "precision_macro": float(np.mean(per_prec)) if per_prec else np.nan,
        "recall_macro": float(np.mean(per_rec)) if per_rec else np.nan,
        "map_macro": float(np.mean(per_ap)) if per_ap else np.nan,
        "precision_micro": total_tp / total_pred if total_pred > 0 else np.nan,
        "recall_micro": total_tp / total_rel if total_rel > 0 else np.nan,
        "users": len(per_prec)
    }