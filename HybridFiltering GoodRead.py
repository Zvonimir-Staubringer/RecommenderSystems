import os
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, SVD
from surprise import accuracy
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

DATA_DIR = "GoodReads"
TRAIN_CSV = os.path.join(DATA_DIR, "train_df.csv")
TEST_CSV = os.path.join(DATA_DIR, "test_df.csv")
BOOKS_SUBSET_CSV = os.path.join(DATA_DIR, "books_subset.csv")
MAPPING_CSV = os.path.join(DATA_DIR, "bookid_to_title.csv")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
books = pd.read_csv(BOOKS_SUBSET_CSV)

print(f"Loaded: train {len(train_df)} rows, test {len(test_df)} rows, books {len(books)}")

for df in (train_df, test_df):
    df['user_id'] = df['user_id'].astype(int)
    df['book_id'] = df['book_id'].astype(int)
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

books['title'] = books['title'].fillna('').astype(str)
books['authors'] = books.get('authors', '').fillna('').astype(str).str.lower()
books['tag_name'] = books.get('tag_name', '').fillna('').astype(str)
if 'description' in books.columns:
    books['description'] = books['description'].fillna('').astype(str)
    books['content'] = (books['title'] + ' ' + books['authors'] + ' ' +
                        books['description'] + ' ' + books['tag_name'])
else:
    books['content'] = books['title'] + ' ' + books['authors'] + ' ' + books['tag_name']

books['id'] = books['id'].astype(int)
bookid_to_idx = dict(zip(books['id'], books.index))
idx_to_bookid = dict(zip(books.index, books['id']))
valid_books = set(books['id'].tolist())

tfidf = TfidfVectorizer(stop_words='english', min_df=2, ngram_range=(1, 1))
tfidf_matrix = tfidf.fit_transform(books['content'].fillna(''))
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix) # dense enough for moderate sizes

min_ratings = 5
min_pool = 1000
pop_counts = train_df['book_id'].value_counts()
popular = pop_counts[pop_counts >= min_ratings].index.tolist()
if len(popular) < min_pool:
    popular = pop_counts.index[:min_pool].tolist()
candidate_book_ids = [int(b) for b in popular if int(b) in valid_books]
candidate_set = set(candidate_book_ids)
print(f"Candidate pool size: {len(candidate_book_ids)}")

def get_user_seen_train(uid):
    return set(train_df[train_df['user_id'] == uid]['book_id'].unique())

def content_scores_for_book(query_bid, candidates):
    """Return dict mapping candidate book_id -> cosine similarity to query book_id."""
    q_idx = bookid_to_idx.get(int(query_bid))
    out = {}
    if q_idx is None:
        for c in candidates:
            out[c] = 0.0
        return out
    sims = np.asarray(cosine_sim[q_idx]).flatten()
    for b in candidates:
        idx = bookid_to_idx.get(int(b))
        out[b] = float(sims[idx]) if idx is not None else 0.0
    return out

def rank_by_content(query_bid, candidates, top_n=10):
    sc = content_scores_for_book(query_bid, candidates)
    ranked = sorted(candidates, key=lambda b: sc.get(b, 0.0), reverse=True)
    return ranked[:top_n]

def content_predict_rating(uid, test_bid):
    """Predict rating using content: weighted average of user's train ratings by similarity to test item."""
    train_rows = train_df[train_df['user_id'] == uid]
    if train_rows.empty:
        return float(train_df['rating'].mean()) if not train_df.empty else 3.0
    weights = 0.0
    weighted_sum = 0.0
    test_idx = bookid_to_idx.get(int(test_bid))
    for _, r in train_rows.iterrows():
        tr_bid = int(r['book_id'])
        tr_rating = float(r['rating'])
        tr_idx = bookid_to_idx.get(tr_bid)
        if tr_idx is None or test_idx is None:
            sim = 0.0
        else:
            sim = float(cosine_sim[tr_idx, test_idx])
        if sim > 0:
            weights += sim
            weighted_sum += sim * tr_rating
    if weights > 0:
        return weighted_sum / weights
    if not train_rows['rating'].dropna().empty:
        return float(train_rows['rating'].mean())
    return float(train_df['rating'].mean()) if not train_df.empty else 3.0

reader = Reader(rating_scale=(train_df['rating'].min(), train_df['rating'].max()))
data = Dataset.load_from_df(train_df[['user_id', 'book_id', 'rating']].astype(str), reader)
trainset = data.build_full_trainset()
svd = SVD(n_factors=50, random_state=42)
svd.fit(trainset)

def collab_predict_rating(uid, bid):
    return svd.predict(str(uid), str(bid)).est

def hybrid_rank(query_bid, uid, candidates, alpha=0.5, top_n=10, min_est=1.0, max_est=5.0):
    content_sc = content_scores_for_book(query_bid, candidates)
    collab_sc = {}
    for b in candidates:
        try:
            est = svd.predict(str(uid), str(b)).est
        except Exception:
            est = (min_est + max_est) / 2.0
        # normalize to [0,1]
        collab_sc[b] = (est - min_est) / (max_est - min_est) if (max_est > min_est) else 0.0
    hybrid_scores = {}
    for b in candidates:
        c = content_sc.get(b, 0.0)
        k = collab_sc.get(b, 0.0)
        hybrid_scores[b] = alpha * c + (1.0 - alpha) * k
    ranked = sorted(candidates, key=lambda x: hybrid_scores.get(x, 0.0), reverse=True)
    return ranked[:top_n]

def hybrid_predict_rating(uid, bid, alpha_rating=0.5):
    c_pred = content_predict_rating(uid, bid)
    s_pred = collab_predict_rating(uid, bid)
    return alpha_rating * c_pred + (1 - alpha_rating) * s_pred

N = 10

# --- SELECT EVAL USERS: only users with >=1 train and >=1 test item, sample up to 200 ---
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
eval_users = sorted(eval_users)
print(f"Evaluating on {len(eval_users)} users (have >=1 train and >=1 test).")

def evaluate_ranking_for_method(get_topn_for_user, users_list):
    precisions = []
    recalls = []
    aps = []
    users_evaluated = 0
    for uid in users_list:
        relevant = set(test_df[test_df['user_id'] == uid]['book_id'].astype(int).tolist())
        if not relevant:
            continue
        train_rows = train_df[train_df['user_id'] == uid]
        if train_rows.empty:
            continue
        best_row = train_rows.loc[train_rows['rating'].idxmax()]
        best_bid = int(best_row['book_id'])
        seen = get_user_seen_train(uid)
        candidates = [b for b in candidate_book_ids if b not in seen]
        if not candidates:
            continue
        top = get_topn_for_user(uid, best_bid, candidates, N)
        if not top:
            continue
        tp = len(set(top) & relevant)
        precisions.append(tp / N)
        recalls.append(tp / len(relevant))
        hits = 0
        sum_prec = 0.0
        for i, bid in enumerate(top):
            if bid in relevant:
                hits += 1
                sum_prec += hits / (i + 1)
        ap = sum_prec / min(len(relevant), N) if relevant else 0.0
        aps.append(ap)
        users_evaluated += 1
    return {
        'precision_at_10': float(np.mean(precisions)) if precisions else np.nan,
        'recall_at_10': float(np.mean(recalls)) if recalls else np.nan,
        'map_at_10': float(np.mean(aps)) if aps else np.nan,
        'users_evaluated': users_evaluated
    }

def get_topn_content(uid, query_bid, candidates, N):
    return rank_by_content(query_bid, candidates, top_n=N)

def get_topn_collab(uid, query_bid, candidates, N):
    scores = []
    for b in candidates:
        try:
            est = svd.predict(str(uid), str(b)).est
        except Exception:
            est = (train_df['rating'].mean() if not train_df.empty else 3.0)
        scores.append((b, est))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [b for b, _ in scores[:N]]

def get_topn_hybrid(uid, query_bid, candidates, N, alpha=0.5):
    return hybrid_rank(query_bid, uid, candidates, alpha=alpha, top_n=N)

content_rank_metrics = evaluate_ranking_for_method(lambda uid, q, c, N: get_topn_content(uid, q, c, N), eval_users)
collab_rank_metrics = evaluate_ranking_for_method(lambda uid, q, c, N: get_topn_collab(uid, q, c, N), eval_users)
hybrid_rank_metrics = evaluate_ranking_for_method(lambda uid, q, c, N: get_topn_hybrid(uid, q, c, N, alpha=0.5), eval_users)

# Evaluate rating metrics (MAE/RMSE) only on test rows of selected users
content_preds = []
collab_preds = []
hybrid_preds = []
trues = []

test_subset = test_df[test_df['user_id'].isin(eval_users)].reset_index(drop=True)
for _, row in test_subset.iterrows():
    uid = int(row['user_id'])
    bid = int(row['book_id'])
    true = float(row['rating'])
    trues.append(true)
    c_pred = content_predict_rating(uid, bid)
    content_preds.append(c_pred)
    try:
        s_pred = collab_predict_rating(uid, bid)
    except Exception:
        s_pred = train_df['rating'].mean() if not train_df.empty else 3.0
    collab_preds.append(s_pred)
    h_pred = hybrid_predict_rating(uid, bid, alpha_rating=0.5)
    hybrid_preds.append(h_pred)

def mae_rmse(true_list, pred_list):
    mae = mean_absolute_error(true_list, pred_list)
    rmse = np.sqrt(mean_squared_error(true_list, pred_list))
    return mae, rmse

content_mae, content_rmse = mae_rmse(trues, content_preds)
collab_mae, collab_rmse = mae_rmse(trues, collab_preds)
hybrid_mae, hybrid_rmse = mae_rmse(trues, hybrid_preds)

print("\n--- Ranking evaluation (per-user, top-10) ---")
print("Content-only:")
print(f"  Users evaluated: {content_rank_metrics['users_evaluated']}")
print(f"  Precision@10: {content_rank_metrics['precision_at_10']:.4f}")
print(f"  Recall@10:    {content_rank_metrics['recall_at_10']:.4f}")
print(f"  MAP@10:       {content_rank_metrics['map_at_10']:.4f}\n")

print("Collaborative-only:")
print(f"  Users evaluated: {collab_rank_metrics['users_evaluated']}")
print(f"  Precision@10: {collab_rank_metrics['precision_at_10']:.4f}")
print(f"  Recall@10:    {collab_rank_metrics['recall_at_10']:.4f}")
print(f"  MAP@10:       {collab_rank_metrics['map_at_10']:.4f}\n")

print("Hybrid (alpha=0.5):")
print(f"  Users evaluated: {hybrid_rank_metrics['users_evaluated']}")
print(f"  Precision@10: {hybrid_rank_metrics['precision_at_10']:.4f}")
print(f"  Recall@10:    {hybrid_rank_metrics['recall_at_10']:.4f}")
print(f"  MAP@10:       {hybrid_rank_metrics['map_at_10']:.4f}\n")

print("--- Rating evaluation (test set subset) ---")
print("Content-only:   MAE={:.4f}, RMSE={:.4f}".format(content_mae, content_rmse))
print("Collaborative:  MAE={:.4f}, RMSE={:.4f}".format(collab_mae, collab_rmse))
print("Hybrid (0.5/0.5) MAE={:.4f}, RMSE={:.4f}".format(hybrid_mae, hybrid_rmse))

