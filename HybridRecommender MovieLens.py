from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_curve, auc

# Učitaj podatke
metadata = pd.read_csv('MovieLens/movies_metadata.csv', low_memory=False)
links_small = pd.read_csv('MovieLens/links_small.csv')
ratings = pd.read_csv('MovieLens/ratings_small.csv')

# Priprema podataka
metadata['overview'] = metadata['overview'].fillna('')
metadata['title'] = metadata['title'].fillna('')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype(int)
metadata = metadata[metadata['id'].apply(lambda x: str(x).isdigit())].copy()
metadata['id'] = metadata['id'].astype(int)
metadata_subset = metadata[metadata['id'].isin(links_small)].copy()
metadata_subset['overview'] = metadata_subset['overview'].fillna('')
metadata_subset['title'] = metadata_subset['title'].fillna('')
metadata_subset['genres_str'] = metadata_subset['genres'].fillna('').astype(str)
metadata_subset['tagline'] = metadata_subset['tagline'].fillna('').astype(str)
metadata_subset['original_language'] = metadata_subset['original_language'].fillna('').astype(str)
metadata_subset['keywords_str'] = metadata_subset['keywords'].fillna('').astype(str) if 'keywords' in metadata_subset.columns else ''
metadata_subset['content'] = (
    metadata_subset['overview'] + ' ' +
    metadata_subset['genres_str'] + ' ' +
    metadata_subset['tagline'] + ' ' +
    metadata_subset['original_language'] + ' ' +
    metadata_subset['keywords_str']
)
movieid_to_title = metadata_subset.set_index('id')['title'].dropna().to_dict()
title_to_movieid = {v: k for k, v in movieid_to_title.items()}

# Content-based preporuka
tfidf = TfidfVectorizer(stop_words='english', min_df=2, ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(metadata_subset['content'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
metadata_subset = metadata_subset.reset_index()
indices = pd.Series(metadata_subset.index, index=metadata_subset['title']).drop_duplicates()

def get_content_recommendations(title, top_n=10):
    idx = indices.get(title)
    if idx is None:
        return []
    sim_scores = list(enumerate(np.asarray(cosine_sim[idx]).flatten()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    valid_indices = [i[0] for i in sim_scores[1:top_n+1] if i[0] < len(metadata_subset)]
    return metadata_subset['title'].iloc[valid_indices].tolist()

# Kolaborativno filtriranje (Surprise SVD)
filtered_ratings = ratings[ratings['movieId'].isin(movieid_to_title.keys())]
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(filtered_ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
svd = SVD(n_factors=50, random_state=42)
svd.fit(trainset)

def get_collab_recommendations(user_id, top_n=10):
    user_movies = filtered_ratings['movieId'].unique()
    predictions_user = [svd.predict(user_id, mid) for mid in user_movies]
    predictions_user = sorted(predictions_user, key=lambda x: x.est, reverse=True)
    recommended_ids = [pred.iid for pred in predictions_user[:top_n]]
    return [movieid_to_title.get(mid, f"Movie ID: {mid}") for mid in recommended_ids]

# Hibridni pristup: kombiniraj content i collaborative score (jednostavno zbroji rankove)

original_indices = pd.Series(metadata_subset.index, index=metadata_subset['title']).drop_duplicates()
metadata_subset = metadata_subset.reset_index()
indices = pd.Series(metadata_subset.index, index=metadata_subset['title']).drop_duplicates()

def get_hybrid_recommendations(user_id, query_title, top_n=10, alpha=0.5):
    idx = original_indices.get(query_title)
    if idx is None:
        return []
    content_scores = np.asarray(cosine_sim[idx]).flatten()
    if len(content_scores) != len(metadata_subset):
        content_scores = content_scores[:len(metadata_subset)]
    collab_scores = np.zeros(len(metadata_subset))
    movie_id_to_idx = dict(zip(metadata_subset['id'], metadata_subset.index))
    for mid in metadata_subset['id'].values:
        idx_cb = movie_id_to_idx.get(mid)
        if idx_cb is not None:
            est = svd.predict(user_id, mid).est
            collab_scores[idx_cb] = est
    hybrid_scores = alpha * content_scores + (1 - alpha) * collab_scores
    ranked_indices = np.argsort(hybrid_scores)[::-1]
    query_idx = indices.get(query_title)
    if isinstance(query_idx, pd.Series):
        query_idx = query_idx.iloc[0]
    ranked_indices = [int(i) for i in ranked_indices if i != query_idx]
    top_indices = ranked_indices[:top_n]
    return metadata_subset['title'].iloc[top_indices].tolist()

# Primjer korištenja hibridnog pristupa
user_id = 1
query_title = 'The Godfather'
print("Content-based preporuke za 'The Godfather':")
for title in get_content_recommendations(query_title, top_n=5):
    print(title)
print("\nCollaborative preporuke za userId=1:")
for title in get_collab_recommendations(user_id, top_n=5):
    print(title)
print("\nHibridne preporuke za userId=1 i 'The Godfather':")
for title in get_hybrid_recommendations(user_id, query_title, top_n=5, alpha=0.5):
    print(title)

# --- Evaluacija hibridnog sustava ---
N = 10
all_true = []
all_pred = []
aps = []
y_true = []
y_score = []

# Za Precision@N
N_range = range(1, 16)
precision_at_n = []

# Pripremi podatke za Precision@N
user_recommendations = {}
for user_id, group in filtered_ratings.groupby('userId'):
    watched_titles = group['movieId'].map(movieid_to_title).dropna().tolist()
    if not watched_titles:
        continue
    query_title = watched_titles[0]
    recommended = get_hybrid_recommendations(user_id, query_title, top_n=max(N_range), alpha=0.5)
    user_recommendations[user_id] = recommended
    # True relevantni: svi filmovi koje je korisnik ocijenio (u podskupu)
    relevant = set(watched_titles)
    # Predicted: preporučeni filmovi
    pred = set(recommended)
    # MAP@10
    hits = 0
    sum_precisions = 0
    for i, rec in enumerate(recommended):
        if rec in relevant:
            hits += 1
            sum_precisions += hits / (i + 1)
    ap = sum_precisions / min(len(relevant), N) if relevant else 0
    aps.append(ap)
    # MAE/RMSE: predviđene ocjene su hibridni score (skaliraj na [0.5, 5.0])
    movie_ids_pred = [title_to_movieid.get(t, -1) for t in recommended]
    for t, mid in zip(recommended, movie_ids_pred):
        true_rating = group[group['movieId'] == mid]['rating']
        if not true_rating.empty:
            idx = indices.get(query_title)
            idx_pred = indices.get(t)
            # If idx or idx_pred is a Series (multiple matches), take the first value
            if isinstance(idx, pd.Series):
                idx = idx.iloc[0]
            if isinstance(idx_pred, pd.Series):
                idx_pred = idx_pred.iloc[0]
            if idx is not None and idx_pred is not None:
                content_score = float(cosine_sim[int(idx)][int(idx_pred)])
                collab_score = svd.predict(user_id, mid).est
                hybrid_score = 0.5 * content_score + 0.5 * (collab_score / 5.0)
                min_val = min(0, hybrid_score)
                hybrid_rating = 0.5 + (hybrid_score - min_val) * (5.0 - 0.5)
                all_true.append(true_rating.values[0])
                all_pred.append(collab_score)  # za evaluaciju koristi collaborative predikciju
                y_true.append(1 if true_rating.values[0] >= 4.0 else 0)
                y_score.append(collab_score)

# MAE i RMSE
if all_true and all_pred:
    mae = mean_absolute_error(all_true, all_pred)
    rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    print(f"Hybrid MAE: {mae:.4f}")
    print(f"Hybrid RMSE: {rmse:.4f}")
else:
    print("Nema dovoljno podataka za MAE/RMSE.")

# MAP@10
if aps:
    print(f"Hybrid MAP@{N}: {np.mean(aps):.4f}")
else:
    print("Nema dovoljno podataka za MAP.")

# ROC
if y_true and y_score:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Hybrid Recommender)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
else:
    print("Nema dovoljno podataka za ROC.")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

# Izračun Precision@N za svaki N u N_range
for N in N_range:
    precisions = []
    for uid, rec_titles in user_recommendations.items():
        relevant = set(filtered_ratings[filtered_ratings['userId'] == uid]['movieId'].map(movieid_to_title).dropna())
        pred = set(rec_titles[:N])
        true_positives = len(relevant & pred)
        precision = true_positives / N if N > 0 else 0
        precisions.append(precision)
    precision_at_n.append(np.mean(precisions) if precisions else 0)

# Prikaz Precision@N
plt.figure(figsize=(8,5))
plt.plot(list(N_range), precision_at_n, marker='o')
plt.title('Precision@N za različite vrijednosti N (Hybrid Recommender)')
plt.xlabel('N')
plt.ylabel('Prosječni Precision@N')
plt.grid(True)
plt.tight_layout()
plt.show()