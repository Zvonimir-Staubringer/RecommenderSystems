import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Učitaj podatke
books = pd.read_csv('GoodReads/books.csv')
ratings = pd.read_csv('GoodReads/ratings.csv')
book_tags = pd.read_csv('GoodReads/book_tags.csv')
tags = pd.read_csv('GoodReads/tags.csv')

# Priprema podataka (kao u ContentFiltering GoodReads)
ratings_rmv_duplicates = ratings.drop_duplicates()
unwanted_users = ratings_rmv_duplicates.groupby('user_id')['user_id'].count()
unwanted_users = unwanted_users[unwanted_users < 3]
unwanted_ratings = ratings_rmv_duplicates[ratings_rmv_duplicates.user_id.isin(unwanted_users.index)]
new_ratings = ratings_rmv_duplicates.drop(unwanted_ratings.index)

# Smanji broj knjiga na 2000 radi bržeg izvođenja
max_books = 2000
books_small = books.iloc[:max_books].copy()
book_ids_small = set(books_small['id'])
ratings_small = new_ratings[new_ratings['book_id'].isin(book_ids_small)].copy()

# Filtriraj korisnike s barem 5 ocjena (manje nego prije)
min_user_ratings = 5
min_book_ratings = 10

user_counts = ratings_small['user_id'].value_counts()
book_counts = ratings_small['book_id'].value_counts()
filtered_ratings = ratings_small[
    (ratings_small['user_id'].isin(user_counts[user_counts >= min_user_ratings].index)) &
    (ratings_small['book_id'].isin(book_counts[book_counts >= min_book_ratings].index))
].copy()

# Stratificirani uzorak korisnika (nasumično odaberi do max_users korisnika)
def get_stratified_users(ratings_df, max_users):
    user_ids = ratings_df['user_id'].unique()
    np.random.seed(42)
    if len(user_ids) > max_users:
        user_ids = np.random.choice(user_ids, max_users, replace=False)
    return user_ids

# Priprema žanrova za svaku knjigu (spoji tagove prema goodreads_book_id)
book_tags_merged = pd.merge(book_tags, tags, left_on='tag_id', right_on='tag_id', how='left')
if 'goodreads_book_id' in book_tags_merged.columns:
    book_genres = book_tags_merged[book_tags_merged['goodreads_book_id'].isin(book_ids_small)]
    book_genres = book_genres.groupby('goodreads_book_id')['tag_name'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
    books_small = pd.merge(books_small, book_genres, left_on='id', right_on='goodreads_book_id', how='left')
else:
    book_genres = book_tags_merged[book_tags_merged['book_id'].isin(book_ids_small)]
    book_genres = book_genres.groupby('book_id')['tag_name'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
    books_small = pd.merge(books_small, book_genres, left_on='id', right_on='book_id', how='left')
books_small['tag_name'] = books_small['tag_name'].fillna('')

books_small['authors'] = books_small['authors'].str.replace(' ', '').str.lower()
if 'description' in books_small.columns:
    books_small['description'] = books_small['description'].fillna('')
    books_small['content'] = books_small['title'] + ' ' + books_small['authors'] + ' ' + books_small['description'] + ' ' + books_small['tag_name']
else:
    books_small['content'] = books_small['title'] + ' ' + books_small['authors'] + ' ' + books_small['tag_name']

# Content-based preporuka (CountVectorizer + cosine similarity)
count = CountVectorizer(stop_words='english', max_features=5000)
count_matrix = count.fit_transform(books_small['content'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)
indices = pd.Series(books_small.index, index=books_small['title']).drop_duplicates()

def get_content_recommendations(title, top_n=10):
    idx = indices.get(title)
    if idx is None:
        return []
    sim_scores = list(enumerate(np.asarray(cosine_sim[idx]).flatten()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    valid_indices = [i[0] for i in sim_scores[1:top_n+1] if i[0] < len(books_small)]
    return books_small['title'].iloc[valid_indices].tolist()

# Collaborative filtering preporuka (Surprise SVD)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_small[['user_id', 'book_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
svd = SVD(n_factors=50, random_state=42)
svd.fit(trainset)

def get_collab_recommendations(user_id, top_n=10):
    user_books = books_small['id'].values
    predictions_user = [svd.predict(user_id, bid) for bid in user_books]
    predictions_user = sorted(predictions_user, key=lambda x: x.est, reverse=True)
    recommended_ids = [pred.iid for pred in predictions_user[:top_n]]
    return [books_small.set_index('id').title.get(bid, f"Book ID: {bid}") for bid in recommended_ids]

def get_hybrid_recommendations(user_id, query_title, top_n=10, alpha=0.3):
    idx = indices.get(query_title)
    if idx is None:
        return []
    content_scores = np.asarray(cosine_sim[idx]).flatten()
    if len(content_scores) != len(books_small):
        content_scores = content_scores[:len(books_small)]
    collab_scores = np.zeros(len(books_small))
    book_id_to_idx = dict(zip(books_small['id'], books_small.index))
    for bid in books_small['id'].values:
        idx_cb = book_id_to_idx.get(bid)
        if idx_cb is not None:
            est = svd.predict(user_id, bid).est
            collab_scores[idx_cb] = est
    hybrid_scores = alpha * content_scores + (1 - alpha) * collab_scores
    ranked_indices = np.argsort(hybrid_scores)[::-1]
    query_idx = indices.get(query_title)
    if isinstance(query_idx, pd.Series):
        query_idx = query_idx.iloc[0]
    ranked_indices = [int(i) for i in ranked_indices if i != query_idx]
    top_indices = ranked_indices[:top_n]
    return books_small['title'].iloc[top_indices].tolist()

# Re-train SVD na filtriranim podacima s manje faktora radi brzine
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(filtered_ratings[['user_id', 'book_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
svd = SVD(n_factors=20, random_state=42)
svd.fit(trainset)

N = 10
user_counts_to_test = [200, 400, 800]
results = []

bookid_to_title = books_small.set_index('id')['title'].dropna().to_dict()
title_to_bookid = {v: k for k, v in bookid_to_title.items()}

for max_users in user_counts_to_test:
    user_precision = []
    user_map = []
    all_true = []
    all_pred = []
    user_ids = get_stratified_users(filtered_ratings, max_users)
    for user_id in user_ids:
        group = filtered_ratings[filtered_ratings['user_id'] == user_id]
        rated_titles = group['book_id'].map(bookid_to_title).dropna().tolist()
        if not rated_titles:
            continue
        recommended_set = set()
        for query_title in rated_titles:
            recommended = get_hybrid_recommendations(user_id, query_title, top_n=N, alpha=0.3)
            recommended_set.update(recommended)
        recommended_list = list(recommended_set)[:N]
        relevant = set(rated_titles)
        pred = set(recommended_list)
        true_positives = len(relevant & pred)
        precision = true_positives / N if N > 0 else 0
        user_precision.append(precision)
        hits = 0
        sum_precisions = 0
        for i, rec in enumerate(recommended_list):
            if rec in relevant:
                hits += 1
                sum_precisions += hits / (i + 1)
        ap = sum_precisions / min(len(relevant), N) if relevant else 0
        user_map.append(ap)
        pred_rating = group['rating'].mean()
        for t in recommended_list:
            true_rating = group[group['book_id'] == title_to_bookid.get(t, -1)]['rating']
            if not true_rating.empty:
                all_true.append(true_rating.values[0])
                all_pred.append(pred_rating if not np.isnan(pred_rating) else 3.0)
    mae = mean_absolute_error(all_true, all_pred) if all_true and all_pred else None
    rmse = np.sqrt(mean_squared_error(all_true, all_pred)) if all_true and all_pred else None
    precision_val = np.mean(user_precision) if user_precision else None
    map_val = np.mean(user_map) if user_map else None
    results.append({
        'users': max_users,
        'MAE': mae,
        'RMSE': rmse,
        'Precision@N': precision_val,
        'MAP': map_val
    })
    mae_str = f"{mae:.4f}" if mae is not None else "None"
    rmse_str = f"{rmse:.4f}" if rmse is not None else "None"
    precision_str = f"{precision_val:.4f}" if precision_val is not None else "None"
    map_str = f"{map_val:.4f}" if map_val is not None else "None"
    print(f"Evaluacija za {max_users} korisnika: MAE={mae_str}, RMSE={rmse_str}, Precision@{N}={precision_str}, MAP={map_str}")

# Grafički prikaz svih metrika na jednom grafu (x-osa: maksimalni broj korisnika)
users = [r['users'] for r in results]
mae_vals = [r['MAE'] if r['MAE'] is not None else np.nan for r in results]
rmse_vals = [r['RMSE'] if r['RMSE'] is not None else np.nan for r in results]
precision_vals = [r['Precision@N'] if r['Precision@N'] is not None else np.nan for r in results]
map_vals = [r['MAP'] if r['MAP'] is not None else np.nan for r in results]

plt.figure(figsize=(10,5))
plt.plot(users, np.round(mae_vals, 4), marker='o', label='MAE')
plt.plot(users, np.round(rmse_vals, 4), marker='o', label='RMSE')
plt.plot(users, np.round(precision_vals, 4), marker='o', label=f'Precision@{N}')
plt.plot(users, np.round(map_vals, 4), marker='o', label='MAP')
plt.xlabel('Maksimalni broj korisnika')
plt.ylabel('Vrijednost metrike')
plt.title('Evaluacija hibridnog sustava za različit broj korisnika (GoodReads)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()