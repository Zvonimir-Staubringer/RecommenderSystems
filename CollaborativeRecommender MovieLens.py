# %% [markdown]
## Collaborative Filtering Movie Recommender System (Surprise SVD)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.metrics import roc_curve, auc

# Učitavanje podataka
ratings = pd.read_csv('MovieLens/ratings_small.csv')
links_small = pd.read_csv('MovieLens/links_small.csv')
movies_metadata = pd.read_csv('MovieLens/movies_metadata.csv', low_memory=False)

# Priprema podatkovnog skupa za daljnje korištenje
movies_metadata = movies_metadata[movies_metadata['id'].apply(lambda x: str(x).isdigit())].copy()
movies_metadata['id'] = movies_metadata['id'].astype(int)
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype(int)
movies_subset = movies_metadata[movies_metadata['id'].isin(links_small)].copy()
movieid_to_title = movies_subset.set_index('id')['title'].dropna().to_dict()

# Filtriraj ratings na podskup filmova
filtered_ratings = ratings[ratings['movieId'].isin(movieid_to_title.keys())]

# Pripema za korištenje Surprise biblioteke
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(filtered_ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Inicijalizacija i treniranje SVD modela
svd = SVD(n_factors=50, random_state=42)
svd.fit(trainset)
predictions = svd.test(testset)

# Evaluacija
mae = accuracy.mae(predictions, verbose=False)
rmse = accuracy.rmse(predictions, verbose=False)
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Precision@N i MAP
def get_top_n(predictions, n=10):
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        top_n.setdefault(uid, [])
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = [iid for (iid, _) in user_ratings[:n]]
    return top_n

N_range = range(1, 16)
precision_at_n = []
for N in N_range:
    top_n = get_top_n(predictions, n=N)
    precisions = []
    for uid, user_ratings in top_n.items():
        relevant = set(filtered_ratings[filtered_ratings['userId'] == uid]['movieId'])
        pred = set(user_ratings)
        true_positives = len(relevant & pred)
        precision = true_positives / N if N > 0 else 0
        precisions.append(precision)
    precision_at_n.append(np.mean(precisions) if precisions else 0)

plt.figure(figsize=(8,5))
plt.plot(list(N_range), precision_at_n, marker='o')
plt.title('Precision@N za različite vrijednosti N (Surprise SVD)')
plt.xlabel('N')
plt.ylabel('Prosječni Precision@N')
plt.grid(True)
plt.tight_layout()
plt.show()

# MAP za sve korisnike za N=10
N = 10
top_n = get_top_n(predictions, n=N)
aps = []
user_aps = {}
for uid, user_ratings in top_n.items():
    relevant = set(filtered_ratings[filtered_ratings['userId'] == uid]['movieId'])
    pred = user_ratings
    hits = 0
    sum_precisions = 0
    for i, rec in enumerate(pred):
        if rec in relevant:
            hits += 1
            sum_precisions += hits / (i + 1)
    ap = sum_precisions / min(len(relevant), N) if relevant else 0
    aps.append(ap)
    user_aps[uid] = ap

if aps:
    print(f"MAP@{N}: {np.mean(aps):.4f}")
else:
    print("Nema dovoljno podataka za MAP.")


# Preporučivanje za specifičnog korisnika
user_id = 1
user_ratings = ratings[ratings['userId'] == user_id]

predictions_for_user = []
for movie_id in ratings['movieId'].unique():
    if movie_id not in user_ratings['movieId'].values:
        pred = svd.predict(user_id, movie_id)
        predictions_for_user.append((movie_id, pred.est))

# Sortiranje po procijenjenoj ocjeni
predictions_for_user.sort(key=lambda x: x[1], reverse=True)

# Uzmi top 5 preporuka filmova
top_n = 5
print(f"Top {top_n} preporuka za korisnika {user_id}:")
for movie_id, est_rating in predictions_for_user[:top_n]:
    title = movieid_to_title.get(movie_id, f"Movie ID: {movie_id}")
    print(f"{title} (Predvidena ocjena: {est_rating:.2f})")

# ROC za predviđanja (binariziraj ocjene: relevant >= 4.0)
y_true = []
y_score = []
for uid, iid, true_r, est, _ in predictions:
    y_true.append(1 if true_r >= 4.0 else 0)
    y_score.append(est)

fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Surprise SVD)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()