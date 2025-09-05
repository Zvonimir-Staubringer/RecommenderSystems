# %% [markdown]
## Collaborative Filtering Movie Recommender System (Surprise SVD)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split as sk_train_test_split
import os

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

# Napravi reproducibilan train/test split na DataFrame razini (koristit ćemo isti split za sve metode)
train_df, test_df = sk_train_test_split(filtered_ratings, test_size=0.2, random_state=42)

# Train Surprise model na train_df
reader = Reader(rating_scale=(0.5, 5.0))
data_train = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
trainset = data_train.build_full_trainset()

svd = SVD(n_factors=50, random_state=42)
svd.fit(trainset)

# MAE/RMSE: testiranje na test_df koristeći Surprise API (testset treba u obliku (uid,iid,r_ui))
testset_surprise = list(zip(test_df['userId'].astype(int), test_df['movieId'].astype(int), test_df['rating'].astype(float)))
predictions = svd.test(testset_surprise)
mae = accuracy.mae(predictions, verbose=False)
rmse = accuracy.rmse(predictions, verbose=False)
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Helper: predviđanja i top-N rangiranje za sve korisnike u test_df
all_movie_ids = list(movieid_to_title.keys())

# izgradi mapu train user -> set(movieId)
train_user_items = train_df.groupby('userId')['movieId'].apply(set).to_dict()

def get_top_n_by_predicting_all(model, users, candidate_items, train_user_items, n=10):
    top_n = {}
    for uid in users:
        seen = train_user_items.get(uid, set())
        candidates = [mid for mid in candidate_items if mid not in seen]
        preds = []
        for mid in candidates:
            # predict koristi model.predict(uid, iid)
            p = model.predict(uid, mid)
            preds.append((mid, p.est))
        preds.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = [iid for iid, _ in preds[:n]]
    return top_n

# računaj Precision@N i MAP@N koristeći test_df kao relevant set
N = 10
test_users = test_df['userId'].unique().tolist()
top_n_preds = get_top_n_by_predicting_all(svd, test_users, all_movie_ids, train_user_items, n=N)

precision_list = []
aps = []
for uid in test_users:
    relevant = set(test_df[test_df['userId'] == uid]['movieId'])
    preds = top_n_preds.get(uid, [])
    # Precision@N
    tp = len(set(preds) & relevant)
    precision = tp / N if N > 0 else 0.0
    precision_list.append(precision)
    # AP@N
    hits = 0
    sum_prec = 0.0
    for i, mid in enumerate(preds):
        if mid in relevant:
            hits += 1
            sum_prec += hits / (i + 1)
    ap = sum_prec / min(len(relevant), N) if relevant else 0.0
    aps.append(ap)

print(f"Precision@{N}: {np.mean(precision_list):.4f}")
print(f"MAP@{N}: {np.mean(aps):.4f}")

# (Opcional) za reproduktivnost: ako želiš istu proceduru za content i hybrid,
# koristi isti train_df / test_df i istu funkciju get_top_n_by_predicting_all
# ali osiguraj da content/hybrid vraćaju liste movieIds (ili titles koje mapiraš na movieIds).

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

os.makedirs("plots", exist_ok=True)

# distribucija broja test-itema po korisniku
counts = test_df.groupby('userId').size()

# ispisi osnovne statistike
print("Statistika broja test-itema po korisniku:")
print(counts.describe(percentiles=[0.25,0.5,0.75,0.9]))
print("\nBroj korisnika po broju test-itema (prvih 30 redova):")
print(counts.value_counts().sort_index().head(30))

# Histogram (s diskretnim binovima)
plt.figure(figsize=(8,4))
bins = range(1, int(counts.max())+2)
plt.hist(counts, bins=bins, edgecolor='black')
plt.xlabel('Broj test-itema po korisniku')
plt.ylabel('Broj korisnika')
plt.title('Histogram: broj test-itema po korisniku (test_df)')
plt.tight_layout()
plt.savefig("plots/test_items_histogram.png", dpi=150)
plt.show()

# Bar plot frekvencija (ograniči prikaz na razumnu širinu, npr. do 50)
freq = counts.value_counts().sort_index()
max_x = min(50, freq.index.max())
freq_small = freq[freq.index <= max_x]

plt.figure(figsize=(10,4))
plt.bar(freq_small.index, freq_small.values, edgecolor='black')
plt.xlabel('Broj test-itema po korisniku')
plt.ylabel('Broj korisnika')
plt.title(f'Frekvencija broja test-itema (prikaz do {max_x})')
plt.tight_layout()
plt.savefig("plots/test_items_bar.png", dpi=150)
plt.show()