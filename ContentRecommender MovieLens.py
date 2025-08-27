import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Učitavanje podataka
metadata = pd.read_csv('MovieLens/movies_metadata.csv', low_memory=False)
links_small = pd.read_csv('MovieLens/links_small.csv')

# Priprema podataka: popunjavanje NaN i pretvaranje u string
metadata['overview'] = metadata['overview'].fillna('')
metadata['title'] = metadata['title'].fillna('')

# Priprema tmdbId iz links_small
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype(int)

# Uzimanje podskupa filmova te spajanje korisnih značajki
metadata = metadata[metadata['id'].apply(lambda x: str(x).isdigit())].copy()
metadata['id'] = metadata['id'].astype(int)
metadata_subset = metadata[metadata['id'].isin(links_small)].copy()

metadata_subset['overview'] = metadata_subset['overview']
metadata_subset['title'] = metadata_subset['title']
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

print(f"Broj filmova u subsetu za preporuke: {len(metadata_subset)}")

tfidf = TfidfVectorizer(stop_words='english', min_df=2, ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(metadata_subset['content'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

metadata_subset = metadata_subset.reset_index()
indices = pd.Series(metadata_subset.index, index=metadata_subset['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices.get(title)
    if idx is None:
        print("Film nije pronađen ili nije u podskupu.")
        return []
    sim_scores = list(enumerate(np.asarray(cosine_sim[idx]).flatten()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    valid_indices = [i[0] for i in sim_scores[1:26] if i[0] < len(metadata_subset)]
    return metadata_subset['title'].iloc[valid_indices].tolist()

# Pripremi mapu: filmId -> title za podskup
movieid_to_title = metadata_subset.set_index('id')['title'].dropna().to_dict()
title_to_movieid = {v: k for k, v in movieid_to_title.items()}

# Dodaj učitavanje ratings_small.csv prije evaluacije
ratings = pd.read_csv('MovieLens/ratings_small.csv')

# Generiraj predviđanja: za svakog korisnika preporuči N filmova
N = 10
user_precision = []
user_map = []
all_true = []
all_pred = []

for user_id, group in ratings.groupby('userId'):
    watched_titles = group['movieId'].map(movieid_to_title).dropna().tolist()
    # Ako korisnik nema filmove u podskupu, preskoči
    if not watched_titles:
        continue
    # Uzmi prvi film koji je korisnik ocijenio kao "query"
    query_title = watched_titles[0]
    recommended = get_recommendations(query_title)
    # True relevantni: svi filmovi koje je korisnik ocijenio (u podskupu)
    relevant = set(watched_titles)
    # Predicted: preporučeni filmovi
    pred = set(recommended)
    # Precision@N
    true_positives = len(relevant & pred)
    precision = true_positives / N if N > 0 else 0
    user_precision.append(precision)
    # MAP
    hits = 0
    sum_precisions = 0
    for i, rec in enumerate(recommended):
        if rec in relevant:
            hits += 1
            sum_precisions += hits / (i + 1)
    ap = sum_precisions / min(len(relevant), N) if relevant else 0
    user_map.append(ap)
# Za MAE/RMSE: predviđene ocjene su prosječna ocjena query filma (jer content-based ne predviđa ocjenu)
    pred_rating = group[group['movieId'] == title_to_movieid.get(query_title, -1)]['rating'].mean()
    for t in recommended:
        true_rating = group[group['movieId'] == title_to_movieid.get(t, -1)]['rating']
        if not true_rating.empty:
            all_true.append(true_rating.values[0])
            all_pred.append(pred_rating if not np.isnan(pred_rating) else 3.0)

# MAE i RMSE
if all_true and all_pred:
    mae = mean_absolute_error(all_true, all_pred)
    rmse = np.sqrt(mean_squared_error(all_true, all_pred))  # ispravno izračunaj RMSE
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
else:
    print("Nema dovoljno podataka za MAE/RMSE.")

# Precision@N i MAP
if user_precision:
    print(f"Prosjecni Precision@{N}: {np.mean(user_precision):.4f}")
else:
    print("Nema dovoljno podataka za Precision@N.")

if user_map:
    print(f"Prosjecni MAP: {np.mean(user_map):.4f}")
else:
    print("Nema dovoljno podataka za MAP.")

# Testiraj Precision@N za N od 1 do 15 i prikaži na grafu
precision_at_n = []
N_range = range(1, 16)
for test_N in N_range:
    user_precision = []
    for user_id, group in ratings.groupby('userId'):
        watched_titles = group['movieId'].map(movieid_to_title).dropna().tolist()
        if not watched_titles:
            continue
        query_title = watched_titles[0]
        recommended = get_recommendations(query_title)[:test_N]
        relevant = set(watched_titles)
        pred = set(recommended)
        true_positives = len(relevant & pred)
        precision = true_positives / test_N if test_N > 0 else 0
        user_precision.append(precision)
    if user_precision:
        precision_at_n.append(np.mean(user_precision))
    else:
        precision_at_n.append(0)

plt.figure(figsize=(8,5))
plt.plot(list(N_range), precision_at_n, marker='o')
plt.title('Precision@N za različite vrijednosti N')
plt.xlabel('N')
plt.ylabel('Prosjecni Precision@N')
plt.grid(True)
plt.tight_layout()
plt.show()

# Primjer preporuke
print("Preporuceni filmovi za 'The Godfather':")
for title in get_recommendations('The Godfather'):
    print(title.encode('utf-8', errors='replace').decode('utf-8'))