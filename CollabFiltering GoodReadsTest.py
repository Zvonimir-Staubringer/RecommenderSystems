import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Učitaj podatke
books = pd.read_csv('GoodReads/books.csv')
ratings = pd.read_csv('GoodReads/ratings.csv')

# Priprema podataka kao u ContentFiltering GoodReads
ratings_rmv_duplicates = ratings.drop_duplicates()
unwanted_users = ratings_rmv_duplicates.groupby('user_id')['user_id'].count()
unwanted_users = unwanted_users[unwanted_users < 3]
unwanted_ratings = ratings_rmv_duplicates[ratings_rmv_duplicates.user_id.isin(unwanted_users.index)]
new_ratings = ratings_rmv_duplicates.drop(unwanted_ratings.index)

# Smanji skup radi brže analize
max_books = 5000
books_small = books.iloc[:max_books].copy()
book_ids_small = set(books_small['id'])
ratings_small = new_ratings[new_ratings['book_id'].isin(book_ids_small)].copy()

# Surprise priprema
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_small[['user_id', 'book_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# SVD model
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

N = 10
max_users = 1000

top_n = get_top_n(predictions, n=N)
precisions = []
aps = []
user_count = 0
for uid, user_ratings in top_n.items():
    if user_count >= max_users:
        break
    user_count += 1
    relevant = set(ratings_small[ratings_small['user_id'] == uid]['book_id'])
    pred = set(user_ratings)
    true_positives = len(relevant & pred)
    precision = true_positives / N if N > 0 else 0
    precisions.append(precision)
    hits = 0
    sum_precisions = 0
    for i, rec in enumerate(user_ratings):
        if rec in relevant:
            hits += 1
            sum_precisions += hits / (i + 1)
    ap = sum_precisions / min(len(relevant), N) if relevant else 0
    aps.append(ap)

if precisions:
    print(f"Prosjecni Precision@{N}: {np.mean(precisions):.4f}")
else:
    print("Nema dovoljno podataka za Precision@N.")

if aps:
    print(f"Prosjecni MAP@{N}: {np.mean(aps):.4f}")
else:
    print("Nema dovoljno podataka za MAP.")


# Grafički prikaz Precision@N i MAP za različite N (samo za prvih 1000 korisnika)
N_range = range(1, 16)
precision_at_n = []
map_at_n = []
for test_N in N_range:
    top_n = get_top_n(predictions, n=test_N)
    precisions = []
    aps = []
    user_count = 0
    for uid, user_ratings in top_n.items():
        if user_count >= max_users:
            break
        user_count += 1
        relevant = set(ratings_small[ratings_small['user_id'] == uid]['book_id'])
        pred = set(user_ratings)
        true_positives = len(relevant & pred)
        precision = true_positives / test_N if test_N > 0 else 0
        precisions.append(precision)
        hits = 0
        sum_precisions = 0
        for i, rec in enumerate(user_ratings):
            if rec in relevant:
                hits += 1
                sum_precisions += hits / (i + 1)
        ap = sum_precisions / min(len(relevant), test_N) if relevant else 0
        aps.append(ap)
    precision_at_n.append(np.mean(precisions) if precisions else 0)
    map_at_n.append(np.mean(aps) if aps else 0)

plt.figure(figsize=(10,5))
plt.plot(list(N_range), np.round(precision_at_n, 4), marker='o', label='Precision@N')
plt.plot(list(N_range), np.round(map_at_n, 4), marker='o', label='MAP@N')
plt.xlabel('N')
plt.ylabel('Vrijednost metrike')
plt.title('Precision@N i MAP@N za razlicite vrijednosti N (GoodReads SVD, prvih 1000 korisnika)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Evaluacija za više brojeva korisnika
user_counts_to_test = [500, 1000, 2000, 4000, 6000]
results = []

for max_users in user_counts_to_test:
    precisions = []
    aps = []
    all_true = []
    all_pred = []
    user_count = 0
    top_n = get_top_n(predictions, n=N)
    for uid, user_ratings in top_n.items():
        if user_count >= max_users:
            break
        user_count += 1
        relevant = set(ratings_small[ratings_small['user_id'] == uid]['book_id'])
        pred = set(user_ratings)
        true_positives = len(relevant & pred)
        precision = true_positives / N if N > 0 else 0
        precisions.append(precision)
        hits = 0
        sum_precisions = 0
        for i, rec in enumerate(user_ratings):
            if rec in relevant:
                hits += 1
                sum_precisions += hits / (i + 1)
        ap = sum_precisions / min(len(relevant), N) if relevant else 0
        aps.append(ap)
        # MAE/RMSE: predviđene ocjene su est iz SVD predikcije
        for rec in user_ratings:
            true_rating = ratings_small[(ratings_small['user_id'] == uid) & (ratings_small['book_id'] == rec)]['rating']
            if not true_rating.empty:
                # user_ratings je lista book_id, pronađi est iz predictions
                est_rating = next((est for u, b, _, est, _ in predictions if u == uid and b == rec), None)
                if est_rating is not None:
                    all_true.append(true_rating.values[0])
                    all_pred.append(est_rating)
    mae = mean_absolute_error(all_true, all_pred) if all_true and all_pred else None
    rmse = np.sqrt(mean_squared_error(all_true, all_pred)) if all_true and all_pred else None
    precision_val = np.mean(precisions) if precisions else None
    map_val = np.mean(aps) if aps else None
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

# Grafički prikaz
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
plt.xlabel('Broj korisnika')
plt.ylabel('Vrijednost metrike')
plt.title('Evaluacija sustava za različit broj korisnika (Collaborative GoodReads)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()