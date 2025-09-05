import os
import pandas as pd
import numpy as np

# Postavke
OUT_DIR = "GoodReads"
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_STATE = 42

# Učitaj sirove podatke
books = pd.read_csv(os.path.join(OUT_DIR, "books.csv"))
ratings = pd.read_csv(os.path.join(OUT_DIR, "ratings.csv"))

# Osiguraj ispravne nazive stupaca (biblioteka GoodReads: ratings.book_id, ratings.user_id)
if 'book_id' not in ratings.columns:
    raise KeyError("ratings.csv mora sadržavati stupac 'book_id'")
if 'user_id' not in ratings.columns:
    raise KeyError("ratings.csv mora sadržavati stupac 'user_id'")

# Zadrži samo ocjene za koje imamo metapodatke o knjizi
valid_book_ids = set(books['id'].astype(int).unique())
ratings = ratings[ratings['book_id'].isin(valid_book_ids)].copy()

# Konvertiraj tipove radi konzistentnosti
ratings['book_id'] = ratings['book_id'].astype(int)
ratings['user_id'] = ratings['user_id'].astype(int)
if 'rating' in ratings.columns:
    ratings['rating'] = pd.to_numeric(ratings['rating'], errors='coerce')

# ovdje uzimamo srednju ocjenu ako postoji više zapisa za isti (user_id, book_id)
ratings = ratings.groupby(['user_id','book_id'], as_index=False)['rating'].mean()

# prije izrade split-a: zadrži samo top M najpopularnijih knjiga u ratings
M = 3200
book_counts = ratings['book_id'].value_counts()
top_m = set(book_counts.index[:M].astype(int))
ratings = ratings[ratings['book_id'].isin(top_m)].copy()

# Per-user holdout: uzmi test_frac udio ocjena po korisniku (ostavi barem 1 u train)
test_frac = 0.2
rng = np.random.RandomState(RANDOM_STATE)
train_rows = []
test_rows = []

for uid, group in ratings.groupby('user_id'):
    n = len(group)
    if n >= 2:
        # broj test stavki = max(1, round(n * test_frac)) ali manje od n
        n_test = max(1, int(round(n * test_frac)))
        n_test = min(n_test, n - 1)
        test_idx = rng.choice(group.index, size=n_test, replace=False)
        test_rows.append(group.loc[test_idx])
        train_rows.append(group.drop(test_idx))
    else:
        train_rows.append(group)

train_df = pd.concat([g if isinstance(g, pd.DataFrame) else pd.DataFrame([g]) for g in train_rows], ignore_index=True)
test_df = pd.concat([g if isinstance(g, pd.DataFrame) else pd.DataFrame([g]) for g in test_rows], ignore_index=True).reset_index(drop=True)

# safety: ukloni duplicate (ako postoje)
train_df = train_df.drop_duplicates().reset_index(drop=True)
test_df = test_df.drop_duplicates().reset_index(drop=True)

# test items u skupu
test_items = set(test_df['book_id'].astype(int).unique())
# koliko test-itema postoji u originalnim metadata (valid_book_ids) i u top-M (ako je primjenjeno)
coverage_in_books = len(test_items & valid_book_ids)
coverage_in_topm = len(test_items & top_m) if 'top_m' in locals() else None
print(f"Coverage diagnostics:")
print(f"  unique test items: {len(test_items)}")
print(f"  in original books metadata: {coverage_in_books} ({coverage_in_books/len(test_items):.2%})")
if coverage_in_topm is not None:
    print(f"  in top-M filtered set:     {coverage_in_topm} ({coverage_in_topm/len(test_items):.2%})")

# broj relevantnih po korisniku (distribucija)
rels_per_user = test_df.groupby('user_id').size()
print(f"  relevant per user: mean={rels_per_user.mean():.2f}, median={rels_per_user.median():.0f}, max={rels_per_user.max():.0f}")

# postoji li curenje (test items koji se pojavljuju i u train za ISTOG korisnika) — sanity check
leaks = 0
for uid, g in test_df.groupby('user_id'):
    user_test_items = set(g['book_id'].astype(int).tolist())
    user_train_items = set(train_df[train_df['user_id'] == uid]['book_id'].astype(int).tolist())
    if user_test_items & user_train_items:
        leaks += 1
print(f"  users with overlap between their train and test items (should be 0): {leaks}")

report_path = os.path.join(OUT_DIR, "split_coverage_report.txt")
with open(report_path, "w", encoding="utf-8") as fh:
    fh.write("Coverage diagnostics\n")
    fh.write(f"unique_test_items: {len(test_items)}\n")
    fh.write(f"in_original_books: {coverage_in_books}\n")
    if coverage_in_topm is not None:
        fh.write(f"in_top_M: {coverage_in_topm}\n")
    fh.write(f"relevants_per_user_mean: {rels_per_user.mean():.4f}\n")
    fh.write(f"relevants_per_user_median: {rels_per_user.median():.0f}\n")
    fh.write(f"relevants_per_user_max: {rels_per_user.max():.0f}\n")
    fh.write(f"users_with_train_test_overlap: {leaks}\n")

# Stats i sanity checks
n_users = ratings['user_id'].nunique()
n_train = len(train_df)
n_test = len(test_df)
users_with_test = test_df['user_id'].nunique()

print(f"Ukupno korisnika u originalu: {n_users}")
print(f"Train rows: {n_train}, Test rows: {n_test}, Users with test item: {users_with_test}")

# Spremanje CSV-ove
train_out = os.path.join(OUT_DIR, "train_df.csv")
test_out = os.path.join(OUT_DIR, "test_df.csv")
mapping_out = os.path.join(OUT_DIR, "bookid_to_title.csv")
books_subset_out = os.path.join(OUT_DIR, "books_subset.csv")

train_df.to_csv(train_out, index=False)
test_df.to_csv(test_out, index=False)

# bookid -> title mapping (samo za knjige koje su u ratingu)
books_subset = books[books['id'].isin(ratings['book_id'].unique())].copy()

try:
    book_tags = pd.read_csv(os.path.join(OUT_DIR, "book_tags.csv"))
    tags = pd.read_csv(os.path.join(OUT_DIR, "tags.csv"))
    bt = pd.merge(book_tags, tags, left_on='tag_id', right_on='tag_id', how='left')
    # book_tags može imati stupac 'goodreads_book_id' ili 'book_id'
    if 'goodreads_book_id' in bt.columns:
        key_col = 'goodreads_book_id'
    elif 'book_id' in bt.columns:
        key_col = 'book_id'
    else:
        key_col = None

    if key_col is not None:
        bt = bt[bt[key_col].isin(books_subset['id'])]
        tag_agg = bt.groupby(key_col)['tag_name'].apply(lambda x: ' '.join(x.dropna().astype(str))).reset_index()
        tag_agg.rename(columns={key_col: 'id'}, inplace=True)
        books_subset = pd.merge(books_subset, tag_agg, on='id', how='left')
        books_subset['tag_name'] = books_subset['tag_name'].fillna('')
    else:
        # ako nema odgovarajućeg ključa, samo dodaj prazan stupac
        books_subset['tag_name'] = ''
except FileNotFoundError:
    books_subset['tag_name'] = ''
except Exception:
    # u slučaju problema, ne prekidaj split, samo nastavi bez tagova
    books_subset['tag_name'] = ''

# Spremi books_subset (s tag_name stupcem ako je dostupno)
books_subset.to_csv(books_subset_out, index=False)

# Spremi mapu id -> title (+ tag_name ako postoji)
book_map = books_subset[['id', 'title']].drop_duplicates()
if 'tag_name' in books_subset.columns:
    book_map = pd.merge(book_map, books_subset[['id', 'tag_name']].drop_duplicates(), on='id', how='left')
book_map.to_csv(mapping_out, index=False)

print(f"Spremljeno: {train_out}, {test_out}, {mapping_out}, {books_subset_out}")
print("Napomena: test skup sadrži po jedan test-item za korisnike koji su imali >=2 ocjene.")