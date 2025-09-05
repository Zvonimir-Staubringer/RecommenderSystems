import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Učitavanje podataka
required_files = [
    'GoodReads/books.csv',
    'GoodReads/ratings.csv'
]
for f in required_files:
    if not os.path.isfile(f):
        print(f"ERROR: Required file not found: {f}")
        exit(1)

books = pd.read_csv('GoodReads/books.csv')
ratings = pd.read_csv('GoodReads/ratings.csv')

# Prikaz osnovnih informacija o books datasetu
print("\nPrvih 5 redaka books dataseta:")
print(books.head())

print("\nShape books:", books.shape)
print("\nStupci books:", books.columns.tolist())

print("\nInfo o books:")
print(books.info())

# Osnovne informacije o ocjenama
print("\nBroj korisnika:", ratings['user_id'].nunique())
print("Broj knjiga:", ratings['book_id'].nunique())
print("Ukupno ocjena:", len(ratings))
print("\nStatistika ocjena:")
print(ratings['rating'].describe())

# Statistika po knjizi
book_stats = ratings.groupby('book_id').agg({'rating': ['count', 'mean']}).reset_index()
book_stats.columns = ['book_id', 'rating_count', 'rating_mean']
book_stats = pd.merge(book_stats, books[['id', 'title', 'authors', 'original_publication_year']], left_on='book_id', right_on='id', how='left')

# Ukloni duplikate i nan vrijednosti iz naslova
book_stats = book_stats[book_stats['title'].notna()]

# Najviše ocjenjivane knjige
most_rated = book_stats.sort_values('rating_count', ascending=False)
most_rated = most_rated[most_rated['title'].notna()]
most_rated = most_rated.drop_duplicates(subset=['title'])
most_rated = most_rated.head(10)

# Top 10 najbolje ocijenjenih knjiga (min 10 ocjena)
top_rated = book_stats[book_stats['rating_count'] >= 10].sort_values('rating_mean', ascending=False).head(10)
top_rated['title_with_rating'] = top_rated.apply(
    lambda row: f"{row['title']} ({row['rating_mean']:.2f})", axis=1
)

# Kreiraj folder za rezultate ako ne postoji
results_dir = "GoodReads Rezultati"
os.makedirs(results_dir, exist_ok=True)

# Vizualizacije
plt.figure(figsize=(20, 10))

# Distribucija ocjena
plt.subplot(2, 2, 1)
sns.histplot(ratings['rating'], bins=[0.5,1.5,2.5,3.5,4.5,5.5], kde=False)
plt.title('Distribucija ocjena')
plt.xlabel('Ocjena')
plt.ylabel('Broj ocjena')
plt.xticks([1,2,3,4,5])

# Top 10 najviše ocjenjivanih knjiga
plt.subplot(2, 2, 2)
sns.barplot(y='title', x='rating_count', data=most_rated)
plt.title('Top 10 najviše ocjenjivanih knjiga')
plt.xlabel('Broj ocjena')
plt.ylabel('Knjiga')

# Top 10 najbolje ocijenjenih knjiga
plt.subplot(2, 2, 3)
sns.barplot(y='title_with_rating', x='rating_mean', data=top_rated)
plt.title('Top 10 najbolje ocijenjenih knjiga (min 10 ocjena)')
plt.xlabel('Prosječna ocjena')
plt.ylabel('Knjiga')

# Distribucija broja ocjena po korisniku
plt.subplot(2, 2, 4)
user_counts = ratings['user_id'].value_counts()
sns.histplot(user_counts, bins=30)
plt.title('Distribucija broja ocjena po korisniku')
plt.xlabel('Broj ocjena')
plt.ylabel('Broj korisnika')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'goodreads_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# 1. Distribucija knjiga po godini izlaska
plt.figure(figsize=(12,6))
sns.histplot(books['original_publication_year'].dropna(), bins=40, kde=False)
plt.title('Distribucija knjiga po godini izlaska')
plt.xlabel('Godina izlaska')
plt.ylabel('Broj knjiga')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'books_year_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Distribucija knjiga po autorima (top 15)
top_authors = books['authors'].value_counts().head(15).reset_index()
top_authors.columns = ['author', 'count']
plt.figure(figsize=(12,6))
sns.barplot(y='author', x='count', data=top_authors)
plt.title('Top 15 autora po broju knjiga')
plt.xlabel('Broj knjiga')
plt.ylabel('Autor')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'books_author_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Prosječna ocjena po autorima (top 15 po broju knjiga)
author_ratings = book_stats.groupby('authors').agg({'rating_mean': 'mean', 'book_id': 'count'}).reset_index()
author_ratings = author_ratings[author_ratings['book_id'] >= 5].sort_values('rating_mean', ascending=False).head(15)
plt.figure(figsize=(12,6))
sns.barplot(y='authors', x='rating_mean', data=author_ratings)
plt.title('Prosječna ocjena autora (min 5 knjiga, top 15)')
plt.xlabel('Prosječna ocjena')
plt.ylabel('Autor')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'author_avg_rating.png'), dpi=300, bbox_inches='tight')
plt.close()

# Distribucija ocjena
plt.figure(figsize=(8, 6))
sns.histplot(ratings['rating'], bins=[0.5,1.5,2.5,3.5,4.5,5.5], kde=False)
plt.title('Distribucija ocjena')
plt.xlabel('Ocjena')
plt.ylabel('Broj ocjena')
plt.xticks([1,2,3,4,5])
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'distribucija_ocjena.png'), dpi=300, bbox_inches='tight')
plt.close()

# Top 10 najviše ocjenjivanih knjiga
plt.figure(figsize=(8, 6))
sns.barplot(y='title', x='rating_count', data=most_rated)
plt.title('Top 10 najviše ocjenjivanih knjiga')
plt.xlabel('Broj ocjena')
plt.ylabel('Knjiga')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'top10_najvise_ocjenjivanih.png'), dpi=300, bbox_inches='tight')
plt.close()

# Top 10 najbolje ocijenjenih knjiga
plt.figure(figsize=(8, 6))
sns.barplot(y='title_with_rating', x='rating_mean', data=top_rated)
plt.title('Top 10 najbolje ocijenjenih knjiga (min 10 ocjena)')
plt.xlabel('Prosječna ocjena')
plt.ylabel('Knjiga')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'top10_najbolje_ocijenjenih.png'), dpi=300, bbox_inches='tight')
plt.close()

# Distribucija broja ocjena po korisniku
plt.figure(figsize=(8, 6))
user_counts = ratings['user_id'].value_counts()
sns.histplot(user_counts, bins=30)
plt.title('Distribucija broja ocjena po korisniku')
plt.xlabel('Broj ocjena')
plt.ylabel('Broj korisnika')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'distribucija_ocjena_po_korisniku.png'), dpi=300, bbox_inches='tight')
plt.close()

# --- Distribucija ocjena po žanrovima i prosječne ocjene po žanrovima ---

# Pretpostavka: žanrovi su u stupcu 'genres' u books.csv kao string s listom žanrova odvojenih '|'
if 'genres' in books.columns:
    # Priprema: proširi žanrove po knjizi
    books['genres_list'] = books['genres'].fillna('').apply(lambda x: [g.strip() for g in x.split('|') if g.strip()])
    # Spoji ocjene s žanrovima
    ratings_genres = pd.merge(ratings, books[['id', 'genres_list']], left_on='book_id', right_on='id', how='left')
    # Eksplodiraj žanrove
    ratings_genres_exploded = ratings_genres.explode('genres_list')
    # Filtriraj samo žanrove koji nisu prazni
    ratings_genres_exploded = ratings_genres_exploded[ratings_genres_exploded['genres_list'].notna() & (ratings_genres_exploded['genres_list'] != '')]

    # Distribucija ocjena po žanrovima (top 10 žanrova)
    top_genres = ratings_genres_exploded['genres_list'].value_counts().head(10).index.tolist()
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='genres_list', y='rating', data=ratings_genres_exploded[ratings_genres_exploded['genres_list'].isin(top_genres)])
    plt.title('Distribucija ocjena po žanrovima (top 10 žanrova)')
    plt.xlabel('Žanr')
    plt.ylabel('Ocjena')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'distribucija_ocjena_po_zanru.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Prosječna ocjena po žanru (top 10 žanrova)
    genre_avg = ratings_genres_exploded.groupby('genres_list')['rating'].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=genre_avg.values, y=genre_avg.index)
    plt.title('Prosječna ocjena po žanru (top 10 žanrova)')
    plt.xlabel('Prosječna ocjena')
    plt.ylabel('Žanr')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'prosjek_ocjena_po_zanru.png'), dpi=300, bbox_inches='tight')
    plt.close()
else:
    print("Stupac 'genres' nije pronađen u books.csv. Analiza po žanrovima nije moguća.")
