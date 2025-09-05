import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast
from collections import Counter

# Provjera datoteka
required_files = [
    'MovieLens/movies_metadata.csv',
    'MovieLens/links_small.csv',
    'MovieLens/ratings_small.csv'
]

# Učitavanje podataka
metadata = pd.read_csv('MovieLens/movies_metadata.csv', low_memory=False)
links_small = pd.read_csv('MovieLens/links_small.csv')
ratings_small = pd.read_csv('MovieLens/ratings_small.csv')

# Priprema movies_metadata: movieId mora biti int
metadata['id'] = pd.to_numeric(metadata['id'], errors='coerce')
links_small['tmdbId'] = pd.to_numeric(links_small['tmdbId'], errors='coerce')

# Spajanje: links_small (movieId, tmdbId) + metadata (id=tmdbId)
movies_small = pd.merge(
    links_small,
    metadata,
    left_on='tmdbId',
    right_on='id',
    how='left'
)

# Filtriraj ocjene prema small podskupu
small_movie_ids = links_small['movieId'].unique()
ratings_small = ratings_small[ratings_small['movieId'].isin(small_movie_ids)]

# Osnovne informacije
print("Broj korisnika:", ratings_small['userId'].nunique())
print("Broj filmova:", ratings_small['movieId'].nunique())
print("Ukupno ocjena:", len(ratings_small))
print("\nStatistika ocjena:")
print(ratings_small['rating'].describe())

# Statistika po filmu
movie_stats = ratings_small.groupby('movieId').agg({'rating': ['count', 'mean']}).reset_index()
movie_stats.columns = ['movieId', 'rating_count', 'rating_mean']

# Dodaj naslov filmu
movie_stats = pd.merge(movie_stats, movies_small[['movieId', 'title']], on='movieId', how='left')

# Najpopularniji filmovi (po broju ocjena)
most_rated = movie_stats.sort_values('rating_count', ascending=False).head(10)

# Najbolje ocijenjeni filmovi (min 5 ocjena)
top_rated = movie_stats[movie_stats['rating_count'] >= 5].sort_values('rating_mean', ascending=False).head(10)
# Dodaj prosječnu ocjenu u naslov
top_rated['title_with_rating'] = top_rated.apply(
    lambda row: f"{row['title']} ({row['rating_mean']:.2f})", axis=1
)

# Prikaz osnovnih informacija o movies_small
print("\nPrvih 5 redaka movies_small dataseta:")
print(movies_small.head())

print("\nShape movies_small:", movies_small.shape)
print("\nStupci movies_small:", movies_small.columns.tolist())

print("\nInfo o movies_small:")
print(movies_small.info())

# Kreiraj folder za rezultate ako ne postoji
results_dir = "MovieLens Rezultati"
os.makedirs(results_dir, exist_ok=True)

# Vizualizacije
plt.figure(figsize=(20, 10))

# Distribucija ocjena
plt.subplot(2, 2, 1)
sns.histplot(ratings_small['rating'], bins=[0.5,1.5,2.5,3.5,4.5,5.5], kde=False)
plt.title('Distribucija ocjena')
plt.xlabel('Ocjena')
plt.ylabel('Broj ocjena')
plt.xticks([1,2,3,4,5])

# Top 10 najviše ocjenjivanih filmova
plt.subplot(2, 2, 2)
sns.barplot(y='title', x='rating_count', data=most_rated)
plt.title('Top 10 najviše ocjenjivanih filmova')
plt.xlabel('Broj ocjena')
plt.ylabel('Film')

# Top 10 najbolje ocijenjenih filmova
plt.subplot(2, 2, 3)
sns.barplot(y='title_with_rating', x='rating_mean', data=top_rated)
plt.title('Top 10 najbolje ocijenjenih filmova (min 5 ocjena)')
plt.xlabel('Prosječna ocjena')
plt.ylabel('Film')

# Distribucija broja ocjena po korisniku
plt.subplot(2, 2, 4)
user_counts = ratings_small['userId'].value_counts()
sns.histplot(user_counts, bins=30)
plt.title('Distribucija broja ocjena po korisniku')
plt.xlabel('Broj ocjena')
plt.ylabel('Broj korisnika')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'movielens_small_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# 1. Distribucija filmova po godini izlaska
# Ekstrahiraj godinu iz 'release_date'
movies_small['release_year'] = pd.to_datetime(movies_small['release_date'], errors='coerce').dt.year
plt.figure(figsize=(12,6))
sns.histplot(movies_small['release_year'].dropna(), bins=40, kde=False)
plt.title('Distribucija filmova po godini izlaska')
plt.xlabel('Godina izlaska')
plt.ylabel('Broj filmova')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'movies_release_year_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Distribucija filmova po žanru (samo trend_genres)
# Žanrovi su zapisani kao string s listom dictova, pa ih treba parsirati
def extract_genres(genres_str):
    try:
        genres = ast.literal_eval(genres_str)
        return [g['name'] for g in genres if 'name' in g]
    except:
        return []
movies_small['genres_list'] = movies_small['genres'].apply(extract_genres)

trend_genres = [
    'Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime',
    'Adventure', 'Science Fiction', 'Mystery', 'Fantasy', 'Animation', 'Documentary', 'Family', 'Foreign'
]

all_genres = movies_small['genres_list'].explode()
genre_counts = Counter([g for g in all_genres.dropna() if g in trend_genres])
top_genres = pd.DataFrame.from_dict(genre_counts, orient='index').reset_index()
top_genres.columns = ['genre', 'count']
top_genres = top_genres.sort_values('count', ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(y='genre', x='count', data=top_genres)
plt.title('Distribucija filmova po žanru (trend žanrovi)')
plt.xlabel('Broj filmova')
plt.ylabel('Žanr')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'movies_genre_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Distribucija prosječne ocjene po žanru (samo trend_genres)
genre_ratings = []
for genre in top_genres['genre']:
    genre_movie_ids = movies_small[movies_small['genres_list'].apply(lambda x: genre in x if isinstance(x, list) else False)]['movieId']
    genre_ratings_data = movie_stats[movie_stats['movieId'].isin(genre_movie_ids)]
    avg_rating = genre_ratings_data['rating_mean'].mean()
    genre_ratings.append({'genre': genre, 'avg_rating': avg_rating, 'count': len(genre_movie_ids)})
genre_ratings_df = pd.DataFrame(genre_ratings).dropna().sort_values('avg_rating', ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(y='genre', x='avg_rating', data=genre_ratings_df)
plt.title('Prosječna ocjena po žanru (trend žanrovi)')
plt.xlabel('Prosječna ocjena')
plt.ylabel('Žanr')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'genre_avg_rating.png'), dpi=300, bbox_inches='tight')
plt.close()

# Analiza najpopularnijih žanrova (top 15, samo trend_genres)
gen_df = movies_small.explode('genres_list')
pop_gen = pd.DataFrame(gen_df['genres_list'].value_counts()).reset_index()
pop_gen.columns = ['genre', 'movies']
pop_gen = pop_gen[pop_gen['genre'].isin(trend_genres)]

plt.figure(figsize=(18,8))
sns.barplot(x='genre', y='movies', data=pop_gen.head(15))
plt.title('Top 15 najčešćih žanrova (trend žanrovi)')
plt.xlabel('Žanr')
plt.ylabel('Broj filmova')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'top15_genres.png'), dpi=300, bbox_inches='tight')
plt.close()

# Trendovi žanrova kroz godine (od 2000., top žanrovi, bez Documentary, Family, Foreign)
exclude_genres = []

trend_df = gen_df[
    (gen_df['release_year'] >= 2000) &
    (gen_df['genres_list'].isin(trend_genres)) &
    (~gen_df['genres_list'].isin(exclude_genres))
]

genre_year = trend_df.groupby(['release_year', 'genres_list']).size().reset_index(name='count')
pivot_genre_year = genre_year.pivot(index='release_year', columns='genres_list', values='count').fillna(0)

plt.figure(figsize=(18,8))
pivot_genre_year.plot(kind='area', stacked=True, figsize=(18,8), colormap='tab20')
plt.title('Trendovi žanrova kroz godine (od 2000.)')
plt.xlabel('Godina izlaska')
plt.ylabel('Broj filmova')
plt.legend(title='Žanr', loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'genre_trends_since_2000.png'), dpi=300, bbox_inches='tight')
plt.close()

# Distribucija ocjena
plt.figure(figsize=(8, 6))
sns.histplot(ratings_small['rating'], bins=[0.5,1.5,2.5,3.5,4.5,5.5], kde=False)
plt.title('Distribucija ocjena')
plt.xlabel('Ocjena')
plt.ylabel('Broj ocjena')
plt.xticks([1,2,3,4,5])
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'distribucija_ocjena.png'), dpi=300, bbox_inches='tight')
plt.close()

# Top 10 najviše ocjenjivanih filmova
plt.figure(figsize=(8, 6))
sns.barplot(y='title', x='rating_count', data=most_rated)
plt.title('Top 10 najviše ocjenjivanih filmova')
plt.xlabel('Broj ocjena')
plt.ylabel('Film')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'top10_najvise_ocjenjivanih.png'), dpi=300, bbox_inches='tight')
plt.close()

# Top 10 najbolje ocijenjenih filmova
plt.figure(figsize=(8, 6))
sns.barplot(y='title_with_rating', x='rating_mean', data=top_rated)
plt.title('Top 10 najbolje ocijenjenih filmova (min 5 ocjena)')
plt.xlabel('Prosječna ocjena')
plt.ylabel('Film')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'top10_najbolje_ocijenjenih.png'), dpi=300, bbox_inches='tight')
plt.close()

# Distribucija broja ocjena po korisniku
plt.figure(figsize=(8, 6))
user_counts = ratings_small['userId'].value_counts()
sns.histplot(user_counts, bins=30)
plt.title('Distribucija broja ocjena po korisniku')
plt.xlabel('Broj ocjena')
plt.ylabel('Broj korisnika')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'distribucija_ocjena_po_korisniku.png'), dpi=300, bbox_inches='tight')
plt.close()