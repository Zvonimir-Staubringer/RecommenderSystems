import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.makedirs("MovieLens", exist_ok=True)

ratings = pd.read_csv('MovieLens/ratings_small.csv')
links = pd.read_csv('MovieLens/links_small.csv')
metadata = pd.read_csv('MovieLens/movies_metadata.csv', low_memory=False)

# prepare mapping identical to other scripts
metadata = metadata[metadata['id'].apply(lambda x: str(x).isdigit())].copy()
metadata['id'] = metadata['id'].astype(int)
links = links[links['tmdbId'].notnull()].copy()
links['tmdbId'] = links['tmdbId'].astype(int)
movies_subset = metadata[metadata['id'].isin(links['tmdbId'])].copy()
movieid_to_title = movies_subset.set_index('id')['title'].dropna()

# filter ratings to subset
filtered = ratings[ratings['movieId'].isin(movieid_to_title.index)]

# reproducible split
train_df, test_df = train_test_split(filtered, test_size=0.2, random_state=42, stratify=None)

# save
train_df.to_csv('MovieLens/train_df.csv', index=False)
test_df.to_csv('MovieLens/test_df.csv', index=False)
# also save mapping for convenience
movieid_to_title.reset_index().to_csv('MovieLens/movieid_to_title.csv', index=False)

print("Saved train_df.csv, test_df.csv, movieid_to_title.csv")