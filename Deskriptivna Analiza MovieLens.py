import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set plotting style and backend
plt.style.use('default')
plt.switch_backend('Agg')

# Read datasets
movies_df = pd.read_csv('MovieLens/movies_large.csv')
ratings_df = pd.read_csv('MovieLens/ratings_large.csv')

# Print basic dataset information
print("\n=== Dataset Characteristics ===")
print(f"Number of users: {ratings_df['userId'].nunique()}")
print(f"Number of movies: {ratings_df['movieId'].nunique()}")
print(f"Total ratings: {len(ratings_df)}")
print("\n=== Ratings Statistics ===")
print(ratings_df['rating'].describe())

# Make sure movies_df and ratings_df have the same movies
cleaned_movies = movies_df[movies_df['movieId'].isin(ratings_df['movieId'])]    
cleaned_ratings = ratings_df[ratings_df['movieId'].isin(cleaned_movies['movieId'])]
# Calculate movie statistics
movie_stats = cleaned_ratings.groupby('movieId').agg({
    'rating': ['count', 'mean']
}).reset_index()
movie_stats.columns = ['movieId', 'rating_count', 'rating_mean']

# Get most rated and top rated movies
most_rated = cleaned_ratings['movieId'].value_counts().head(10)
popular_movies = movie_stats[movie_stats['rating_count'] > 20].sort_values('rating_mean', ascending=False).head(10)

# Merge with movie titles
most_rated_movies = pd.merge(
    most_rated.reset_index(),
    cleaned_movies,
    on='movieId',
    how='left'
)
top_rated = pd.merge(
    popular_movies,
    cleaned_movies,
    on='movieId',
    how='left'
)

# Fill missing titles
most_rated_movies['title'] = most_rated_movies['title'].fillna('[Movie ID not found]')
top_rated['title'] = top_rated['title'].fillna('[Movie ID not found]')

# Create visualizations
plt.figure(figsize=(20, 15))

# 1. Rating Distribution
plt.subplot(2, 2, 1)
bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
sns.histplot(data=cleaned_ratings, x='rating', bins=bins)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.xticks([1, 2, 3, 4, 5])

# 2. Top 10 Most Rated Movies
plt.subplot(2, 2, 2)
top_10_most_rated = most_rated_movies.sort_values('count', ascending=False).head(10)
sns.barplot(y='title', x='count', data=top_10_most_rated)
plt.title('Top 10 Most Rated Movies')
plt.xlabel('Number of Ratings')

# 3. Top 10 Highest Rated Movies
plt.subplot(2, 2, 3)
top_10_rated = top_rated.sort_values('rating_mean', ascending=False).head(10)
sns.barplot(y='title', x='rating_mean', data=top_10_rated) 
plt.title('Top 10 Highest Rated Movies')
plt.xlabel('Average Rating')

# 4. Ratings per User Distribution
plt.subplot(2, 2, 4)
user_ratings_count = cleaned_ratings['userId'].value_counts()
sns.histplot(user_ratings_count, bins=50)
plt.title('Distribution of Ratings per User')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Users')

# Save main visualization
plt.tight_layout()
plt.savefig('movielens_analysis.png', dpi=300, bbox_inches='tight')
plt.close()