"""
K-Nearest Neighbors Example: Movie Recommendation System

This example demonstrates how to use K-Nearest Neighbors for a movie recommendation system.
The example shows:

1. Loading and preprocessing a movie ratings dataset
2. Building a user-based recommendation system using KNN
3. Finding similar users based on rating patterns
4. Recommending movies for a given user
5. Evaluating recommendation quality
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import random
import time

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.supervised.knn import KNNModel


def create_synthetic_movie_data(n_users=100, n_movies=50, sparsity=0.8):
    """
    Create a synthetic movie ratings dataset.
    
    Args:
        n_users: Number of users
        n_movies: Number of movies
        sparsity: Proportion of user-movie pairs that have no rating (0.0-1.0)
        
    Returns:
        DataFrame with user_id, movie_id, rating, and timestamp columns
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create movie genres and titles
    genres = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Thriller', 'Romance', 'Horror', 'Adventure', 'Animation', 'Fantasy']
    
    movies = []
    for i in range(1, n_movies + 1):
        # Assign 1-3 genres to each movie
        n_genres = np.random.randint(1, 4)
        movie_genres = np.random.choice(genres, size=n_genres, replace=False)
        genre_str = '|'.join(movie_genres)
        
        # Create movie title based on genres
        adjectives = ['Amazing', 'Incredible', 'Fantastic', 'Mysterious', 'Deadly', 'Epic', 'Funny', 'Dark', 'Scary', 'Beautiful']
        nouns = ['Journey', 'Adventure', 'Story', 'Tale', 'Mission', 'Quest', 'Day', 'Night', 'Love', 'Game']
        
        title_adj = np.random.choice(adjectives)
        title_noun = np.random.choice(nouns)
        title = f"The {title_adj} {title_noun}"
        
        if 'Sci-Fi' in movie_genres:
            title += " in Space"
        elif 'Romance' in movie_genres:
            title += " of Love"
        elif 'Horror' in movie_genres:
            title += " of Doom"
        
        # Store movie info
        movies.append({
            'movie_id': i,
            'title': title,
            'genres': genre_str,
            'year': np.random.randint(1990, 2022)
        })
    
    movies_df = pd.DataFrame(movies)
    
    # Create user-movie ratings
    ratings = []
    
    # For each user
    for user_id in range(1, n_users + 1):
        # Determine user's genre preferences (some genres they rate higher)
        preferred_genres = np.random.choice(genres, size=3, replace=False)
        
        # Determine which movies this user will rate
        n_ratings = int(n_movies * (1 - sparsity))
        movies_to_rate = np.random.choice(n_movies, size=n_ratings, replace=False) + 1
        
        for movie_id in movies_to_rate:
            # Get movie genres
            movie_genres = movies_df[movies_df['movie_id'] == movie_id]['genres'].values[0].split('|')
            
            # Base rating - random between 2 and 4
            rating = np.random.uniform(2, 4)
            
            # If the movie has one of the user's preferred genres, likely rate it higher
            for genre in preferred_genres:
                if genre in movie_genres and np.random.random() < 0.7:
                    rating += np.random.uniform(0, 1)  # Add between 0 and 1 stars
            
            # If user strongly dislikes a genre in the movie, likely rate it lower
            disliked_genres = [g for g in genres if g not in preferred_genres][:2]
            for genre in disliked_genres:
                if genre in movie_genres and np.random.random() < 0.5:
                    rating -= np.random.uniform(0, 1)  # Subtract between 0 and 1 stars
            
            # Ensure rating is between 0.5 and 5, round to nearest 0.5
            rating = max(0.5, min(5, rating))
            rating = round(rating * 2) / 2
            
            # Create timestamp (random date in the last year)
            days_ago = np.random.randint(0, 365)
            timestamp = int(time.time() - days_ago * 86400)
            
            ratings.append({
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': rating,
                'timestamp': timestamp
            })
    
    ratings_df = pd.DataFrame(ratings)
    
    # Create user info
    users = []
    for user_id in range(1, n_users + 1):
        age = np.random.randint(18, 65)
        gender = np.random.choice(['M', 'F'])
        occupation = np.random.choice(['student', 'engineer', 'artist', 'doctor', 'teacher', 'retired', 'other'])
        users.append({
            'user_id': user_id,
            'age': age,
            'gender': gender,
            'occupation': occupation
        })
    
    users_df = pd.DataFrame(users)
    
    return ratings_df, movies_df, users_df


def create_user_item_matrix(ratings_df, n_users=None, n_movies=None):
    """
    Create a user-item matrix from the ratings DataFrame.
    
    Args:
        ratings_df: DataFrame with user_id, movie_id, and rating columns
        n_users: Maximum user_id (if None, inferred from data)
        n_movies: Maximum movie_id (if None, inferred from data)
        
    Returns:
        user_item_matrix: Matrix where rows are users, columns are movies, and values are ratings
        user_means: Mean rating for each user (for centering)
    """
    if n_users is None:
        n_users = ratings_df['user_id'].max()
    
    if n_movies is None:
        n_movies = ratings_df['movie_id'].max()
    
    # Create an empty matrix
    user_item_matrix = np.zeros((n_users, n_movies))
    
    # Fill the matrix with ratings
    for _, row in ratings_df.iterrows():
        user_idx = int(row['user_id']) - 1  # Convert to 0-based index
        movie_idx = int(row['movie_id']) - 1  # Convert to 0-based index
        user_item_matrix[user_idx, movie_idx] = row['rating']
    
    # Compute user means (ignoring zeros)
    user_means = np.zeros(n_users)
    for i in range(n_users):
        user_ratings = user_item_matrix[i]
        mask = user_ratings > 0  # Only consider rated movies
        if np.sum(mask) > 0:
            user_means[i] = np.mean(user_ratings[mask])
    
    return user_item_matrix, user_means


def normalize_user_item_matrix(user_item_matrix, user_means):
    """
    Normalize the user-item matrix by subtracting the user mean from each rating.
    
    Args:
        user_item_matrix: Matrix where rows are users, columns are movies, and values are ratings
        user_means: Mean rating for each user
        
    Returns:
        normalized_matrix: Normalized user-item matrix
    """
    normalized_matrix = user_item_matrix.copy()
    
    for i in range(user_item_matrix.shape[0]):
        mask = normalized_matrix[i] > 0  # Only normalize rated items
        normalized_matrix[i, mask] = normalized_matrix[i, mask] - user_means[i]
    
    return normalized_matrix


def recommend_movies(knn_model, user_id, user_item_matrix, user_means, movies_df, n_recommendations=5):
    """
    Recommend movies for a user based on KNN.
    
    Args:
        knn_model: Trained KNN model
        user_id: ID of the user to recommend for (1-based)
        user_item_matrix: User-item matrix
        user_means: Mean rating for each user
        movies_df: DataFrame with movie information
        n_recommendations: Number of recommendations to make
        
    Returns:
        recommendations: DataFrame with recommended movies
    """
    # Convert to 0-based index
    user_idx = user_id - 1
    
    # Get the user's vector
    user_vector = user_item_matrix[user_idx]
    
    # Find movies the user hasn't rated
    unrated_mask = user_vector == 0
    
    if np.sum(~unrated_mask) == 0:
        print(f"User {user_id} hasn't rated any movies yet.")
        return pd.DataFrame()
    
    # Skip if the user has rated all movies (unlikely in a real system)
    if np.sum(unrated_mask) == 0:
        print(f"User {user_id} has already rated all movies.")
        return pd.DataFrame()
    
    # Find similar users
    similar_user_indices = knn_model.get_neighbors(
        user_item_matrix[user_idx].reshape(1, -1)
    )[0]
    
    # Predict ratings for unrated movies
    predicted_ratings = np.zeros(user_item_matrix.shape[1])
    
    for movie_idx in range(user_item_matrix.shape[1]):
        if unrated_mask[movie_idx]:  # Only predict for unrated movies
            # Get ratings for this movie from similar users
            neighbor_ratings = user_item_matrix[similar_user_indices, movie_idx]
            
            # Calculate weighted average based on similarity (simplification: use uniform weights)
            # In a real system, you'd weigh by similarity
            if np.sum(neighbor_ratings > 0) > 0:  # If at least one neighbor rated this movie
                # Get only the ratings > 0 (movies that were rated)
                valid_ratings = neighbor_ratings[neighbor_ratings > 0]
                valid_user_indices = similar_user_indices[neighbor_ratings > 0]
                
                # Get the normalized ratings
                valid_normalized_ratings = valid_ratings - user_means[valid_user_indices]
                
                # Predict rating as user's mean plus weighted average of normalized ratings
                predicted_ratings[movie_idx] = user_means[user_idx] + np.mean(valid_normalized_ratings)
    
    # Get top n recommendations
    unrated_movie_indices = np.where(unrated_mask)[0]
    unrated_predicted_ratings = predicted_ratings[unrated_mask]
    
    # Sort by predicted rating (descending)
    sorted_indices = np.argsort(-unrated_predicted_ratings)
    
    # Get top n movie indices
    top_movie_indices = unrated_movie_indices[sorted_indices[:n_recommendations]]
    
    # Convert to 1-based movie_id
    top_movie_ids = top_movie_indices + 1
    
    # Get movie information
    recommendations = movies_df[movies_df['movie_id'].isin(top_movie_ids)].copy()
    
    # Add predicted rating
    recommendations['predicted_rating'] = [predicted_ratings[idx] for idx in top_movie_indices]
    
    # Sort by predicted rating
    recommendations = recommendations.sort_values('predicted_rating', ascending=False)
    
    return recommendations


def evaluate_recommendations(knn_model, user_item_matrix, user_means, ratings_test):
    """
    Evaluate the recommendation system using RMSE.
    
    Args:
        knn_model: Trained KNN model
        user_item_matrix: User-item matrix
        user_means: Mean rating for each user
        ratings_test: Test set of ratings
        
    Returns:
        rmse: Root mean squared error of predictions
    """
    predictions = []
    actuals = []
    
    for _, row in ratings_test.iterrows():
        user_idx = int(row['user_id']) - 1
        movie_idx = int(row['movie_id']) - 1
        actual_rating = row['rating']
        
        # Skip if user or movie isn't in the training data
        if user_idx >= user_item_matrix.shape[0] or movie_idx >= user_item_matrix.shape[1]:
            continue
        
        # Get similar users
        similar_user_indices = knn_model.get_neighbors(
            user_item_matrix[user_idx].reshape(1, -1)
        )[0]
        
        # Get ratings for this movie from similar users
        neighbor_ratings = user_item_matrix[similar_user_indices, movie_idx]
        
        # If none of the neighbors rated this movie, use the user's mean
        if np.sum(neighbor_ratings > 0) == 0:
            predicted_rating = user_means[user_idx]
        else:
            # Get only the ratings > 0 (movies that were rated)
            valid_ratings = neighbor_ratings[neighbor_ratings > 0]
            valid_user_indices = similar_user_indices[neighbor_ratings > 0]
            
            # Get the normalized ratings
            valid_normalized_ratings = valid_ratings - user_means[valid_user_indices]
            
            # Predict rating as user's mean plus weighted average of normalized ratings
            predicted_rating = user_means[user_idx] + np.mean(valid_normalized_ratings)
        
        predictions.append(predicted_rating)
        actuals.append(actual_rating)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    return rmse, predictions, actuals


def main():
    print("=" * 80)
    print("K-Nearest Neighbors Example: Movie Recommendation System")
    print("=" * 80)
    
    # 1. Load data or create synthetic data
    print("\n1. Loading movie ratings dataset...")
    ratings_df, movies_df, users_df = create_synthetic_movie_data(
        n_users=100, n_movies=50, sparsity=0.7
    )
    
    print(f"Created synthetic dataset with {len(ratings_df)} ratings from {ratings_df['user_id'].nunique()} users for {ratings_df['movie_id'].nunique()} movies")
    
    # 2. Explore the data
    print("\n2. Exploring the dataset...")
    
    print("\nMovies sample:")
    print(movies_df.head())
    
    print("\nRatings sample:")
    print(ratings_df.head())
    
    print("\nRating distribution:")
    plt.figure(figsize=(8, 5))
    sns.countplot(x='rating', data=ratings_df)
    plt.title('Rating Distribution')
    plt.show()
    
    print("\nRatings per user:")
    ratings_per_user = ratings_df.groupby('user_id').size()
    plt.figure(figsize=(10, 5))
    plt.hist(ratings_per_user, bins=20)
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    plt.title('Ratings per User')
    plt.show()
    
    # 3. Create user-item matrix
    print("\n3. Creating user-item matrix...")
    n_users = ratings_df['user_id'].max()
    n_movies = ratings_df['movie_id'].max()
    
    user_item_matrix, user_means = create_user_item_matrix(ratings_df, n_users, n_movies)
    
    print(f"User-item matrix shape: {user_item_matrix.shape}")
    print(f"Sparsity: {np.sum(user_item_matrix == 0) / (user_item_matrix.shape[0] * user_item_matrix.shape[1]):.2%}")
    
    # Visualize the user-item matrix (sample)
    max_users_to_show = 10
    max_movies_to_show = 20
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(user_item_matrix[:max_users_to_show, :max_movies_to_show], 
               annot=True, cmap='YlGnBu', fmt='.1f', cbar_kws={'label': 'Rating'})
    plt.xlabel('Movie ID')
    plt.ylabel('User ID')
    plt.title('User-Item Matrix (Sample)')
    plt.show()
    
    # 4. Split the data into train and test sets
    print("\n4. Splitting data into training and testing sets...")
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(train_df)}")
    print(f"Testing set size: {len(test_df)}")
    
    # Create training user-item matrix
    train_matrix, train_user_means = create_user_item_matrix(train_df, n_users, n_movies)
    
    # 5. Normalize the user-item matrix
    print("\n5. Normalizing the user-item matrix...")
    normalized_matrix = normalize_user_item_matrix(train_matrix, train_user_means)
    
    # 6. Train KNN model
    print("\n6. Training the KNN model...")
    knn_model = KNNModel(
        task_type="regression",  # We'll use regression to predict ratings
        n_neighbors=5,           # Number of similar users to consider
        weights='uniform',       # Equal weight for all neighbors (could use 'distance')
        metric='cosine'          # Use cosine similarity for sparse data
    )
    
    # Time the training process
    start_time = time.time()
    knn_model.train(normalized_matrix, np.zeros(n_users))  # Target is not used for recommendation
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # 7. Make recommendations for a sample user
    print("\n7. Generating movie recommendations for sample users...")
    
    # Select 3 random users with at least 5 ratings
    users_with_many_ratings = ratings_df.groupby('user_id').size()
    users_with_many_ratings = users_with_many_ratings[users_with_many_ratings >= 5].index.tolist()
    
    if len(users_with_many_ratings) > 3:
        sample_user_ids = random.sample(users_with_many_ratings, 3)
    else:
        sample_user_ids = users_with_many_ratings
    
    for user_id in sample_user_ids:
        print(f"\nRecommendations for User {user_id}:")
        
        # Get the user's rated movies
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        print(f"User has rated {len(user_ratings)} movies:")
        
        # Display the user's top rated movies
        top_rated = user_ratings.sort_values('rating', ascending=False).head(3)
        for _, row in top_rated.iterrows():
            movie_info = movies_df[movies_df['movie_id'] == row['movie_id']].iloc[0]
            print(f"  - {movie_info['title']} ({movie_info['year']}): {row['rating']} stars")
        
        # Generate recommendations
        recommendations = recommend_movies(
            knn_model, user_id, train_matrix, train_user_means, movies_df
        )
        
        print("\nRecommended movies:")
        for _, movie in recommendations.iterrows():
            print(f"  - {movie['title']} ({movie['year']}): Predicted rating: {movie['predicted_rating']:.2f} stars")
    
    # 8. Evaluate the recommendation system
    print("\n8. Evaluating the recommendation system...")
    rmse, predictions, actuals = evaluate_recommendations(
        knn_model, train_matrix, train_user_means, test_df
    )
    
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    # Plot predicted vs actual ratings
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('Actual vs Predicted Ratings')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    
    # 9. Try different values of k and find the optimal value
    print("\n9. Finding the optimal number of neighbors (k)...")
    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    rmse_values = []
    
    for k in k_values:
        # Create and train KNN model
        temp_knn_model = KNNModel(
            task_type="regression",
            n_neighbors=k,
            weights='uniform',
            metric='cosine'
        )
        temp_knn_model.train(normalized_matrix, np.zeros(n_users))
        
        # Evaluate
        rmse, _, _ = evaluate_recommendations(
            temp_knn_model, train_matrix, train_user_means, test_df
        )
        rmse_values.append(rmse)
        print(f"k={k}: RMSE={rmse:.4f}")
    
    # Plot RMSE vs k
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, rmse_values, 'o-')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Number of Neighbors')
    plt.grid(True)
    plt.show()
    
    # Find the best k
    best_k = k_values[np.argmin(rmse_values)]
    print(f"\nBest number of neighbors (k): {best_k} with RMSE: {min(rmse_values):.4f}")
    
    # 10. Visualize the nearest neighbors for a random user
    print("\n10. Visualizing nearest neighbors in user space...")
    
    # Perform dimensionality reduction for visualization
    from sklearn.decomposition import PCA
    
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    user_matrix_2d = pca.fit_transform(normalized_matrix)
    
    # Select a random user
    random_user_id = random.choice(sample_user_ids)
    random_user_idx = random_user_id - 1
    
    # Create and train KNN model with the best k
    best_knn_model = KNNModel(
        task_type="regression",
        n_neighbors=best_k,
        weights='uniform',
        metric='cosine'
    )
    best_knn_model.train(normalized_matrix, np.zeros(n_users))
    
    # Find nearest neighbors
    neighbor_indices = best_knn_model.get_neighbors(
        normalized_matrix[random_user_idx].reshape(1, -1)
    )[0]
    
    # Visualize users in 2D space
    plt.figure(figsize=(10, 8))
    
    # Plot all users
    plt.scatter(user_matrix_2d[:, 0], user_matrix_2d[:, 1], alpha=0.5, label='All Users')
    
    # Highlight the selected user
    plt.scatter(user_matrix_2d[random_user_idx, 0], user_matrix_2d[random_user_idx, 1], 
               color='red', s=100, label=f'User {random_user_id}')
    
    # Highlight nearest neighbors
    plt.scatter(user_matrix_2d[neighbor_indices, 0], user_matrix_2d[neighbor_indices, 1], 
               color='green', s=50, label=f'{best_k} Nearest Neighbors')
    
    plt.title('User Space Visualization (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 11. Make final recommendations using the best model
    print("\n11. Making final recommendations using the optimal model...")
    
    # Pick a user from the sample users
    final_user_id = random.choice(sample_user_ids)
    
    print(f"\nFinal recommendations for User {final_user_id}:")
    
    # Get the user's rated movies
    user_ratings = ratings_df[ratings_df['user_id'] == final_user_id]
    print(f"User has rated {len(user_ratings)} movies:")
    
    # Display the user's top rated movies
    top_rated = user_ratings.sort_values('rating', ascending=False).head(3)
    for _, row in top_rated.iterrows():
        movie_info = movies_df[movies_df['movie_id'] == row['movie_id']].iloc[0]
        print(f"  - {movie_info['title']} ({movie_info['year']}) ({movie_info['genres']}): {row['rating']} stars")
    
    # Generate recommendations using the best model
    recommendations = recommend_movies(
        best_knn_model, final_user_id, train_matrix, train_user_means, movies_df, n_recommendations=10
    )
    
    print("\nRecommended movies:")
    for _, movie in recommendations.iterrows():
        print(f"  - {movie['title']} ({movie['year']}) ({movie['genres']}): Predicted rating: {movie['predicted_rating']:.2f} stars")
    
    # Save the best model
    print("\n12. Saving the best model...")
    model_path = "../models/knn_movie_recommender.joblib"
    best_knn_model.save_model(model_path)
    
    print("\nExample complete! The KNN model has been trained for movie recommendations.")


if __name__ == "__main__":
    main() 