import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("dataset/movies.csv")

# Fill missing values
movies['genres'] = movies['genres'].fillna('')

# Convert genres to numerical vectors
tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Calculate similarity
cosine_sim = cosine_similarity(tfidf_matrix)

# Create title-index mapping
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def recommend(movie_title):
    if movie_title not in indices:
        return ["Movie not found."]

    idx = indices[movie_title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:6]

    movie_indices = [i[0] for i in sim_scores]

    return movies['title'].iloc[movie_indices].tolist()

# Example
movie_name = input("Enter Movie Name: ")

matches = movies[movies['title'].str.contains(movie_name, case=False, na=False)]

if matches.empty:
    print("Movie not found.")
else:
    print("\nMatching Movies:")
    for i, title in enumerate(matches['title'].head(10), 1):
        print(f"{i}. {title}")

    selected_movie = matches['title'].iloc[0]

    recommendations = recommend(selected_movie)

    print("\nRecommended Movies:")
    for movie in recommendations:
        print(movie)

