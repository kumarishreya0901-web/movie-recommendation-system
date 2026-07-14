import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("dataset/movies.csv")

# Handle missing values
movies['genres'] = movies['genres'].fillna('')

genres = sorted(set(
    genre
    for row in movies['genres']
    for genre in row.split('|')
))

# TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix)

# Mapping
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Recommendation Function
def recommend(movie_title):
    if movie_title not in indices:
        return ["Movie not found"]

    idx = indices[movie_title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:6]

    movie_indices = [i[0] for i in sim_scores]

    recommended_movies = []

    for movie_idx, score in sim_scores:
        recommended_movies.append(
            (
                movies['title'].iloc[movie_idx],
                round(score * 100, 2)
            )
        )

    return recommended_movies

def get_local_poster(movie_title):

    poster_path = os.path.join(
        "posters",
        movie_title + ".jpg"
    )

    if os.path.exists(poster_path):
        return poster_path

    return "assets/default_poster.jpg"

st.sidebar.title("Navigation")

st.sidebar.info("""
AI Movie Recommendation System

Developed using:
- Python
- Streamlit
- Scikit-Learn
- MovieLens Dataset
""")
st.sidebar.subheader("Dataset Summary")
st.sidebar.write(f"Movies: {len(movies)}")
st.sidebar.write(f"Genres: {movies['genres'].nunique()}")

# Streamlit UI
st.title("🎬 AI-Based Movie Recommendation System")
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Home",
    "🎬 Recommend",
    "📊 Statistics",
    "ℹ️ About"
])

with tab1:

    st.markdown("""
    Welcome to the **AI-Based Movie Recommendation System**.

    This application recommends movies based on **Content-Based Filtering**
    using **TF-IDF Vectorization** and **Cosine Similarity**.

    ### Features
    - 🎭 Genre-wise movie selection
    - 🔍 Movie search
    - 🎬 Top 5 similar movie recommendations
    - 📊 Similarity percentage
    - 🖼️ Offline movie posters
    - 📥 Download recommendations as CSV
    """)

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🎬 Movies", len(movies))

    with col2:
        st.metric("🎭 Genres", len(genres))

    with col3:
        st.metric("🤖 Model", "TF-IDF")
    
    st.subheader("Dataset Information")

    st.write("Total Movies:", len(movies))
    st.write("Total Genres Available:", movies['genres'].nunique())

with tab2:
    with st.expander("About This Project"):
        st.write("""
        This AI-Based Movie Recommendation System uses
        TF-IDF Vectorization and Cosine Similarity
        to recommend movies based on genre similarity.
        """)

    st.subheader("🎬 Movie Recommendation")
    st.write("Choose a genre, search for a movie (optional), and click **Recommend**.")

    selected_genre = st.selectbox(
        "Select Genre",
        genres
    )
    search_movie = st.text_input(
        "Search Movie (Optional)"
    )
    filtered_movies = movies[
        movies['genres'].str.contains(selected_genre, case=False, na=False)
    ]

    if search_movie:
        filtered_movies = filtered_movies[
            filtered_movies['title'].str.contains(
                search_movie,
                case=False,
                na=False
            )
        ]

    movie_name = st.selectbox(
        "Select a Movie",
        sorted(filtered_movies['title'].tolist())
    )

    # Initialize recommendation history
    if "history" not in st.session_state:
        st.session_state.history = []
    if st.button("Recommend"):

        st.success(f"Selected Movie: {movie_name}")
        # Store recent searches
        if movie_name not in st.session_state.history:
            st.session_state.history.insert(0, movie_name)

        # Keep only the latest 5 searches
        st.session_state.history = st.session_state.history[:5]
        poster = get_local_poster(movie_name)

        st.image(
            poster,
            width=250
        )
        selected_genres = movies[movies['title'] == movie_name]['genres'].values[0]
        st.write("**Genres:**", selected_genres)

        recommendations = recommend(movie_name)

        # Create DataFrame for download
        download_list = []

        for movie, score in recommendations:
            genre = movies[movies["title"] == movie]["genres"].values[0]

            download_list.append({
                "Movie Title": movie,
                "Genres": genre,
                "Similarity Score (%)": score
            })

        download_data = pd.DataFrame(download_list)

        st.markdown("---")
        st.success(
            "Recommendations generated successfully!"
        )
        st.subheader(f"Top {len(recommendations)} Recommended Movies")
        for i, (movie, score) in enumerate(recommendations, start=1):

            with st.container(border=True):

                col1, col2 = st.columns([1, 3])

                with col1:
                    st.image(get_local_poster(movie), width=120)

                with col2:
                    st.markdown(f"### {i}. 🎬 {movie}")

                    genre = movies[movies['title'] == movie]['genres'].values[0]
                    st.write(f"**Genres:** {genre}")

                    st.progress(min(int(score), 100))

                    st.write(f"**Similarity Score:** {score}%")
        
        st.download_button(
            label="📥 Download Recommendations (CSV)",
            data=download_data.to_csv(index=False),
            file_name="recommended_movies.csv",
            mime="text/csv"
        )
        st.markdown("---")

        st.subheader("🕒 Recent Searches")

        if st.session_state.history:

            for movie in st.session_state.history:

                st.write("🎬", movie)

        else:

            st.info("No recent searches yet.")

with tab3:

    st.header("📊 Dataset Statistics")

    # Count genres
    genre_count = {}

    for row in movies["genres"]:
        for genre in row.split("|"):
            genre_count[genre] = genre_count.get(genre, 0) + 1

    # Convert to DataFrame
    genre_df = pd.DataFrame(
        genre_count.items(),
        columns=["Genre", "Count"]
    )

    genre_df = genre_df.sort_values(
        by="Count",
        ascending=False
    )

    st.subheader("Top 10 Most Common Genres")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(
        genre_df["Genre"][:10],
        genre_df["Count"][:10]
    )

    ax.set_xlabel("Genre")
    ax.set_ylabel("Number of Movies")
    ax.set_title("Top 10 Genres")

    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.dataframe(genre_df)

with tab4:

    st.header("ℹ️ About This Project")

    st.write("""
    ### AI-Based Movie Recommendation System

    This project recommends movies based on **genre similarity** using
    **Content-Based Filtering**.

    ### Technologies Used

    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - TF-IDF Vectorization
    - Cosine Similarity

    ### Dataset

    MovieLens Dataset

    ### Features

    - Genre-based filtering
    - Movie search
    - Top 5 recommendations
    - Offline poster support
    - CSV download
    - Interactive dashboard

    ### Developer

    **Shreya**

    ### Project

    College Mini Project
    """)
    
st.caption(
    f"Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
)

st.markdown("---")
st.caption("AI-Based Movie Recommendation System | Developed by Shreya")