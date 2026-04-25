"""
cbf_engine_simple.py
====================
Phase 2 – Content-Based Filtering (library version)
Uses: pandas, scikit-learn, pickle
"""

import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── 1. Load movies data ───────────────────────────────────────────────────────
def load_movies(movies_path: str):
    df = pd.read_csv(movies_path)
    # Replace the '|' separator with spaces so TF-IDF treats each genre as a word
    df["genres"] = df["genres"].str.replace("|", " ", regex=False)
    df = df.fillna("")
    print(f"Loaded {len(df):,} movies")
    return df


# ── 2. Build TF-IDF genre matrix ──────────────────────────────────────────────
def build_tfidf_matrix(df):
    # TF-IDF turns genre strings into weighted vectors automatically
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["genres"])
    print(f"TF-IDF matrix: {tfidf_matrix.shape}")  # (n_movies, n_genres)
    return vectorizer, tfidf_matrix


# ── 3. Get similar movies ─────────────────────────────────────────────────────
def get_similar_movies(movie_id: int, df, tfidf_matrix, n: int = 10):
    if movie_id not in df["movieId"].values:
        print(f"Movie {movie_id} not found.")
        return []

    # Find the row index for this movie
    idx = df.index[df["movieId"] == movie_id][0]

    # Compute cosine similarity between this movie and all others
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Sort by similarity (descending), skip the movie itself (score = 1.0)
    similar_indices = sim_scores.argsort()[::-1]

    results = []
    for i in similar_indices:
        if df.iloc[i]["movieId"] == movie_id:
            continue   # skip self
        if sim_scores[i] <= 0.0:
            break      # no genre overlap, stop
        results.append({
            "movieId":          int(df.iloc[i]["movieId"]),
            "title":            df.iloc[i]["title"],
            "genres":           df.iloc[i]["genres"],
            "similarity_score": round(float(sim_scores[i]), 4),
        })
        if len(results) >= n:
            break

    return results


# ── 4. Save model ─────────────────────────────────────────────────────────────
def export(vectorizer, tfidf_matrix, df, output_path: str = "cbf_model.pkl"):
    model = {
        "vectorizer":   vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "df":           df,
    }
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved -> {output_path}")


# ── 5. Load model (for Member 3) ─────────────────────────────────────────────
def load_cbf_model(pkl_path: str):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ── Run everything ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MOVIES_PATH = "movies.csv"
    OUTPUT_PATH = "cbf_model.pkl"

    # Step-by-step pipeline
    df                    = load_movies(MOVIES_PATH)
    vectorizer, tfidf_mtx = build_tfidf_matrix(df)

    # Quick API test — Toy Story (movieId = 1)
    print("\nTop 5 similar to Toy Story:")
    for res in get_similar_movies(1, df, tfidf_mtx, n=5):
        print(f"  {res['title']:<40} | score {res['similarity_score']:.2f}")

    export(vectorizer, tfidf_mtx, df, OUTPUT_PATH)
    print("\nDone")
