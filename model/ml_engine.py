import pickle
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np


# ── 1. Load & prepare data ──────────────────────────────────────────────────
def load_data(ratings_path: str, test_size: float = 0.2):
    df = pd.read_csv(ratings_path, usecols=["userId", "movieId", "rating"])
    print(f"Loaded {len(df):,} ratings | {df['userId'].nunique():,} users | {df['movieId'].nunique():,} movies")

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")
    return train_df, test_df


# ── 2. Build user-item matrix ────────────────────────────────────────────────
def build_matrix(train_df):
    # Pivot: rows = users, columns = movies, values = ratings (0 if unseen)
    matrix = train_df.pivot_table(
        index="userId", columns="movieId", values="rating", fill_value=0
    )
    print(f"Matrix shape: {matrix.shape}")
    return matrix


# ── 3. Train SVD model ───────────────────────────────────────────────────────
def train_svd(matrix, n_components: int = 50):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    # Decompose the matrix into latent factors
    user_factors = svd.fit_transform(matrix)
    item_factors = svd.components_   # shape: (n_components, n_movies)

    print(f"SVD trained | components={n_components} | explained variance: {svd.explained_variance_ratio_.sum():.2%}")
    return svd, user_factors, item_factors


# ── 4. Reconstruct & evaluate ─────────────────────────────────────────────────
def evaluate(matrix, user_factors, item_factors, test_df):
    # Reconstruct the full ratings matrix from latent factors
    reconstructed = user_factors @ item_factors   # matrix multiplication

    # Map user/movie IDs back to matrix indices
    user_index = {uid: i for i, uid in enumerate(matrix.index)}
    movie_index = {mid: i for i, mid in enumerate(matrix.columns)}

    preds, actuals = [], []
    for row in test_df.itertuples(index=False):
        u = user_index.get(row.userId)
        m = movie_index.get(row.movieId)
        if u is not None and m is not None:
            preds.append(reconstructed[u, m])
            actuals.append(row.rating)

    rmse = root_mean_squared_error(actuals, preds)
    mae  = np.mean(np.abs(np.array(preds) - np.array(actuals)))
    acc  = max(0.0, 100.0 * (1 - mae / 4.5))
    print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | Accuracy: {acc:.1f}%")
    return rmse


# ── 5. Get top-N recommendations for a user ──────────────────────────────────
def get_recommendations(user_id: int, matrix, user_factors, item_factors, n: int = 10):
    if user_id not in matrix.index:
        print(f"User {user_id} not found.")
        return []

    user_idx = list(matrix.index).index(user_id)

    # Predict scores for all movies
    scores = user_factors[user_idx] @ item_factors   # dot product

    # Build result — skip movies the user already rated
    seen = set(matrix.columns[matrix.loc[user_id] > 0])
    results = [
        {"movieId": mid, "score": round(float(scores[i]), 4)}
        for i, mid in enumerate(matrix.columns)
        if mid not in seen
    ]

    # Sort by predicted score descending, return top N
    top_n = sorted(results, key=lambda x: x["score"], reverse=True)[:n]
    return top_n


# ── 6. Save model ─────────────────────────────────────────────────────────────
def export(svd, matrix, user_factors, item_factors, output_path: str = "svd_model.pkl"):
    model = {
        "svd":          svd,
        "matrix":       matrix,
        "user_factors": user_factors,
        "item_factors": item_factors,
    }
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved -> {output_path}")


# ── 7. Load model (for Member 3) ─────────────────────────────────────────────
def load_model(pkl_path: str):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ── Run everything ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RATINGS_PATH = "ratings_clean.csv"
    OUTPUT_PATH  = "svd_model.pkl"
    N_COMPONENTS = 50

    # Step-by-step pipeline
    train_df, test_df  = load_data(RATINGS_PATH)
    matrix             = build_matrix(train_df)
    svd, U, V          = train_svd(matrix, n_components=N_COMPONENTS)
    evaluate(matrix, U, V, test_df)

    # Quick API test
    sample_user = matrix.index[0]
    print(f"\nTop 5 for user {sample_user}:")
    for rec in get_recommendations(sample_user, matrix, U, V, n=5):
        print(f"  movieId {rec['movieId']:>6} | score {rec['score']:.2f}")

    export(svd, matrix, U, V, OUTPUT_PATH)
    print("\nDone")