"""
cbf_engine.py
=============
Phase 2 — Content-Based Filtering (CBF) engine.
Generates recommendations by finding movies with similar genres.
"""

import pickle, logging, os
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

class ContentBasedRecommender:
    def __init__(self):
        self.movie_idx_map = {} 
        self.idx_movie_map = {}
        self.movie_info = {} 
        self.feature_matrix = None 

    def load_and_train(self, movies_path: str):
        """
        Parses exactly the data Member 1 provided (movies.csv) 
        and computes a functional one-hot-encoded genre matrix.
        """
        logger.info("Loading Content Data: %s", movies_path)
        if not os.path.exists(movies_path):
             logger.error("Dataset not found! Expected %s", movies_path)
             return self
             
        df = pd.read_csv(movies_path)
        
        # 1. Build Vocabulary of Unique Genres
        all_genres = set()
        for genres in df['genres'].dropna():
            for g in genres.split('|'):
                all_genres.add(g)
        
        genre_list = sorted(list(all_genres))
        genre_to_idx = {g: i for i, g in enumerate(genre_list)}
        num_genres = len(genre_list)
        N = len(df)
        
        # 2. Build feature matrix using numpy (No Scikit-Learn dependencies required!)
        logger.info("Generating Neural Similarity Matrix for %d movies...", N)
        self.feature_matrix = np.zeros((N, num_genres), dtype=np.float32)
        
        for idx, row in df.iterrows():
            movie_id = row['movieId']
            self.movie_idx_map[movie_id] = idx
            self.idx_movie_map[idx] = movie_id
            self.movie_info[movie_id] = {"title": row['title'], "genres": row['genres']}
            
            if pd.notna(row['genres']) and row['genres'] != '(no genres listed)':
                for g in row['genres'].split('|'):
                    if g in genre_to_idx:
                        self.feature_matrix[idx, genre_to_idx[g]] = 1.0
                        
        # 3. L2 Normalize the matrix so we can use quick Dot-Products for Cosine Similarity
        norms = np.linalg.norm(self.feature_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0 # Prevent division by zero mathematically
        self.feature_matrix = self.feature_matrix / norms
        
        logger.info("Content Matrix Built Successfully. Ready for queries.")
        return self

    def get_similar_movies(self, movie_id: int, n: int = 10):
        """
        API Method: Fast lookup for N similar movies.
        Member 3 will use this for the "You might also like..." component.
        """
        if movie_id not in self.movie_idx_map:
            logger.warning("Movie %d not found in database.", movie_id)
            return []
            
        target_idx = self.movie_idx_map[movie_id]
        target_vec = self.feature_matrix[target_idx]
        
        # Fast Vectorized Cosine Similarity
        similarities = self.feature_matrix.dot(target_vec)
        
        # Sort and fetch best matches
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices:
            sim_score = similarities[idx]
            match_movie_id = self.idx_movie_map[idx]
            
            # Skip the exact same movie
            if match_movie_id == movie_id:
                continue
                
            # If there's 0% overlap in genres, stop predicting
            if sim_score <= 0.0:
                break 
                
            info = self.movie_info[match_movie_id]
            results.append({
                "movieId": match_movie_id,
                "title": info["title"],
                "genres": info["genres"],
                "similarity_score": round(float(sim_score), 4)
            })
            
            if len(results) >= n:
                break
                
        return results

    def export(self, output_path="cbf_model.pkl"):
        """Saves compressed model for Member 3's API"""
        with open(output_path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Exported Content Engine -> %s (%.1f MB)", output_path, os.path.getsize(output_path)/(1024*1024))
        return output_path

def load_cbf_model(pkl_path: str) -> ContentBasedRecommender:
    """Helper for Member 3 to reload engine swiftly"""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--movies", default="data/ml-25m/movies.csv")
    p.add_argument("--output", default="cbf_model.pkl")
    args = p.parse_args()
    
    engine = ContentBasedRecommender()
    engine.load_and_train(args.movies)
    
    print("\n" + "="*60)
    print(" [TEST] Testing CBF API Integration for Member 3...")
    test_id = 1 # Toy Story
    title = engine.movie_info.get(test_id, {}).get("title", 'Unknown')
    print(f"Target: {title} (ID: {test_id})")
    
    top_5 = engine.get_similar_movies(test_id, n=5)
    for i, res in enumerate(top_5, 1):
        print(f"  {i}. {res['title']:<38} | Score: {res['similarity_score']:.2f}")
    print("="*60 + "\n")
    
    engine.export(args.output)
    print("✅ CBF Build Complete.")
