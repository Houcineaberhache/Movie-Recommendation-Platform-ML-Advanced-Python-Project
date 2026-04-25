import sys
import os

# Ensure the local directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_engine import load_model, RecommenderEngine
import ml_engine
from cbf_engine import load_cbf_model
import cbf_engine

# CRITICAL FIX: Pickled objects remember they were created in "__main__". 
# To load them in another script, we must map them to their respective modules.
setattr(sys.modules['__main__'], 'SVDModel', ml_engine.SVDModel)
setattr(sys.modules['__main__'], 'ContentBasedRecommender', cbf_engine.ContentBasedRecommender)

print("="*60)
print(" [START] STARTING INTEGRATION TEST")
print("="*60)

# ---------------------------------------------------------
# Test 1: Collaborative Filtering (ML Engine)
# ---------------------------------------------------------
print("\n[1] Loading SVD Model (Collaborative Filtering)...")
try:
    svd = load_model("svd_model.pkl")
    cf_engine = RecommenderEngine()
    cf_engine.model = svd
    
    # Pick a random valid user from the vocabulary
    sample_user = next(iter(cf_engine.model.user2idx.keys()))
    
    print(f" -> Querying Top 5 Recommendations for User {sample_user}:")
    top_movies = cf_engine.get_top_n_recommendations(sample_user, n=5)
    for index, rec in enumerate(top_movies, 1):
        print(f"    {index}. Movie ID {rec['movieId']} | Predicted Score: {rec['score']:.2f}")
except Exception as e:
    print(f" ERROR in ML Engine: {e}")

# ---------------------------------------------------------
# Test 2: Content-Based Filtering (CBF Engine)
# ---------------------------------------------------------
print("\n[2] Loading CBF Model (Content-Based Filtering)...")
try:
    cbf_engine = load_cbf_model("cbf_model.pkl")
    
    test_movie_id = 1 # Toy Story
    movie_title = cbf_engine.movie_info.get(test_movie_id, {}).get("title", f"Movie #{test_movie_id}")
    
    print(f" -> Querying Top 5 Similar Movies to: '{movie_title}':")
    similar = cbf_engine.get_similar_movies(test_movie_id, n=5)
    for index, rec in enumerate(similar, 1):
        print(f"    {index}. {rec['title']} | Score: {rec['similarity_score']:.2f}")
except Exception as e:
    print(f" ERROR in CBF Engine: {e}")

print("\n" + "="*60)
print(" [SUCCESS] TEST COMPLETE! NO ERRORS FOUND.")
print("="*60 + "\n")
