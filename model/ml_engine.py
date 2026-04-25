"""
ml_engine.py
============
Phase 2 — ML Engineering (numpy-only, no scikit-surprise)
"""

import os, pickle, logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

@dataclass
class SVDConfig:
    n_factors: int    = 50
    n_epochs: int     = 20
    lr: float         = 0.005
    reg: float        = 0.02
    test_size: float  = 0.20
    random_state: int = 42

class SVDModel:
    def __init__(self):
        self.P=None; self.Q=None; self.bu=None; self.bi=None
        self.mu=0.0
        self.user2idx={}; self.idx2user={}
        self.item2idx={}; self.idx2item={}
        self.user_history={} # Store user's watched movies so M3 doesn't need to load the dataframe

    def predict(self, user_id, movie_id) -> float:
        u = self.user2idx.get(user_id)
        i = self.item2idx.get(movie_id)
        if u is None or i is None:
            return float(self.mu)
        return float(self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i])

class RecommenderEngine:
    def __init__(self, config=None):
        self.config = config or SVDConfig()
        self.model  = SVDModel()
        self.rmse   = None
        self.accuracy = None
        self._is_trained  = False
        self._train_data  = None
        self._test_data   = None

    def load_data(self, ratings_path: str):
        logger.info("Loading: %s", ratings_path)
        df = pd.read_csv(ratings_path, usecols=["userId","movieId","rating"])
        logger.info("%d ratings | %d users | %d movies",
            len(df), df["userId"].nunique(), df["movieId"].nunique())
        users = sorted(df["userId"].unique())
        items = sorted(df["movieId"].unique())
        self.model.user2idx = {u:i for i,u in enumerate(users)}
        self.model.idx2user = {i:u for u,i in self.model.user2idx.items()}
        self.model.item2idx = {m:i for i,m in enumerate(items)}
        self.model.idx2item = {i:m for m,i in self.model.item2idx.items()}
        df["u"] = df["userId"].map(self.model.user2idx)
        df["i"] = df["movieId"].map(self.model.item2idx)
        np.random.seed(self.config.random_state)
        mask = np.random.rand(len(df)) < (1 - self.config.test_size)
        self._train_data = df[mask].reset_index(drop=True)
        self._test_data  = df[~mask].reset_index(drop=True)
        
        # Populate user history so Member 3 has access to it when the model is pickled
        self.model.user_history = self._train_data.groupby('userId')['movieId'].apply(set).to_dict()
        
        logger.info("Train: %d | Test: %d", len(self._train_data), len(self._test_data))
        return self

    def train(self):
        cfg = self.config
        n_users = len(self.model.user2idx)
        n_items = len(self.model.item2idx)
        np.random.seed(cfg.random_state)
        self.model.P  = np.random.normal(0, 0.1, (n_users, cfg.n_factors))
        self.model.Q  = np.random.normal(0, 0.1, (n_items, cfg.n_factors))
        self.model.bu = np.zeros(n_users)
        self.model.bi = np.zeros(n_items)
        self.model.mu = self._train_data["rating"].mean()
        logger.info("Training | factors=%d epochs=%d lr=%.4f reg=%.4f",
            cfg.n_factors, cfg.n_epochs, cfg.lr, cfg.reg)
        train = self._train_data
        n_samples = len(train)
        u_all = train["u"].values
        i_all = train["i"].values
        r_all = train["rating"].values
        batch_size = 100000

        for epoch in range(cfg.n_epochs):
            indices = np.random.permutation(n_samples)
            u_s = u_all[indices]
            i_s = i_all[indices]
            r_s = r_all[indices]

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                u = u_s[start_idx:end_idx]
                i = i_s[start_idx:end_idx]
                r = r_s[start_idx:end_idx]
                
                pred = self.model.mu + self.model.bu[u] + self.model.bi[i] + np.sum(self.model.P[u] * self.model.Q[i], axis=1)
                err = r - pred
                
                bu_grad = cfg.lr * (err - cfg.reg * self.model.bu[u])
                bi_grad = cfg.lr * (err - cfg.reg * self.model.bi[i])
                np.add.at(self.model.bu, u, bu_grad)
                np.add.at(self.model.bi, i, bi_grad)
                
                P_grad = cfg.lr * (err[:, None] * self.model.Q[i] - cfg.reg * self.model.P[u])
                Q_grad = cfg.lr * (err[:, None] * self.model.P[u] - cfg.reg * self.model.Q[i])
                np.add.at(self.model.P, u, P_grad)
                np.add.at(self.model.Q, i, Q_grad)

            if (epoch+1) % 5 == 0:
                logger.info("  Epoch %d/%d", epoch+1, cfg.n_epochs)
        self._is_trained = True
        logger.info("Training complete.")
        return self

    def evaluate(self, k: int = 10, threshold: float = 3.5) -> float:
        from collections import defaultdict
        logger.info("Evaluating RMSE, Precision@%d and Recall@%d...", k, k)
        
        preds = []
        actuals = []
        user_pred_true = defaultdict(list)
        
        # Populate prediction tracking
        for r in self._test_data.itertuples(index=False):
            pred = self.model.predict(r.userId, r.movieId)
            true_r = r.rating
            preds.append(pred)
            actuals.append(true_r)
            user_pred_true[r.userId].append({'pred': pred, 'true': true_r})
            
        preds = np.array(preds)
        actuals = np.array(actuals)
        self.rmse = float(np.sqrt(np.mean((preds-actuals)**2)))
        mae = float(np.mean(np.abs(preds-actuals)))
        self.accuracy = max(0.0, 100.0 * (1 - mae / 4.5))
        
        # Functional programming chunk to calculate Precision and Recall
        def calc_user_metrics(user_items):
            """Pure function to calculate metrics for a single user."""
            sorted_items = sorted(user_items, key=lambda x: x['pred'], reverse=True)
            n_rel = sum(1 for item in user_items if item['true'] >= threshold)
            n_rec_k = sum(1 for item in sorted_items[:k] if item['true'] >= threshold)
            
            p_k = n_rec_k / k if k > 0 else 0
            r_k = n_rec_k / n_rel if n_rel > 0 else 0
            return p_k, r_k

        # Apply pure function across all users using 'map' (FP paradigm)
        user_metrics = list(map(calc_user_metrics, user_pred_true.values()))

        self.precision_at_k = np.mean([m[0] for m in user_metrics])
        self.recall_at_k = np.mean([m[1] for m in user_metrics])
        
        logger.info("RMSE: %.4f | MAE: %.4f | Acc: %.1f%% | P@%d: %.4f | R@%d: %.4f", 
                    self.rmse, mae, self.accuracy, k, self.precision_at_k, k, self.recall_at_k)
        return self.rmse

    def get_top_n_recommendations(self, user_id: int, n: int = 10):
        """
        Generate Top-N recommendations. 
        M3: Import this module and call this function from your FastAPI endpoint!
        """
        if user_id not in self.model.user2idx:
            logger.warning("User %s not found in model vocabulary.", user_id)
            return []
            
        # Get seen movies directly from the pickled model, not the dataframe (M3 won't have the memory)
        user_seen = self.model.user_history.get(user_id, set())
        all_movies = set(self.model.item2idx.keys())
        
        # Functional programming elements (Filter and Map)
        unseen_movies = filter(lambda m: m not in user_seen, all_movies)
        
        # Returns an iterable of dicts with calculated predictions
        scored_movies = map(
            lambda m: {"movieId": m, "score": self.model.predict(user_id, m)}, 
            unseen_movies
        )
        
        # Sort functionally by score descending and take top N
        top_n = sorted(scored_movies, key=lambda x: x['score'], reverse=True)[:n]
        return top_n

    def export(self, output_path="svd_model.pkl") -> str:
        with open(output_path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("Exported → %s (%.1f KB)", output_path, os.path.getsize(output_path)/1024)
        return output_path

def load_model(pkl_path: str) -> SVDModel:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ratings", default="ratings_clean.csv")
    p.add_argument("--output",  default="svd_model.pkl")
    p.add_argument("--factors", type=int,   default=50)
    p.add_argument("--epochs",  type=int,   default=20)
    p.add_argument("--lr",      type=float, default=0.005)
    p.add_argument("--reg",     type=float, default=0.02)
    args = p.parse_args()
    cfg = SVDConfig(n_factors=args.factors, n_epochs=args.epochs, lr=args.lr, reg=args.reg)
    engine = RecommenderEngine(config=cfg)
    rmse = engine.load_data(args.ratings).train().evaluate()
    acc  = engine.accuracy
    p_k  = getattr(engine, "precision_at_k", 0.0)
    r_k  = getattr(engine, "recall_at_k", 0.0)
    
    print(f"\n{'='*45}")
    print(f"  Final Evaluation Metrics:")
    print(f"  RMSE       : {rmse:.4f}")
    print(f"  Accuracy   : {acc:.1f}%")
    print(f"  Prec@10    : {p_k:.4f}")
    print(f"  Recall@10  : {r_k:.4f}")
    print(f"{'='*45}\n")
    
    # Verification test for Member 3 Integration
    sample_user = next(iter(engine.model.user2idx.keys()))
    print(f"[TEST] API readiness (Top 5 items for random User ID {sample_user}):")
    top_5 = engine.get_top_n_recommendations(sample_user, n=5)
    for rec in top_5:
        print(f"    -> MovieID: {rec['movieId']:>6} | Predicted Rating: {rec['score']:.2f}/5.0")
    print("\n")
    
    engine.export(args.output)
    print(f"✅  Done -> {args.output}")