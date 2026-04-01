# Movie Recommendation Platform

> A full-stack recommendation engine built with MovieLens 25M, IMDb, and TMDB — exposing intelligent movie suggestions through a REST API, containerized with Docker and deployed on Google Cloud Platform.

---

## Table of contents

- [Project overview](#project-overview)
- [Team structure](#team-structure)
- [Architecture](#architecture)
- [Dataset sources](#dataset-sources)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Member 1 — Data pipeline](#member-1--data-pipeline)
- [Member 2 — Model training](#member-2--model-training)
- [Member 3 — API](#member-3--api)
- [Member 4 — Deployment](#member-4--deployment)
- [Tech stack](#tech-stack)
- [Deliverables](#deliverables)

---

## Project overview

This project was developed as part of a Master's-level Python course in Data Engineering and AI. The goal is to build a production-ready movie recommendation system that:

- Ingests and merges data from three real-world sources (MovieLens 25M, IMDb, TMDB API)
- Trains a hybrid recommendation engine combining collaborative filtering and content-based filtering
- Exposes predictions through a clean FastAPI REST interface
- Is fully containerized and deployable on GCP Cloud Run

---

## Team structure

| Member | Role | Responsibility |
|--------|------|----------------|
| Member 1 | Data Engineer | Data acquisition, cleaning, merging, feature engineering, export |
| Member 2 | ML Engineer | Model design, training, evaluation, Python module |
| Member 3 | Backend Developer | FastAPI endpoints, decorators, Swagger UI, testing |
| Member 4 | DevOps + Reporter | Docker, GCP deployment, Git management, report & presentation |

---

## Architecture

```
[MovieLens 25M] ─┐
[IMDb TSVs]     ─┼─► [Data Pipeline (M1)] ──► [Model Training (M2)] ──► [FastAPI (M3)] ──► [Docker + GCP (M4)]
[TMDB API]      ─┘
```

**Data flow:**
1. Raw data is acquired and cleaned by Member 1
2. A sparse user–item matrix and enriched movie metadata are exported as handoff files
3. Member 2 loads those files, trains the recommendation model, and packages it as a Python module
4. Member 3 wraps the module in a FastAPI app with REST endpoints
5. Member 4 containerizes the entire app and deploys it to GCP Cloud Run

---

## Dataset sources

| Source | What it provides | Access |
|--------|-----------------|--------|
| [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) | 25M ratings by 162K users across 62K movies | Free download |
| [IMDb Datasets](https://developer.imdb.com/non-commercial-datasets/) | Title basics, runtime, genres, IMDb scores | Free TSV download |
| [TMDB API](https://www.themoviedb.org/settings/api) | Movie overviews, posters, cast, TMDB scores | Free API key |

> **Note:** Register a free TMDB API key at https://www.themoviedb.org/settings/api and add it to your `.env` file before running the data pipeline.

---

## Repository structure

```
movie-recommendation-platform/
│
├── data/                        # Raw downloaded data (git-ignored)
│   ├── ml-25m/                  # MovieLens 25M files
│   └── imdb/                    # IMDb TSV files
│
├── outputs/                     # Processed files handed off between members
│   ├── ratings_clean.csv        # Cleaned ratings with re-indexed user/movie ids
│   ├── movies_enriched.csv      # Movie metadata + TF-IDF genre features
│   ├── user_item_matrix.npz     # Sparse interaction matrix (scipy CSR format)
│   ├── user_id_map.csv          # userId ↔ user_idx lookup table
│   └── movie_id_map.csv         # movieId ↔ movie_idx lookup table
│
├── notebooks/                   # Google Colab notebooks (Member 1)
│   └── 01_data_pipeline.ipynb   # Full data pipeline notebook
│
├── model/                       # Recommendation engine (Member 2)
│   ├── recommender.py           # Main model class
│   ├── train.py                 # Training script
│   └── evaluate.py              # Evaluation metrics
│
├── api/                         # FastAPI application (Member 3)
│   ├── main.py                  # App entry point
│   ├── routes/
│   │   └── recommend.py         # Recommendation endpoints
│   └── schemas.py               # Pydantic request/response schemas
│
├── Dockerfile                   # Container definition (Member 4)
├── docker-compose.yml           # Local multi-service orchestration
├── .env.example                 # Environment variable template
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Getting started

### Prerequisites

- Python 3.10+
- Docker Desktop (for local containerized run)
- A TMDB API key
- A Google Colab account (for Member 1's data pipeline)

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/movie-recommendation-platform.git
cd movie-recommendation-platform
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Then edit .env and fill in your TMDB_API_KEY
```

`.env.example`:
```
TMDB_API_KEY=your_key_here
MODEL_PATH=outputs/model.pkl
PORT=8000
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run with Docker (recommended)

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`.  
Swagger UI: `http://localhost:8000/docs`

---

## Member 1 — Data pipeline

**Notebook:** `notebooks/01_data_pipeline.ipynb`  
**Environment:** Google Colab

### Steps

| Phase | Description |
|-------|-------------|
| Phase 1 | Install libraries, download MovieLens 25M + IMDb TSVs, set up TMDB API |
| Phase 2 | Exploratory data analysis — shapes, nulls, distributions |
| Phase 3 | Clean ratings (dedup, cold-start filter) and movies (extract year, split genres) |
| Phase 4 | Merge MovieLens ↔ IMDb via `links.csv`, enrich with TMDB API metadata |
| Phase 5 | TF-IDF genre features, normalize IMDb scores, encode decade |
| Phase 6 | Build sparse user–item matrix, export all output files |
| Phase 7 | Assertions and quality checks before handoff to Member 2 |

### Output files for Member 2

```
outputs/
├── ratings_clean.csv        # All ratings, deduplicated, cold-start filtered
├── movies_enriched.csv      # Full movie metadata with engineered features
├── user_item_matrix.npz     # scipy CSR sparse matrix (users × movies)
├── user_id_map.csv          # Original userId → integer index
└── movie_id_map.csv         # Original movieId → integer index
```

### Cold-start thresholds applied

- Users with fewer than **20 ratings** are excluded
- Movies with fewer than **10 ratings** are excluded

---

## Member 2 — Model training

**Module:** `model/recommender.py`

### Algorithms

- **Collaborative Filtering** — matrix factorization on the sparse user–item matrix (SVD / NMF)
- **Content-Based Filtering** — cosine similarity on TF-IDF genre vectors from `movies_enriched.csv`
- **Hybrid** — weighted combination of both scores

### Evaluation metrics

| Metric | Target |
|--------|--------|
| RMSE | < 1.0 |
| Precision@10 | > 0.15 |
| Recall@10 | > 0.10 |

### Usage

```python
from model.recommender import MovieRecommender

rec = MovieRecommender(matrix_path="outputs/user_item_matrix.npz",
                       movies_path="outputs/movies_enriched.csv")
rec.train()
recommendations = rec.recommend(user_id=42, top_k=10)
```

---

## Member 3 — API

**Framework:** FastAPI  
**Entry point:** `api/main.py`

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/recommend` | Get top-K recommendations for a user |
| `GET` | `/movie/{movie_id}` | Get metadata for a specific movie |
| `GET` | `/docs` | Swagger UI |

### Example request

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 42, "top_k": 10}'
```

### Example response

```json
{
  "user_id": 42,
  "recommendations": [
    {"movie_id": 318, "title": "Shawshank Redemption", "score": 0.94},
    {"movie_id": 858, "title": "Godfather, The", "score": 0.91}
  ]
}
```

---

## Member 4 — Deployment

### Local Docker build

```bash
docker build -t movie-rec-api .
docker run -p 8000:8000 --env-file .env movie-rec-api
```

### GCP Cloud Run deployment

```bash
# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build and push image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/movie-rec-api

# Deploy
gcloud run deploy movie-rec-api \
  --image gcr.io/YOUR_PROJECT_ID/movie-rec-api \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated
```

---

## Tech stack

| Layer | Tools |
|-------|-------|
| Data | pandas, numpy, scipy, scikit-learn, requests, TMDB API |
| Model | scikit-learn (SVD/NMF), cosine similarity, MLOps patterns |
| API | FastAPI, Pydantic, Uvicorn |
| Infra | Docker, docker-compose, GCP Cloud Run, Git/GitHub |
| Dev | Google Colab, VS Code, Postman, Swagger UI |

---

## Deliverables

- [ ] `notebooks/01_data_pipeline.ipynb` — complete and validated (Member 1)
- [ ] `outputs/` — all export files passing quality assertions (Member 1)
- [ ] `model/recommender.py` — trained and evaluated (Member 2)
- [ ] `api/` — FastAPI app with all endpoints tested (Member 3)
- [ ] `Dockerfile` + GCP deployment URL (Member 4)
- [ ] Final report (PDF)
- [ ] Presentation slides

---

## License

This project is developed for academic purposes as part of a Master's program in Data Engineering and AI.
