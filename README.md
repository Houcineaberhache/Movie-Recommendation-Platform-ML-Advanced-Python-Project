# 🎬 Movie Recommendation System

A machine learning API that recommends movies based on user preferences using Collaborative Filtering and Content-Based Filtering.

## Team
- M1 — Data Engineer
- M2 — ML Engineer
- M3 — Backend Developer
- M4 — DevOps & Reporter

## Project Structure
```
movie-recommender/
├── data/                     ← dataset files (not pushed to GitHub, too large)
├── model/
│   ├── ml_engine.py          ← collaborative filtering engine (M2)
│   └── cbf_engine.py         ← content-based filtering engine (M2)
├── api/
│   ├── main.py               ← FastAPI app (M3)
│   └── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
└── README.md
```

## Note on large files
The following files are not pushed to GitHub due to size but are included in the Docker container:
- `data/movies.csv`
- `data/ratings_clean.csv`
- `model/svd_model.pkl`
- `model/cbf_model.pkl`

## How to Run Locally

1. Clone this repo
```bash
git clone https://github.com/yourname/movie-recommender.git
cd movie-recommender
```

2. Build and run with Docker
```bash
docker build -t movie-recommender .
docker run -p 8080:8080 movie-recommender
```

3. Open in browser
```
http://localhost:8080/docs
```

## Live API
https://abdessamadf-movie-recommender.hf.space/
https://your-cloud-run-link-here/docs

## Dataset
MovieLens dataset — cleaned and preprocessed by M1.
