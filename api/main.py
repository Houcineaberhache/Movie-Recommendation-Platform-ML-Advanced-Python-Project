"""
main.py
=======
Member 3 — Backend/API Developer
FastAPI wrapper around M2's ML engines (SVD + CBF).
Includes a built-in HTML UI for user-friendly interaction.
"""

import os
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import ml_engine
import cbf_engine
import sys
sys.modules['__main__'].SVDModel = ml_engine.SVDModel
sys.modules['__main__'].ContentBasedRecommender = cbf_engine.ContentBasedRecommender

from ml_engine import SVDModel, load_model
from cbf_engine import ContentBasedRecommender, load_cbf_model

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Movie Recommender API",
    description="Hybrid recommendation system combining SVD Collaborative Filtering and Content-Based Filtering.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SVD_MODEL_PATH = os.getenv("SVD_MODEL_PATH", "svd_model.pkl")
CBF_MODEL_PATH = os.getenv("CBF_MODEL_PATH", "cbf_model.pkl")

svd_model: Optional[SVDModel] = None
cbf_model: Optional[ContentBasedRecommender] = None


@app.on_event("startup")
def load_models():
    global svd_model, cbf_model
    try:
        import ml_engine
        svd_model = ml_engine.load_model(SVD_MODEL_PATH)
        logger.info("SVD model loaded")
    except Exception as e:
        logger.error("Failed to load SVD model: %s", e)
    try:
        import cbf_engine
        cbf_model = cbf_engine.load_cbf_model(CBF_MODEL_PATH)
        logger.info("CBF model loaded")
    except Exception as e:
        logger.error("Failed to load CBF model: %s", e)


# ── Pydantic schemas ─────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    user_id: int = Field(..., example=1, description="The user ID to generate recommendations for.")
    n: int = Field(10, ge=1, le=50, example=10, description="Number of recommendations to return (1-50).")

class SimilarMoviesRequest(BaseModel):
    movie_id: int = Field(..., example=1, description="The movie ID to find similar movies for.")
    n: int = Field(10, ge=1, le=50, example=10, description="Number of similar movies to return (1-50).")

class MovieRecommendation(BaseModel):
    movieId: int
    score: float

class SimilarMovie(BaseModel):
    movieId: int
    title: str
    genres: str
    similarity_score: float

class RecommendResponse(BaseModel):
    user_id: int
    recommendations: list[MovieRecommendation]

class SimilarMoviesResponse(BaseModel):
    movie_id: int
    similar_movies: list[SimilarMovie]

class HealthResponse(BaseModel):
    status: str
    svd_model_loaded: bool
    cbf_model_loaded: bool


# ── HTML UI ──────────────────────────────────────────────────────────

HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>CinemaIQ - Movie Recommender</title>
  <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #0a0a0f;
      --surface: #13131a;
      --card: #1a1a24;
      --border: #2a2a3a;
      --accent: #e8c76a;
      --text: #e8e4d8;
      --muted: #6b6880;
      --red: #c45c3a;
      --radius: 12px;
    }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'DM Sans', sans-serif;
      min-height: 100vh;
    }
    body::before {
      content: '';
      position: fixed;
      inset: 0;
      background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
      pointer-events: none;
      z-index: 9999;
      opacity: 0.35;
    }
    header {
      padding: 44px 48px 32px;
      border-bottom: 1px solid var(--border);
      position: relative;
    }
    header::after {
      content: '';
      position: absolute;
      bottom: -1px; left: 0; right: 0; height: 1px;
      background: linear-gradient(90deg, transparent, var(--accent), transparent);
    }
    .logo {
      font-family: 'Bebas Neue', sans-serif;
      font-size: 54px;
      letter-spacing: 4px;
      line-height: 1;
      background: linear-gradient(135deg, var(--accent) 0%, #f0a050 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .tagline {
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 3px;
      text-transform: uppercase;
      margin-top: 4px;
    }
    .main { max-width: 920px; margin: 0 auto; padding: 44px 24px; }
    .tabs {
      display: flex; gap: 4px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 4px; margin-bottom: 32px;
      width: fit-content;
    }
    .tab {
      padding: 10px 30px; border-radius: 8px;
      border: none; background: transparent;
      color: var(--muted);
      font-family: 'DM Sans', sans-serif;
      font-size: 14px; font-weight: 500;
      cursor: pointer; transition: all 0.2s;
    }
    .tab.active { background: var(--accent); color: #0a0a0f; font-weight: 600; }
    .tab:not(.active):hover { color: var(--text); }
    .panel { display: none; }
    .panel.active { display: block; animation: fadeUp 0.3s ease; }
    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(10px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    .card {
      background: var(--card); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 32px; margin-bottom: 24px;
    }
    .card-title {
      font-family: 'Bebas Neue', sans-serif;
      font-size: 22px; letter-spacing: 2px;
      color: var(--accent); margin-bottom: 6px;
    }
    .card-desc { color: var(--muted); font-size: 13px; margin-bottom: 28px; line-height: 1.6; }
    .row { display: flex; gap: 16px; }
    .field { flex: 1; margin-bottom: 20px; }
    label {
      display: block; font-size: 11px; font-weight: 500;
      letter-spacing: 2px; text-transform: uppercase;
      color: var(--muted); margin-bottom: 8px;
    }
    input[type="number"] {
      width: 100%; background: var(--surface);
      border: 1px solid var(--border); border-radius: 8px;
      padding: 14px 16px; color: var(--text);
      font-family: 'DM Sans', sans-serif; font-size: 16px;
      transition: border-color 0.2s, box-shadow 0.2s; outline: none;
      -moz-appearance: textfield;
    }
    input[type="number"]::-webkit-inner-spin-button,
    input[type="number"]::-webkit-outer-spin-button { -webkit-appearance: none; }
    input[type="number"]:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(232,199,106,0.1);
    }
    .btn {
      width: 100%; padding: 16px;
      background: var(--accent); color: #0a0a0f;
      border: none; border-radius: 8px;
      font-family: 'Bebas Neue', sans-serif;
      font-size: 18px; letter-spacing: 2px;
      cursor: pointer; transition: all 0.2s; margin-top: 4px;
    }
    .btn:hover { transform: translateY(-1px); box-shadow: 0 8px 24px rgba(232,199,106,0.2); }
    .btn:active { transform: translateY(0); }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
    .loader { display: none; text-align: center; padding: 40px; color: var(--muted); font-size: 12px; letter-spacing: 2px; text-transform: uppercase; }
    .loader.visible { display: block; }
    .loader-dots span {
      display: inline-block; width: 6px; height: 6px;
      background: var(--accent); border-radius: 50%;
      margin: 0 3px; animation: bounce 1.2s infinite;
    }
    .loader-dots span:nth-child(2) { animation-delay: 0.2s; }
    .loader-dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce {
      0%,60%,100% { transform: translateY(0); }
      30% { transform: translateY(-8px); }
    }
    .results { display: none; }
    .results.visible { display: block; animation: fadeUp 0.3s ease; }
    .results-header {
      display: flex; align-items: center;
      justify-content: space-between; margin-bottom: 16px;
    }
    .results-title { font-family: 'Bebas Neue', sans-serif; font-size: 20px; letter-spacing: 2px; }
    .results-count {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 20px; padding: 4px 12px; font-size: 12px; color: var(--muted);
    }
    .movie-list { display: flex; flex-direction: column; gap: 10px; }
    .movie-item {
      display: flex; align-items: center; gap: 16px;
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 10px; padding: 16px 20px;
      transition: border-color 0.2s, transform 0.15s;
      animation: fadeUp 0.3s ease both;
    }
    .movie-item:hover { border-color: rgba(232,199,106,0.3); transform: translateX(4px); }
    .movie-rank {
      font-family: 'Bebas Neue', sans-serif; font-size: 28px;
      color: var(--border); min-width: 36px; text-align: right; line-height: 1;
    }
    .movie-rank.top { color: var(--accent); }
    .movie-info { flex: 1; min-width: 0; }
    .movie-title {
      font-weight: 500; font-size: 15px;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 5px;
    }
    .movie-meta { font-size: 12px; color: var(--muted); display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .genre-tag {
      background: rgba(232,199,106,0.08);
      border: 1px solid rgba(232,199,106,0.15);
      border-radius: 4px; padding: 2px 7px;
      font-size: 11px; color: var(--accent);
    }
    .id-badge {
      background: var(--card); border: 1px solid var(--border);
      border-radius: 4px; padding: 2px 7px; font-size: 11px; color: var(--muted);
    }
    .score-badge {
      background: var(--card); border: 1px solid var(--border);
      border-radius: 8px; padding: 8px 14px;
      text-align: center; min-width: 72px; flex-shrink: 0;
    }
    .score-value { font-family: 'Bebas Neue', sans-serif; font-size: 22px; color: var(--accent); line-height: 1; }
    .score-label { font-size: 10px; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; margin-top: 2px; }
    .error-box {
      background: rgba(196,92,58,0.1); border: 1px solid rgba(196,92,58,0.3);
      border-radius: 8px; padding: 16px 20px;
      color: #e87a5a; font-size: 14px; display: none; margin-bottom: 16px;
    }
    .error-box.visible { display: block; animation: fadeUp 0.2s ease; }
    @media (max-width: 600px) {
      header { padding: 28px 20px 24px; }
      .logo { font-size: 38px; }
      .main { padding: 28px 16px; }
      .card { padding: 20px; }
      .row { flex-direction: column; gap: 0; }
    }
  </style>
</head>
<body>
<header>
  <div class="logo">CINEMAIQ</div>
  <div class="tagline">ML-Powered Movie Recommendations</div>
</header>

<div class="main">
  <div class="tabs">
    <button class="tab active" onclick="switchTab('collab')">By User ID</button>
    <button class="tab" onclick="switchTab('content')">By Movie</button>
  </div>

  <!-- Collaborative Filtering -->
  <div class="panel active" id="panel-collab">
    <div class="card">
      <div class="card-title">Personalized Recommendations</div>
      <div class="card-desc">Enter a User ID to get movies tailored to that user's taste using SVD Collaborative Filtering.</div>
      <div class="row">
        <div class="field">
          <label>User ID</label>
          <input type="number" id="collab-user-id" placeholder="e.g. 1" min="1"/>
        </div>
        <div class="field">
          <label>Number of Results</label>
          <input type="number" id="collab-n" value="10" min="1" max="50"/>
        </div>
      </div>
      <button class="btn" id="collab-btn" onclick="getRecommendations()">GET RECOMMENDATIONS</button>
    </div>
    <div class="error-box" id="collab-error"></div>
    <div class="loader" id="collab-loader">
      <div class="loader-dots"><span></span><span></span><span></span></div>
      <div style="margin-top:12px">Crunching the numbers...</div>
    </div>
    <div class="results" id="collab-results">
      <div class="results-header">
        <div class="results-title">Top Picks for User <span id="collab-result-user"></span></div>
        <div class="results-count"><span id="collab-result-count"></span> results</div>
      </div>
      <div class="movie-list" id="collab-list"></div>
    </div>
  </div>

  <!-- Content-Based Filtering -->
  <div class="panel" id="panel-content">
    <div class="card">
      <div class="card-title">Similar Movies</div>
      <div class="card-desc">Enter a Movie ID to find films with a similar genre profile using Content-Based Filtering. Try ID 1 for Toy Story.</div>
      <div class="row">
        <div class="field">
          <label>Movie ID</label>
          <input type="number" id="cbf-movie-id" placeholder="e.g. 1" min="1"/>
        </div>
        <div class="field">
          <label>Number of Results</label>
          <input type="number" id="cbf-n" value="10" min="1" max="50"/>
        </div>
      </div>
      <button class="btn" id="cbf-btn" onclick="getSimilar()">FIND SIMILAR MOVIES</button>
    </div>
    <div class="error-box" id="cbf-error"></div>
    <div class="loader" id="cbf-loader">
      <div class="loader-dots"><span></span><span></span><span></span></div>
      <div style="margin-top:12px">Scanning genre space...</div>
    </div>
    <div class="results" id="cbf-results">
      <div class="results-header">
        <div class="results-title">Movies Similar to <span id="cbf-result-movie"></span></div>
        <div class="results-count"><span id="cbf-result-count"></span> results</div>
      </div>
      <div class="movie-list" id="cbf-list"></div>
    </div>
  </div>
</div>

<script>
  function switchTab(tab) {
    document.querySelectorAll('.tab').forEach((t,i) =>
      t.classList.toggle('active', (i===0&&tab==='collab')||(i===1&&tab==='content'))
    );
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.getElementById('panel-'+tab).classList.add('active');
  }

  function showError(id, msg) {
    const el = document.getElementById(id);
    el.textContent = '\u26a0 ' + msg;
    el.classList.add('visible');
  }
  function hideError(id) { document.getElementById(id).classList.remove('visible'); }

  function setLoading(prefix, on) {
    document.getElementById(prefix+'-loader').classList.toggle('visible', on);
    document.getElementById(prefix+'-btn').disabled = on;
    if (on) document.getElementById(prefix+'-results').classList.remove('visible');
  }

  async function getRecommendations() {
    const userId = parseInt(document.getElementById('collab-user-id').value);
    const n = parseInt(document.getElementById('collab-n').value) || 10;
    hideError('collab-error');
    if (!userId) { showError('collab-error', 'Please enter a valid User ID.'); return; }

    setLoading('collab', true);
    try {
      const res = await fetch('/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, n })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Request failed');

      document.getElementById('collab-result-user').textContent = '#' + data.user_id;
      document.getElementById('collab-result-count').textContent = data.recommendations.length;

      document.getElementById('collab-list').innerHTML = data.recommendations.map((r, i) => `
        <div class="movie-item" style="animation-delay:${i*0.035}s">
          <div class="movie-rank ${i<3?'top':''}">${String(i+1).padStart(2,'0')}</div>
          <div class="movie-info">
            <div class="movie-title">Movie ID ${r.movieId}</div>
            <div class="movie-meta"><span class="id-badge">ID #${r.movieId}</span></div>
          </div>
          <div class="score-badge">
            <div class="score-value">${r.score.toFixed(2)}</div>
            <div class="score-label">/ 5.0</div>
          </div>
        </div>`).join('');

      setLoading('collab', false);
      document.getElementById('collab-results').classList.add('visible');
    } catch(e) {
      setLoading('collab', false);
      showError('collab-error', e.message);
    }
  }

  async function getSimilar() {
    const movieId = parseInt(document.getElementById('cbf-movie-id').value);
    const n = parseInt(document.getElementById('cbf-n').value) || 10;
    hideError('cbf-error');
    if (!movieId) { showError('cbf-error', 'Please enter a valid Movie ID.'); return; }

    let targetTitle = 'ID #' + movieId;
    try {
      const infoRes = await fetch('/movie/' + movieId);
      if (infoRes.ok) { const info = await infoRes.json(); targetTitle = info.title; }
    } catch(_) {}

    setLoading('cbf', true);
    try {
      const res = await fetch('/similar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ movie_id: movieId, n })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Request failed');

      document.getElementById('cbf-result-movie').textContent = targetTitle;
      document.getElementById('cbf-result-count').textContent = data.similar_movies.length;

      document.getElementById('cbf-list').innerHTML = data.similar_movies.map((r, i) => {
        const genres = r.genres.split('|').slice(0,3).map(g => `<span class="genre-tag">${g}</span>`).join('');
        return `
        <div class="movie-item" style="animation-delay:${i*0.035}s">
          <div class="movie-rank ${i<3?'top':''}">${String(i+1).padStart(2,'0')}</div>
          <div class="movie-info">
            <div class="movie-title">${r.title}</div>
            <div class="movie-meta">${genres}</div>
          </div>
          <div class="score-badge">
            <div class="score-value">${(r.similarity_score*100).toFixed(0)}%</div>
            <div class="score-label">match</div>
          </div>
        </div>`;
      }).join('');

      setLoading('cbf', false);
      document.getElementById('cbf-results').classList.add('visible');
    } catch(e) {
      setLoading('cbf', false);
      showError('cbf-error', e.message);
    }
  }

  document.addEventListener('keydown', e => {
    if (e.key !== 'Enter') return;
    if (document.getElementById('panel-collab').classList.contains('active')) getRecommendations();
    else getSimilar();
  });
</script>
</body>
</html>"""


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui():
    """Serves the interactive UI at http://localhost:8000"""
    return HTML_UI


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    return HealthResponse(
        status="ok",
        svd_model_loaded=svd_model is not None,
        cbf_model_loaded=cbf_model is not None,
    )


@app.post("/recommend", response_model=RecommendResponse, tags=["Collaborative Filtering"])
def recommend(request: RecommendRequest):
    """SVD-based personalized recommendations for a given user_id."""
    if svd_model is None:
        raise HTTPException(status_code=503, detail="SVD model is not loaded.")
    if request.user_id not in svd_model.user2idx:
        raise HTTPException(status_code=404, detail=f"User ID {request.user_id} not found in model vocabulary.")

    user_seen = svd_model.user_history.get(request.user_id, set())
    all_movies = set(svd_model.item2idx.keys())
    unseen = filter(lambda m: m not in user_seen, all_movies)
    scored = map(
        lambda m: MovieRecommendation(movieId=m, score=round(svd_model.predict(request.user_id, m), 4)),
        unseen
    )
    top_n = sorted(scored, key=lambda x: x.score, reverse=True)[:request.n]
    logger.info("✅ /recommend -> user_id=%d, n=%d", request.user_id, request.n)
    return RecommendResponse(user_id=request.user_id, recommendations=top_n)


@app.post("/similar", response_model=SimilarMoviesResponse, tags=["Content-Based Filtering"])
def similar_movies(request: SimilarMoviesRequest):
    """Genre-based similar movies for a given movie_id."""
    if cbf_model is None:
        raise HTTPException(status_code=503, detail="CBF model is not loaded.")
    if request.movie_id not in cbf_model.movie_idx_map:
        raise HTTPException(status_code=404, detail=f"Movie ID {request.movie_id} not found in CBF model.")

    results = cbf_model.get_similar_movies(request.movie_id, n=request.n)
    if not results:
        raise HTTPException(status_code=404, detail=f"No similar movies found for movie ID {request.movie_id}.")

    similar = [SimilarMovie(**r) for r in results]
    logger.info("✅ /similar -> movie_id=%d, n=%d", request.movie_id, request.n)
    return SimilarMoviesResponse(movie_id=request.movie_id, similar_movies=similar)


@app.get("/movie/{movie_id}", tags=["Movie Info"])
def get_movie_info(movie_id: int):
    """Returns title and genres for a given movie_id."""
    if cbf_model is None:
        raise HTTPException(status_code=503, detail="CBF model is not loaded.")
    info = cbf_model.movie_info.get(movie_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Movie ID {movie_id} not found.")
    return {"movieId": movie_id, **info}