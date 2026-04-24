import os
import logging
import traceback
import sys
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# --- Model Compatibility Setup ---
import ml_engine
import cbf_engine
sys.modules['__main__'].SVDModel = ml_engine.SVDModel
sys.modules['__main__'].ContentBasedRecommender = cbf_engine.ContentBasedRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CinemaIQ v3.2 // Neural Nexus")

# --- GLOBAL ERROR CATCHER ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global crash: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Neural Engine Error. Check server logs for type mismatches."},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models Safely
SVD_MODEL_PATH = os.getenv("SVD_MODEL_PATH", "svd_model.pkl")
CBF_MODEL_PATH = os.getenv("CBF_MODEL_PATH", "cbf_model.pkl")
svd_model = None
cbf_model = None

@app.on_event("startup")
def load_models():
    global svd_model, cbf_model
    try:
        svd_model = ml_engine.load_model(SVD_MODEL_PATH)
        cbf_model = cbf_engine.load_cbf_model(CBF_MODEL_PATH)
        logger.info("✅ ML Engines Online")
    except Exception as e:
        logger.error(f"❌ Initialization Failed: {e}")

# --- SCHEMAS ---
class RecommendRequest(BaseModel):
    user_id: int
    n: int = Field(10, ge=1, le=50)

class SimilarMoviesRequest(BaseModel):
    query: str # Matches the frontend payload key
    n: int = Field(10, ge=1, le=50)

# --- UI (Cyber-Grid Graphite) ---
HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CinemaIQ // Neural Nexus</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #050508;
            --surface: #0f111a;
            --border: rgba(255, 255, 255, 0.08);
            --accent: #818cf8;
            --accent-glow: rgba(129, 140, 248, 0.4);
            --text: #f8fafc;
            --text-dim: #94a3b8;
            --glass: rgba(15, 17, 26, 0.7);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1); }
        
        body { 
            background-color: var(--bg); 
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.1) 0px, transparent 50%);
            color: var(--text); 
            font-family: 'Plus Jakarta Sans', sans-serif;
            min-height: 100vh;
            overflow-x: hidden;
        }

        .app-container { max-width: 1200px; margin: 80px auto; padding: 0 24px; }

        header { text-align: center; margin-bottom: 60px; }
        .badge { 
            background: rgba(129, 140, 248, 0.1); 
            color: var(--accent); 
            padding: 8px 16px; 
            border-radius: 100px; 
            font-size: 12px; 
            font-weight: 700; 
            letter-spacing: 3px; 
            border: 1px solid var(--accent-glow);
            display: inline-block;
            margin-bottom: 16px;
            text-transform: uppercase;
        }
        h1 { font-size: 56px; font-weight: 800; letter-spacing: -2px; background: linear-gradient(to bottom, #fff, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; }
        
        .panel { 
            background: var(--glass); 
            border: 1px solid var(--border); 
            border-radius: 32px; 
            padding: 40px; 
            position: relative; 
            backdrop-filter: blur(20px);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .input-group { margin-bottom: 28px; }
        label { display: block; font-size: 12px; text-transform: uppercase; color: var(--text-dim); margin-bottom: 12px; letter-spacing: 1.5px; font-weight: 700;}
        
        input {
            width: 100%; background: rgba(0, 0, 0, 0.2); border: 1px solid var(--border);
            padding: 20px; border-radius: 18px; color: white; font-size: 16px; font-family: inherit;
        }
        input:focus { border-color: var(--accent); outline: none; }

        .btn {
            width: 100%; padding: 20px; background: var(--accent); color: white; border: none;
            border-radius: 18px; font-weight: 700; font-size: 16px; cursor: pointer;
        }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }

        .res-container { margin-top: 40px; display: flex; flex-direction: column; gap: 16px;}
        
        .m-card {
            background: rgba(255, 255, 255, 0.03); 
            border: 1px solid var(--border); 
            border-radius: 20px; 
            padding: 24px;
            display: flex; justify-content: space-between; align-items: center;
            animation: slideUp 0.5s ease forwards; opacity: 0;
        }
        
        @keyframes slideUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

        .m-info { flex: 1; padding-right: 20px; }
        .m-name { font-weight: 700; font-size: 18px; color: var(--text); margin-bottom: 6px; }
        .m-genre { font-size: 12px; color: var(--text-dim); background: rgba(255,255,255,0.05); padding: 4px 10px; border-radius: 6px; display: inline-block; }
        
        .m-val { 
            min-width: 70px; text-align: center;
            font-size: 16px; color: var(--accent); font-weight: 800; 
            background: rgba(129, 140, 248, 0.1); padding: 12px; 
            border-radius: 14px; border: 1px solid var(--accent-glow); 
        }

        .loader { display: none; text-align: center; margin: 40px 0; }
        .spinner { 
            width: 30px; height: 30px; border: 3px solid rgba(129, 140, 248, 0.1); 
            border-top-color: var(--accent); border-radius: 50%; 
            animation: spin 0.8s linear infinite; margin: 0 auto 15px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .loading-text { font-size: 10px; letter-spacing: 3px; color: var(--accent); font-weight: 800; text-transform: uppercase; }
        .error-msg { color: #f87171; background: rgba(248, 113, 113, 0.05); padding: 20px; border-radius: 18px; text-align: center;}
    </style>
</head>
<body>

<div class="app-container">
    <header>
        <div class="badge">Engine v3.2 // Stable</div>
        <h1>CinemaIQ</h1>
    </header>

    <div class="grid">
        <div class="panel">
            <div class="input-group">
                <label>User Identity</label>
                <input type="number" id="uid" placeholder="ID (e.g. 1)">
            </div>
            <div class="input-group">
                <label>Result Density</label>
                <input type="number" id="un" value="8" min="1" max="50">
            </div>
            <button class="btn" id="btn-rec" onclick="run('recommend')">GENERATE PREDICTIONS</button>
            <div id="loader-recommend" class="loader">
                <div class="spinner"></div>
                <div class="loading-text">Analyzing Patterns</div>
            </div>
            <div id="list-recommend" class="res-container"></div>
        </div>

        <div class="panel">
            <div class="input-group">
                <label>Movie Seed</label>
                <input type="text" id="mquery" placeholder="Title (e.g. Toy Story)">
            </div>
            <div class="input-group">
                <label>Discovery Depth</label>
                <input type="number" id="mn" value="8" min="1" max="50">
            </div>
            <button class="btn" style="background:transparent; border:1px solid var(--border)" id="btn-sim" onclick="run('similar')">FIND SIMILARITIES</button>
            <div id="loader-similar" class="loader">
                <div class="spinner"></div>
                <div class="loading-text">Mapping Space</div>
            </div>
            <div id="list-similar" class="res-container"></div>
        </div>
    </div>
</div>

<script>
    async function run(type) {
        const btn = document.getElementById(type === 'recommend' ? 'btn-rec' : 'btn-sim');
        const loader = document.getElementById('loader-' + type);
        const list = document.getElementById('list-' + type);
        
        btn.disabled = true;
        loader.style.display = 'block';
        list.innerHTML = '';

        const payload = type === 'recommend' 
            ? { user_id: parseInt(document.getElementById('uid').value), n: parseInt(document.getElementById('un').value) }
            : { query: document.getElementById('mquery').value, n: parseInt(document.getElementById('mn').value) };

        try {
            const r = await fetch('/' + type, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            const data = await r.json();
            if (!r.ok) throw new Error(data.detail || "Connection Error");

            const results = type === 'recommend' ? data.recommendations : data.similar_movies;
            
            if (!results || results.length === 0) {
                list.innerHTML = `<div class="error-msg">No matches found.</div>`;
                return;
            }

            list.innerHTML = results.map((m, i) => `
                <div class="m-card" style="animation-delay: ${i * 0.08}s">
                    <div class="m-info">
                        <div class="m-name">${m.title}</div>
                        <div class="m-genre">${m.genres ? m.genres.split('|').join(' • ') : 'Neural Match'}</div>
                    </div>
                    <div class="m-val">
                        ${type === 'recommend' ? m.score.toFixed(1) : (m.similarity_score * 100).toFixed(0) + '%'}
                    </div>
                </div>
            `).join('');

        } catch (e) {
            list.innerHTML = `<div class="error-msg">ERR: ${e.message.toUpperCase()}</div>`;
        } finally {
            btn.disabled = false;
            loader.style.display = 'none';
        }
    }
</script>
</body>
</html>"""

# --- ENDPOINTS ---
@app.get("/", response_class=HTMLResponse)
def root(): return HTML_UI

@app.post("/recommend")
def recommend(request: RecommendRequest):
    if not svd_model: raise HTTPException(status_code=503, detail="SVD Engine Offline")
    if request.user_id not in svd_model.user2idx:
        raise HTTPException(status_code=404, detail="User ID not in database")

    user_seen = svd_model.user_history.get(request.user_id, set())
    all_movies = list(svd_model.item2idx.keys())
    unseen = [m for m in all_movies if m not in user_seen]
    
    scored = []
    for mid in unseen:
        try:
            score = svd_model.predict(request.user_id, mid)
            info = cbf_model.movie_info.get(mid, {"title": f"ID {mid}", "genres": "N/A"}) if cbf_model else {"title": f"ID {mid}", "genres": "N/A"}
            
            # CRITICAL FIX: Convert NumPy types to standard Python types
            scored.append({
                "movieId": int(mid),
                "title": str(info['title']),
                "genres": str(info['genres']),
                "score": float(score)
            })
        except: continue

    top_n = sorted(scored, key=lambda x: x['score'], reverse=True)[:request.n]
    return {"recommendations": top_n}

@app.post("/similar")
def similar(request: SimilarMoviesRequest):
    if not cbf_model: raise HTTPException(status_code=503, detail="CBF Engine Offline")
    if not request.query: raise HTTPException(status_code=400, detail="Null search query")

    target_id = None
    query = request.query.strip()
    
    # ID check
    if query.isdigit() and int(query) in cbf_model.movie_info:
        target_id = int(query)
    else:
        # Fuzzy Title Search
        search = query.lower()
        target_id = next((mid for mid, info in cbf_model.movie_info.items() if search in info['title'].lower()), None)
    
    if target_id is None:
        raise HTTPException(status_code=404, detail="Movie not found")

    raw_results = cbf_model.get_similar_movies(target_id, n=request.n)
    
    # CRITICAL FIX: Convert NumPy types
    clean_results = []
    for r in raw_results:
        clean_results.append({
            "movieId": int(r['movieId']),
            "title": str(r['title']),
            "genres": str(r['genres']),
            "similarity_score": float(r['similarity_score'])
        })

    return {"similar_movies": clean_results}