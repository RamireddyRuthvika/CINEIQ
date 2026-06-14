import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import recommender as rec


@asynccontextmanager
async def lifespan(app: FastAPI):
    models_dir = "models"
    required_pickles = [
        "df.pkl", "df_final.pkl", "tfidf_matrix.pkl",
        "ratings.pkl", "movie.pkl", "links.pkl",
        "sparse_ratings.pkl", "reduced_matrix.pkl",
        "user_index.pkl", "movie_index.pkl",
        "user_map.pkl", "movie_map.pkl",
        "tmdb_to_ml.pkl", "ml_to_tmdb.pkl",
        "sentiment_scores.pkl"
    ]
    all_exist = all(
        os.path.exists(f"{models_dir}/{f}")
        for f in required_pickles
    )
    if not all_exist:
        print("Models not found — generating pickles (first time setup, ~15-20 mins)...")
        rec.generate_pickles()
    else:
        print("Models found — loading from pickle...")
    rec.load_models()
    yield


app = FastAPI(title="CINEIQ API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    user_id: int
    top_n:   int   = 5
    alpha:   float = 0.5
    beta:    float = 0.4
    gamma:   float = 0.1

class SimilarRequest(BaseModel):
    movie_name: str
    top_n:      int = 5


@app.get("/health")
def health():
    return {"status": "ok", "model": "CINEIQ v1.0"}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    results = rec.hybrid_recommendation(
        user_id=req.user_id,
        top_n=req.top_n,
        alpha=req.alpha,
        beta=req.beta,
        gamma=req.gamma,
    )
    if not results:
        raise HTTPException(status_code=404, detail=f"No recommendations found for user {req.user_id}.")
    return results


@app.post("/similar")
def similar(req: SimilarRequest):
    results = rec.recommend(req.movie_name, top_n=req.top_n)
    if not results:
        raise HTTPException(status_code=404, detail=f"Movie '{req.movie_name}' not found.")
    return results


@app.get("/taste/{user_id}")
def taste_profile(user_id: int):
    profile = rec.get_user_taste_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")
    return profile