"""
CINEIQ — recommender.py
Run generate_pickles() ONCE from your notebook to build models/.
Then FastAPI calls load_models() once at startup.
"""

import ast
import os
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

# ── GLOBALS ────────────────────────────────────────────────────────────────
df               = None
df_final         = None
tfidf_matrix     = None
ratings          = None
movie            = None
links            = None
sparse_ratings   = None
reduced_matrix   = None
user_index       = None
movie_index      = None
user_map         = None
movie_map        = None
tmdb_to_ml       = None
ml_to_tmdb       = None
ratings_by_user  = None
sentiment_scores = None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def fetch_names(text):
    names = []
    try:
        for i in ast.literal_eval(text):
            names.append(i['name'])
    except Exception:
        pass
    return names


def fetch_cast(text):
    cast_list = []
    counter = 0
    try:
        for i in ast.literal_eval(text):
            if counter < 3:
                cast_list.append(i['name'])
                counter += 1
            else:
                break
    except Exception:
        pass
    return cast_list


def fetch_director(text):
    crew_list = []
    try:
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                crew_list.append(i['name'])
                break
    except Exception:
        pass
    return crew_list


def normalize_dict(d):
    if not d:
        return {}
    max_v = max(d.values())
    if max_v == 0:
        return {k: 0.0 for k in d}
    return {k: v / max_v for k, v in d.items()}


def has_watched(user_id, ml_id):
    row = user_index.get(user_id)
    col = movie_index.get(ml_id)
    if row is None or col is None:
        return False
    return sparse_ratings[row, col] > 0


def get_movie_genres(ml_id):
    row = movie[movie['movieId'] == ml_id]
    if row.empty:
        return []
    genres = row['genres'].values[0]
    if pd.isna(genres) or genres == '(no genres listed)':
        return []
    return genres.split('|')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — GENERATE PICKLES (run once from notebook)
# ══════════════════════════════════════════════════════════════════════════════

def generate_pickles(
    archive_path="archive",
    ml_path="ml-25m",
    imdb_path="IMDB_Dataset.csv",
    models_dir="models"
):
    os.makedirs(models_dir, exist_ok=True)

    # ── Content ────────────────────────────────────────────────────────────
    print("Loading TMDB metadata...")
    movies_df = pd.read_csv(f"{archive_path}/movies_metadata.csv", low_memory=False)
    keywords  = pd.read_csv(f"{archive_path}/keywords.csv")
    credits   = pd.read_csv(f"{archive_path}/credits.csv")

    movies_df = movies_df[['id','title','overview','genres','vote_average','vote_count','popularity']]
    movies_df = movies_df[movies_df['id'].str.isnumeric()]
    movies_df['id'] = movies_df['id'].astype(int)

    df_t = pd.merge(movies_df, keywords, on='id', how='left')
    df_t = pd.merge(df_t, credits, on='id', how='left')
    df_t.dropna(subset=['overview'], inplace=True)

    df_t['genres']   = df_t['genres'].apply(fetch_names)
    df_t['keywords'] = df_t['keywords'].apply(fetch_names)
    df_t['cast']     = df_t['cast'].apply(fetch_cast)
    df_t['Director'] = df_t['crew'].apply(fetch_director)
    df_t['overview'] = df_t['overview'].apply(lambda x: x.split())

    for key in ['genres', 'cast', 'keywords', 'Director']:
        df_t[key] = df_t[key].apply(lambda x: [i.replace(' ', '') for i in x])

    df_t['tags'] = (
        df_t['overview'] * 2 + df_t['genres'] * 3 +
        df_t['keywords'] * 1 + df_t['cast']   + df_t['Director']
    )
    df_t['tags'] = df_t['tags'].apply(lambda x: " ".join(x))

    df_final_t = df_t[['id', 'title', 'tags']].reset_index(drop=True)
    df_final_t['tags'] = df_final_t['tags'].str.lower()

    print("Fitting TF-IDF...")
    tfidf_t        = TfidfVectorizer(stop_words='english')
    tfidf_matrix_t = tfidf_t.fit_transform(df_final_t['tags'])

    # ── Collaborative ──────────────────────────────────────────────────────
    print("Loading MovieLens ratings (full 25M)...")
    ratings_t = pd.read_csv(f"{ml_path}/ratings.csv")
    movie_t   = pd.read_csv(f"{ml_path}/movies.csv")
    links_t   = pd.read_csv(f"{ml_path}/links.csv")

    links_t.dropna(subset=['tmdbId'], inplace=True)
    links_t['tmdbId'] = links_t['tmdbId'].astype(int)

    tmdb_to_ml_t = dict(zip(links_t['tmdbId'], links_t['movieId']))
    ml_to_tmdb_t = dict(zip(links_t['movieId'], links_t['tmdbId']))

    print("Building sparse matrix...")
    user_cat  = ratings_t['userId'].astype('category')
    movie_cat = ratings_t['movieId'].astype('category')

    sparse_t = csr_matrix(
        (ratings_t['rating'].values,
         (user_cat.cat.codes, movie_cat.cat.codes))
    )

    user_map_t    = dict(enumerate(user_cat.cat.categories))
    movie_map_t   = dict(enumerate(movie_cat.cat.categories))
    user_index_t  = {v: k for k, v in user_map_t.items()}
    movie_index_t = {v: k for k, v in movie_map_t.items()}

    print("Fitting SVD (this takes a few minutes)...")
    svd_t            = TruncatedSVD(n_components=100, random_state=42)
    reduced_matrix_t = svd_t.fit_transform(sparse_t)

    # ── Sentiment ──────────────────────────────────────────────────────────
    print("Training sentiment classifier...")
    imdb = pd.read_csv(imdb_path)
    imdb['label'] = (imdb['sentiment'] == 'positive').astype(int)

    sent_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
        ('clf',   LogisticRegression(max_iter=300))
    ])
    sent_pipe.fit(imdb['review'], imdb['label'])

    probs          = sent_pipe.predict_proba(df_final_t['tags'])[:, 1]
    tmdb_sentiment = dict(zip(df_final_t['id'], probs))

    sentiment_t = {}
    for tmdb_id, score in tmdb_sentiment.items():
        ml_id = tmdb_to_ml_t.get(tmdb_id)
        if ml_id:
            sentiment_t[int(ml_id)] = round(float(score), 4)

    # ── Save ───────────────────────────────────────────────────────────────
    print("Saving pickles...")
    pickle.dump(df_t,            open(f"{models_dir}/df.pkl",              "wb"))
    pickle.dump(df_final_t,      open(f"{models_dir}/df_final.pkl",        "wb"))
    pickle.dump(tfidf_matrix_t,  open(f"{models_dir}/tfidf_matrix.pkl",    "wb"))
    pickle.dump(ratings_t,       open(f"{models_dir}/ratings.pkl",         "wb"))
    pickle.dump(movie_t,         open(f"{models_dir}/movie.pkl",           "wb"))
    pickle.dump(links_t,         open(f"{models_dir}/links.pkl",           "wb"))
    pickle.dump(sparse_t,        open(f"{models_dir}/sparse_ratings.pkl",  "wb"))
    pickle.dump(reduced_matrix_t,open(f"{models_dir}/reduced_matrix.pkl",  "wb"))
    pickle.dump(user_index_t,    open(f"{models_dir}/user_index.pkl",      "wb"))
    pickle.dump(movie_index_t,   open(f"{models_dir}/movie_index.pkl",     "wb"))
    pickle.dump(user_map_t,      open(f"{models_dir}/user_map.pkl",        "wb"))
    pickle.dump(movie_map_t,     open(f"{models_dir}/movie_map.pkl",       "wb"))
    pickle.dump(tmdb_to_ml_t,    open(f"{models_dir}/tmdb_to_ml.pkl",      "wb"))
    pickle.dump(ml_to_tmdb_t,    open(f"{models_dir}/ml_to_tmdb.pkl",      "wb"))
    pickle.dump(sentiment_t,     open(f"{models_dir}/sentiment_scores.pkl","wb"))
    print("Done. All pickles saved.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOAD MODELS (FastAPI startup)
# ══════════════════════════════════════════════════════════════════════════════

def load_models(models_dir="models"):
    global df, df_final, tfidf_matrix, ratings, movie, links
    global sparse_ratings, reduced_matrix
    global user_index, movie_index, user_map, movie_map
    global tmdb_to_ml, ml_to_tmdb, ratings_by_user, sentiment_scores

    print("Loading models from pickle...")
    df               = pickle.load(open(f"{models_dir}/df.pkl",              "rb"))
    df_final         = pickle.load(open(f"{models_dir}/df_final.pkl",        "rb"))
    tfidf_matrix     = pickle.load(open(f"{models_dir}/tfidf_matrix.pkl",    "rb"))
    ratings          = pickle.load(open(f"{models_dir}/ratings.pkl",         "rb"))
    movie            = pickle.load(open(f"{models_dir}/movie.pkl",           "rb"))
    links            = pickle.load(open(f"{models_dir}/links.pkl",           "rb"))
    sparse_ratings   = pickle.load(open(f"{models_dir}/sparse_ratings.pkl",  "rb"))
    reduced_matrix   = pickle.load(open(f"{models_dir}/reduced_matrix.pkl",  "rb"))
    user_index       = pickle.load(open(f"{models_dir}/user_index.pkl",      "rb"))
    movie_index      = pickle.load(open(f"{models_dir}/movie_index.pkl",     "rb"))
    user_map         = pickle.load(open(f"{models_dir}/user_map.pkl",        "rb"))
    movie_map        = pickle.load(open(f"{models_dir}/movie_map.pkl",       "rb"))
    tmdb_to_ml       = pickle.load(open(f"{models_dir}/tmdb_to_ml.pkl",      "rb"))
    ml_to_tmdb       = pickle.load(open(f"{models_dir}/ml_to_tmdb.pkl",      "rb"))
    sentiment_scores = pickle.load(open(f"{models_dir}/sentiment_scores.pkl","rb"))

    ratings_by_user  = ratings.groupby('userId')
    print("All models loaded. Ready.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CONTENT-BASED
# ══════════════════════════════════════════════════════════════════════════════

def recommend(movie_name, top_n=5):
    mask = df_final['title'].str.strip().str.lower().str.replace(" ", "", regex = False) == movie_name.strip().lower().replace(" ", "")
    if not mask.any():
        return []
    idx    = df_final[mask].index[0]
    scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[1:top_n+1]

    anchor_tmdb_id = df_final.iloc[idx]['id']
    anchor_ml_id   = tmdb_to_ml.get(anchor_tmdb_id)

    results = []
    for i, score in ranked:
        rec_tmdb_id = df_final.iloc[i]['id']
        rec_ml_id   = tmdb_to_ml.get(rec_tmdb_id)
        reason      = explain_content(rec_ml_id, anchor_ml_id) if anchor_ml_id and rec_ml_id else f"Similar to {movie_name}"
        results.append({
            "title":  df_final['title'].iloc[i],
            "score":  round(float(score), 4),
            "reason": reason
        })
    return results

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — COLLABORATIVE
# ══════════════════════════════════════════════════════════════════════════════

def recomm_3(user_id):
    if user_id not in user_index:
        return {}
    idx         = user_index[user_id]
    user_vector = reduced_matrix[idx].reshape(1, -1)
    sim_arr     = cosine_similarity(user_vector, reduced_matrix).flatten()
    top_users   = sorted(enumerate(sim_arr), key=lambda x: x[1], reverse=True)[1:11]

    movie_scores = {}
    for similar_idx, similar_score in top_users:
        similar_uid = user_map[similar_idx]
        if similar_uid not in ratings_by_user.groups:
            continue
        for _, row in ratings_by_user.get_group(similar_uid).iterrows():
            ml_id  = row['movieId']
            rating = row['rating']
            movie_scores[ml_id] = movie_scores.get(ml_id, 0) + similar_score * rating
    return movie_scores


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

def explain_content(recommended_ml_id, anchor_ml_id):
    anchor_genres = set(get_movie_genres(anchor_ml_id))
    rec_genres    = set(get_movie_genres(recommended_ml_id))
    common        = anchor_genres & rec_genres

    anchor_title = movie[movie['movieId'] == anchor_ml_id]['title']
    anchor_title = anchor_title.values[0] if not anchor_title.empty else "a movie you liked"

    if common:
        return f"Because you liked {anchor_title} — shared genres: {', '.join(sorted(common))}"
    elif rec_genres:
        return f"Because you liked {anchor_title} — similar {rec_genres.pop()} film"
    else:
        return f"Because you liked {anchor_title} — similar style and themes"


def explain_collaborative():
    return "Users with similar taste to you also rated this highly"


def explain_hybrid(recommended_ml_id, anchor_ml_id, s_cf, s_con):
    if s_cf == 0 and s_con == 0:
        return "Recommended based on overall audience reception"
    elif s_con > s_cf:
        return explain_content(recommended_ml_id, anchor_ml_id)
    else:
        return explain_collaborative()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — HYBRID
# ══════════════════════════════════════════════════════════════════════════════

def hybrid_recommendation(user_id, top_n=5, alpha=0.5, beta = 0.4, gamma=0.1):
    """
    alpha         → collaborative weight
    beta          → content weight
    gamma         → sentiment weight
    """
    if user_id not in user_index:
        return []

    # ── Collaborative ──────────────────────────────────────────────────────
    svd_scores = recomm_3(user_id)

    # ── Content ────────────────────────────────────────────────────────────
    user_rated     = ratings[ratings['userId'] == user_id]
    user_favorites = (user_rated[user_rated['rating'] >= 4.0]
                      .sort_values('rating', ascending=False)['movieId'].tolist())
    if not user_favorites:
        user_favorites = user_rated['movieId'].tolist()

    user_ratings_dict = dict(zip(user_rated['movieId'], user_rated['rating']))
    content_scores    = {}

    for anchor_ml_id in user_favorites[:3]:
        anchor_tmdb_id = ml_to_tmdb.get(anchor_ml_id)
        if anchor_tmdb_id is None:
            continue
        match = df_final[df_final['id'] == anchor_tmdb_id]
        if match.empty:
            continue
        df_final_idx  = match.index[0]
        anchor_rating = user_ratings_dict[anchor_ml_id]
        sim_vector    = cosine_similarity(tfidf_matrix[df_final_idx], tfidf_matrix).flatten()

        for i, score in enumerate(sim_vector):
            row_tmdb_id  = df_final.iloc[i]['id']
            target_ml_id = tmdb_to_ml.get(row_tmdb_id)
            if target_ml_id:
                content_scores[target_ml_id] = (
                    content_scores.get(target_ml_id, 0) + score * anchor_rating
                )

    # ── Normalize ──────────────────────────────────────────────────────────
    norm_svd     = normalize_dict(svd_scores)
    norm_content = normalize_dict(content_scores)

    # ── Sentiment ──────────────────────────────────────────────────────────
    all_movies = set(norm_svd.keys()) | set(norm_content.keys())
    candidate_sentiment = {
        ml_id: sentiment_scores.get(int(ml_id), 0.5)
        for ml_id in all_movies
        if not has_watched(user_id, ml_id)
    }
    norm_sentiment = normalize_dict(candidate_sentiment)

    # ── Best anchor for explainability ────────────────────────────────────
    best_anchor = next(
        (mid for mid in user_favorites[:3]
         if ml_to_tmdb.get(mid) and
         not df_final[df_final['id'] == ml_to_tmdb.get(mid)].empty),
        None
    )

    # ── Blend ──────────────────────────────────────────────────────────────
    hybrid_scores = []
    total = alpha + beta + gamma
    if total == 0:
        return []
    for ml_id in all_movies:
        if has_watched(user_id, ml_id):
            continue
        s_cf  = norm_svd.get(ml_id, 0)
        s_con = norm_content.get(ml_id, 0)
        if s_cf == 0 and s_con == 0:
            continue
        s_sent      = norm_sentiment.get(ml_id, 0)
        final_score = ((alpha * s_cf) + (beta * s_con) + (gamma * s_sent)) / total
        hybrid_scores.append((ml_id, final_score))

    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:top_n]

    # ── Results with reasons ───────────────────────────────────────────────
    recommended = []
    for ml_id, score in hybrid_scores:
        title_match = movie[movie['movieId'] == ml_id]['title']
        if title_match.empty:
            continue
        s_cf   = norm_svd.get(ml_id, 0)
        s_con  = norm_content.get(ml_id, 0)
        reason = explain_hybrid(ml_id, best_anchor, s_cf, s_con)
        recommended.append({
            "title":  title_match.values[0],
            "genres": get_movie_genres(ml_id),
            "score":  round(float(score), 4),
            "reason": reason
        })

    return recommended


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — TASTE PROFILE
# ══════════════════════════════════════════════════════════════════════════════

def get_user_taste_profile(user_id):
    if user_id not in user_index:
        return None

    user_rated = ratings[ratings['userId'] == user_id]
    if user_rated.empty:
        return None

    user_movies = user_rated.merge(movie, on='movieId', how='left')

    genre_counts  = {}
    genre_ratings = {}
    genre_n       = {}
    for _, row in user_movies.iterrows():
        if pd.isna(row['genres']):
            continue
        for g in row['genres'].split('|'):
            genre_counts[g]  = genre_counts.get(g, 0) + 1
            genre_ratings[g] = genre_ratings.get(g, 0) + row['rating']
            genre_n[g]       = genre_n.get(g, 0) + 1

    decade_counts = {}
    for title in user_movies['title']:
        t = str(title)
        if '(' in t and ')' in t:
            try:
                year   = int(t[t.rfind('(')+1:t.rfind(')')])
                decade = f"{(year // 10) * 10}s"
                decade_counts[decade] = decade_counts.get(decade, 0) + 1
            except ValueError:
                pass

    avg_genre_ratings = {
        g: round(genre_ratings[g] / genre_n[g], 2)
        for g in genre_ratings
    }

    return {
        "user_id":           user_id,
        "total_rated":       int(len(user_rated)),
        "avg_rating":        round(float(user_rated['rating'].mean()), 2),
        "genre_counts":      genre_counts,
        "decade_counts":     decade_counts,
        "avg_genre_ratings": avg_genre_ratings,
    }
