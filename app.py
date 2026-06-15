import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="CINEIQ", page_icon="🎬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Inter:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0f0f0f; color: #e8e8e8; }
[data-testid="stSidebar"] { background-color: #161616; border-right: 1px solid #2a2a2a; }
.cineiq-logo { font-family: 'DM Serif Display', serif; font-size: 2.2rem; color: #e8c77d; }
.cineiq-tagline { font-size: 0.75rem; color: #666; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 2rem; }
.page-title { font-family: 'DM Serif Display', serif; font-size: 1.8rem; color: #e8c77d; margin-bottom: 0.25rem; }
.page-subtitle { font-size: 0.85rem; color: #888; margin-bottom: 1.5rem; }
.movie-card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 0.75rem; }
.movie-card:hover { border-color: #e8c77d; }
.movie-title { font-size: 1rem; font-weight: 600; color: #f0f0f0; margin-bottom: 0.3rem; }
.movie-meta { font-size: 0.78rem; color: #777; margin-bottom: 0.4rem; }
.movie-reason { font-size: 0.82rem; color: #e8c77d; font-style: italic; }
.genre-tag { display: inline-block; background: #252525; border: 1px solid #333; border-radius: 4px; padding: 0.15rem 0.5rem; font-size: 0.72rem; color: #aaa; margin-right: 0.3rem; }
.stat-card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; padding: 1rem; text-align: center; }
.stat-value { font-family: 'DM Serif Display', serif; font-size: 1.8rem; color: #e8c77d; }
.stat-label { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.08em; }
.gold-divider { height: 1px; background: linear-gradient(to right, #e8c77d33, #e8c77d88, #e8c77d33); margin: 1.5rem 0; }
.info-box { background: #1a1a1a; border-left: 3px solid #e8c77d; padding: 0.75rem 1rem; border-radius: 0 6px 6px 0; font-size: 0.85rem; color: #aaa; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


#  HELPERS

def api_get(path):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=120)
        if r.status_code == 200:
            return r.json(), None
        try:
            detail = r.json().get("detail", "Something went wrong.")
        except Exception:
            detail = f"API error {r.status_code} — check uvicorn terminal."
        return None, detail
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Run: uvicorn main:app --reload"


def api_post(path, payload):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=120)
        if r.status_code == 200:
            return r.json(), None
        try:
            detail = r.json().get("detail", "Something went wrong.")
        except Exception:
            detail = f"API error {r.status_code} — check uvicorn terminal."
        return None, detail
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Run: uvicorn main:app --reload"


def render_movie_card(item, index=None, show_reason=True):
    title      = item.get("title", "Unknown")
    genres     = item.get("genres", [])
    reason     = item.get("reason", "")
    score      = item.get("score", None)
    main_genre = genres[0] if genres else None

    rank        = f'<span style="color:#555;margin-right:0.5rem;">#{index}</span>' if index else ""
    genre_tags  = "".join([f'<span class="genre-tag">{g}</span>' for g in genres])
    score_str   = f"Score: {score:.3f}" if score else ""
    main_str    = f'<span style="color:#e8c77d;font-size:0.75rem;margin-left:0.5rem;font-weight:400;">● {main_genre}</span>' if main_genre else ""
    reason_html = f'<div class="movie-reason">💡 {reason}</div>' if show_reason and reason else ""

    st.markdown(f"""
    <div class="movie-card">
        <div class="movie-title">{rank}{title}{main_str}</div>
        <div class="movie-meta">{score_str}</div>
        <div style="margin-bottom:0.4rem">{genre_tags}</div>
        {reason_html}
    </div>""", unsafe_allow_html=True)


# SIDEBAR 

with st.sidebar:
    st.markdown('<div class="cineiq-logo">CINEIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="cineiq-tagline">Open · Explainable · Yours</div>', unsafe_allow_html=True)

    page = st.radio("Navigate", ["🔍 Movie Search", "🎯 Recommendations", "📊 Taste Dashboard"],
                    label_visibility="collapsed")

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    health, _ = api_get("/health")
    if health:
        st.markdown('<div style="font-size:0.75rem;color:#4caf50;">● API Online</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:0.75rem;color:#f44336;">● API Offline — run uvicorn main:app --reload</div>', unsafe_allow_html=True)



# PAGE 1 — MOVIE SEARCH


if page == "🔍 Movie Search":
    st.markdown('<div class="page-title">Find Similar Movies</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Content-based recommendations using TF-IDF cosine similarity.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        movie_name = st.text_input("Movie title", placeholder="e.g. Toy Story, Interstellar...", label_visibility="collapsed")
    with col2:
        top_n = st.selectbox("Results", [5, 10, 15], label_visibility="collapsed")

    if st.button("Search", use_container_width=True):
        if not movie_name.strip():
            st.warning("Please enter a movie title.")
        else:
            with st.spinner("Finding similar movies..."):
                results, err = api_post("/similar", {"movie_name": movie_name, "top_n": top_n})
            if err:
                st.markdown(f'<div class="info-box">🎬 {err}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:0.85rem;color:#666;margin-bottom:1rem;">Results for <span style="color:#e8c77d">"{movie_name}"</span></div>', unsafe_allow_html=True)
                for i, item in enumerate(results, 1):
                    render_movie_card(item, index=i, show_reason=True)



# PAGE 2 — RECOMMENDATIONS

elif page == "🎯 Recommendations":
    st.markdown('<div class="page-title">Your Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Hybrid engine — collaborative + content + sentiment.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        user_id = st.number_input("User ID", min_value=1, max_value=162541, value=200, step=1)
    with col2:
        alpha = st.slider("Collaborative",  0.0, 1.0, 0.5, 0.05)
        beta  = st.slider("Content",        0.0, 1.0, 0.4, 0.05)
        gamma = st.slider("Sentiment",      0.0, 1.0, 0.1, 0.05)
        total = alpha + beta + gamma
        label = f"{round(alpha/total*100)}% Collab · {round(beta/total*100)}% Content · {round(gamma/total*100)}% Sentiment" if total > 0 else "Set at least one weight"
        st.markdown(f'<div style="font-size:0.75rem;color:#888;">{label}</div>', unsafe_allow_html=True)
    with col3:
        top_n = st.selectbox("Results", [5, 10, 15], key="rec_n")

    if st.button("Get Recommendations", use_container_width=True):
        with st.spinner("Generating recommendations..."):
            results, err = api_post("/recommend", {
                "user_id": int(user_id),
                "top_n":   top_n,
                "alpha":   alpha,
                "beta":    beta,
                "gamma":   gamma
            })
        if err:
            st.markdown(f'<div class="info-box">🎬 {err}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:0.85rem;color:#666;margin-bottom:1rem;">Top {len(results)} picks for User <span style="color:#e8c77d">{user_id}</span></div>', unsafe_allow_html=True)
            for i, item in enumerate(results, 1):
                render_movie_card(item, index=i, show_reason=True)



# PAGE 3 — TASTE DASHBOARD


elif page == "📊 Taste Dashboard":
    st.markdown('<div class="page-title">Your Taste Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Visualize your preferences from rating history.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        user_id = st.number_input("User ID", min_value=1, max_value=162541, value=200, step=1, key="dash_uid")
    with col2:
        st.write("")
        st.write("")
        load = st.button("Load Profile", use_container_width=True)

    if load:
        with st.spinner("Loading taste profile..."):
            profile, err = api_get(f"/taste/{int(user_id)}")

        if err:
            st.markdown(f'<div class="info-box">🎬 {err}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

            # stat cards
            c1, c2, c3 = st.columns(3)
            top_genre = max(profile['genre_counts'], key=profile['genre_counts'].get) if profile['genre_counts'] else "N/A"
            with c1:
                st.markdown(f'<div class="stat-card"><div class="stat-value">{profile["total_rated"]}</div><div class="stat-label">Movies Rated</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="stat-card"><div class="stat-value">{profile["avg_rating"]}</div><div class="stat-label">Avg Rating</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="stat-card"><div class="stat-value" style="font-size:1.2rem">{top_genre}</div><div class="stat-label">Top Genre</div></div>', unsafe_allow_html=True)

            st.write("")

            # charts
            ch1, ch2 = st.columns(2)

            with ch1:
                st.markdown('<div style="font-size:0.9rem;color:#aaa;margin-bottom:0.5rem;">Genre Radar</div>', unsafe_allow_html=True)
                gc = profile['genre_counts']
                if gc:
                    top8 = dict(sorted(gc.items(), key=lambda x: x[1], reverse=True)[:8])
                    cats = list(top8.keys()) + [list(top8.keys())[0]]
                    vals = list(top8.values()) + [list(top8.values())[0]]
                    fig = go.Figure(go.Scatterpolar(
                        r=vals, theta=cats, fill='toself',
                        fillcolor='rgba(232,199,125,0.15)',
                        line=dict(color='#e8c77d', width=2),
                        marker=dict(color='#e8c77d', size=5)
                    ))
                    fig.update_layout(
                        polar=dict(
                            bgcolor='#1a1a1a',
                            radialaxis=dict(visible=True, showticklabels=False, gridcolor='#333'),
                            angularaxis=dict(gridcolor='#333', tickfont=dict(color='#aaa', size=11))
                        ),
                        paper_bgcolor='#0f0f0f', showlegend=False,
                        margin=dict(t=20, b=20, l=40, r=40), height=320
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with ch2:
                st.markdown('<div style="font-size:0.9rem;color:#aaa;margin-bottom:0.5rem;">Decade Preferences</div>', unsafe_allow_html=True)
                dc = profile['decade_counts']
                if dc:
                    df_dec = pd.DataFrame(sorted(dc.items()), columns=['Decade', 'Count'])
                    fig2 = px.bar(df_dec, x='Decade', y='Count', color_discrete_sequence=['#e8c77d'])
                    fig2.update_layout(
                        paper_bgcolor='#0f0f0f', plot_bgcolor='#1a1a1a',
                        font=dict(color='#aaa'),
                        xaxis=dict(gridcolor='#2a2a2a'),
                        yaxis=dict(gridcolor='#2a2a2a'),
                        margin=dict(t=10, b=10, l=10, r=10), height=320
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

            # avg rating per genre
            st.markdown('<div style="font-size:0.9rem;color:#aaa;margin-bottom:0.75rem;">Avg Rating by Genre</div>', unsafe_allow_html=True)
            ar = profile['avg_genre_ratings']
            if ar:
                df_ar = pd.DataFrame(sorted(ar.items(), key=lambda x: x[1], reverse=True), columns=['Genre', 'Avg Rating'])
                fig3 = px.bar(df_ar, x='Avg Rating', y='Genre', orientation='h',
                              color='Avg Rating',
                              color_continuous_scale=[[0,'#2a2a2a'],[0.5,'#b8973d'],[1,'#e8c77d']],
                              range_x=[0, 5])
                fig3.update_layout(
                    paper_bgcolor='#0f0f0f', plot_bgcolor='#1a1a1a',
                    font=dict(color='#aaa'),
                    xaxis=dict(gridcolor='#2a2a2a', range=[0, 5]),
                    yaxis=dict(gridcolor='#2a2a2a'),
                    coloraxis_showscale=False,
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=max(300, len(ar) * 28)
                )
                st.plotly_chart(fig3, use_container_width=True)
