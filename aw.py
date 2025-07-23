import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- Load and Clean Data --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("imdb_top_1000.csv")
    df.dropna(subset=['Series_Title'], inplace=True)
    df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')
    df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce')
    df['Genre'] = df['Genre'].fillna('')
    df[['Star1', 'Star2', 'Star3', 'Star4']] = df[['Star1', 'Star2', 'Star3', 'Star4']].fillna('')
    return df

df = load_data()

# -------------------- Preprocessing for Content-Based --------------------
def build_similarity_matrix(data):
    data['metadata'] = data['Overview'].fillna('') + ' ' + data['Genre'] + ' ' + data['Director'] + ' ' + \
                       data['Star1'] + ' ' + data['Star2'] + ' ' + data['Star3'] + ' ' + data['Star4']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['metadata'])
    return cosine_similarity(tfidf_matrix)

similarity_matrix = build_similarity_matrix(df)

# -------------------- Poster Validation --------------------
def is_valid_image(url):
    try:
        r = requests.head(url, timeout=2)
        return r.status_code == 200 and 'image' in r.headers.get('Content-Type', '')
    except:
        return False

# -------------------- Content-Based Recommendation --------------------
def recommend_content(title, df, sim_matrix):
    if title not in df['Series_Title'].values:
        return pd.DataFrame()
    idx = df[df['Series_Title'] == title].index[0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in scores]
    return df.iloc[movie_indices]

# -------------------- Attribute-Based Recommendation (no director) --------------------
def recommend_attributes(df, genres, stars, years, rating):
    filtered = df.copy()

    if genres:
        filtered = filtered[filtered['Genre'].apply(lambda x: any(g in x for g in genres))]

    if stars:
        filtered = filtered[filtered[['Star1', 'Star2', 'Star3', 'Star4']].apply(
            lambda row: any(star in stars for star in row), axis=1)]

    if years:
        filtered = filtered[filtered['Released_Year'].apply(lambda y: y in years)]

    filtered = filtered[filtered['IMDB_Rating'] >= rating]

    return filtered.sort_values(by="IMDB_Rating", ascending=False).head(10)

# -------------------- Streamlit Layout --------------------
st.set_page_config(page_title="Smart Movie Recommender", layout="wide")
st.title("🎬 Smart Movie Recommender")
st.markdown("Personalized recommendations powered by content similarity and user preferences.")

# ------------- Sidebar for Tab Selection and Inputs -------------
with st.sidebar:
    st.header("🎯 Choose Recommendation Mode")

    selected_tab = st.radio("Select one ", ["🎥 Find Similar Movies", "🔍 Filter by Attributes"])

    # Common dropdowns
    unique_titles = sorted(df['Series_Title'].unique())
    unique_genres = sorted({g.strip() for sublist in df['Genre'].dropna().str.split(',') for g in sublist})
    unique_stars = sorted(set(df['Star1'].dropna().tolist() + df['Star2'].dropna().tolist() +
                              df['Star3'].dropna().tolist() + df['Star4'].dropna().tolist()))
    unique_years = sorted(df['Released_Year'].dropna().astype(int).unique())

    if selected_tab == "🎥 Find Similar Movies":
        selected_title = st.selectbox("🎬 Select a Movie Title", [""] + unique_titles)
        genres = []
        stars = []
        year = None
        min_rating = 0.0
    else:
        selected_title = ""
        genres = st.multiselect("🎞️ Select Genre(s)", unique_genres)
        stars = st.multiselect("🌟 Select Star(s)", unique_stars)
        years = st.multiselect("📅 Select Released Year(s)", list(unique_years))
        min_rating = st.slider("⭐ Minimum IMDb Rating", 0.0, 10.0, 8.0, step=0.1)

# -------------------- Display Recommendations --------------------
st.subheader("🎥Top Recommendations")

# Logic for output
if selected_tab == "🎥 Find Similar Movies" and selected_title:
    results = recommend_content(selected_title, df, similarity_matrix)
elif selected_tab == "🔍 Filter by Attributes" and (genres or stars or years):
    results = recommend_attributes(df, genres, stars, years, min_rating)
else:
    results = df.sort_values(by="IMDB_Rating", ascending=False).head(10)

# -------------------- Show Results --------------------
if not results.empty:
    for _, row in results.iterrows():
        st.markdown(f"### {row['Series_Title']} ({int(row['Released_Year']) if not pd.isna(row['Released_Year']) else 'N/A'})")
        if is_valid_image(row['Poster_Link']):
            st.image(row['Poster_Link'], width=150)
        else:
            st.warning("🎭 Poster not available")
        st.write(f"**⭐ IMDb Rating:** {row['IMDB_Rating']}  \n"
                 f"🎞️ **Genre:** {row['Genre']}  \n"
                 f"👨‍🎬 **Director:** {row['Director']}  \n"
                 f"🌟 **Stars:** {row['Star1']}, {row['Star2']}, {row['Star3']}, {row['Star4']}")
        st.write(f"📝 **Overview:** {row['Overview']}")
        st.markdown("---")
else:
    st.info("No matching movies found. Try different filters.")

# Footer
st.markdown("⭐Developed by POOJA BORUDE⭐")
