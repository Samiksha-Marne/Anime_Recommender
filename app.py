from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__, static_url_path='/static')

# Load the anime dataset
anime = pd.read_csv('anime.csv')

# Drop rows with missing values
anime.dropna(subset=['name', 'genre'], inplace=True)

# Text Preprocessing
anime['genre'] = anime['genre'].str.replace(' ', '')  # Remove spaces for multi-word genres

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
tfidf_matrix = tfidf_vectorizer.fit_transform(anime['genre'])

# Calculate cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get anime recommendations
def get_recommendations(genres, cosine_sim=cosine_sim):
    genre_query = ''.join(genres.split()).lower()  # Prepare genre query
    match_all_indices = []
    match_one_indices = []

    for idx, genre_list in enumerate(anime['genre']):
        genre_list = ''.join(genre_list.split()).lower()
        if all(genre in genre_list for genre in genre_query.split(',')):
            match_all_indices.append(idx)
        elif any(genre in genre_list for genre in genre_query.split(',')):
            match_one_indices.append(idx)

    anime_match_all = anime.iloc[match_all_indices][['name', 'genre', 'rating']]
    anime_match_one = anime.iloc[match_one_indices][['name', 'genre', 'rating']]

    return anime_match_all, anime_match_one
    
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        anime_match_all, anime_match_one = get_recommendations(user_input)
        return render_template('index.html', anime_match_all=anime_match_all.to_dict(orient='records'), anime_match_one=anime_match_one.to_dict(orient='records'))

    return render_template('index.html', anime_match_all=None, anime_match_one=None)


if __name__ == '__main__':
    app.run(debug=True)
