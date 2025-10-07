import math
import os
import sys
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Helper Functions ---
def prepare_data(x):
    # Converts string to lowercase and removes spaces for consistent comparison
    return str.lower(x.replace(" ", ""))

def create_soup(x):
    # Combines selected features into one string for vectorization
    return x['Genre'] + ' ' + x['Tags'] + ' ' + x['Actors'] + ' ' + x['ViewerRating']

def get_recommendations(title, cosine_sim):
    global result
    title = title.replace(' ', '').lower()
    
    # Safely get the index
    if title not in indices:
        # This shouldn't happen if the index is built correctly, but safety first
        print(f"Error: Title '{title}' not found in indices.", file=sys.stderr)
        return pd.DataFrame() # Return empty if title is missing
        
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:51]
    movie_indices = [i[0] for i in sim_scores]
    result = netflix_data.iloc[movie_indices]
    result.reset_index(inplace=True)
    return result

# --- Data Loading and Feature Engineering (Critical Startup Code) ---
try:
    # Use os.path.join for path robustness, assuming NetflixDataset.csv is in the same directory
    DATA_PATH = os.path.join(os.path.dirname(__file__), 'NetflixDataset.csv')
    
    # If the CSV is inside a 'data/' folder, you would change the above line to:
    # DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'NetflixDataset.csv')

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at expected path: {DATA_PATH}")

    netflix_data = pd.read_csv(DATA_PATH, encoding='latin-1', index_col='Title')

except Exception as e:
    # CRITICAL: Print error to console and exit if data fails to load
    print(f"FATAL ERROR: Failed to load NetflixDataset.csv. Application will not start. Error: {e}", file=sys.stderr)
    # The application initialization must still proceed below, but we use an empty dataframe 
    # to avoid crashing the model building steps.
    netflix_data = pd.DataFrame(columns=['Title', 'Languages', 'Genre', 'Tags', 'Actors', 'ViewerRating', 'IMDb Score', 'Image'])


# Proceed with initialization only if data loaded successfully
if not netflix_data.empty:
    netflix_data.index = netflix_data.index.str.title()
    netflix_data = netflix_data[~netflix_data.index.duplicated()]
    netflix_data.rename(columns={'View Rating': 'ViewerRating'}, inplace=True)

    # Extract unique languages and titles for the multi-select dropdowns
    Language = netflix_data.Languages.str.get_dummies(',')
    Lang = Language.columns.str.strip().values.tolist()
    Lang = set(Lang)
    Titles = set(netflix_data.index.to_list())

    # Prepare features for similarity calculation
    netflix_data['Genre'] = netflix_data['Genre'].astype('str')
    netflix_data['Tags'] = netflix_data['Tags'].astype('str')
    # Handle NaN values in IMDb Score
    netflix_data['IMDb Score'] = netflix_data['IMDb Score'].apply(lambda x: 6.6 if math.isnan(x) else x)
    netflix_data['Actors'] = netflix_data['Actors'].astype('str')
    netflix_data['ViewerRating'] = netflix_data['ViewerRating'].astype('str')

    new_features = ['Genre', 'Tags', 'Actors', 'ViewerRating']
    selected_data = netflix_data[new_features].copy() 
    for new_feature in new_features:
        selected_data.loc[:, new_feature] = selected_data.loc[:, new_feature].apply(prepare_data)

    selected_data.index = selected_data.index.str.lower()
    selected_data.index = selected_data.index.str.replace(" ", '')
    selected_data['soup'] = selected_data.apply(create_soup, axis=1)

    # --- Recommendation Model (Count Vectorizer and Cosine Similarity) ---
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(selected_data['soup'])
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
    selected_data.reset_index(inplace=True)
    indices = pd.Series(selected_data.index, index=selected_data['Title'])

else:
    # If data failed to load, initialize defaults to prevent crash
    Lang = set()
    Titles = set()
    cosine_sim2 = None
    indices = pd.Series()
    print("WARNING: Application initialized without data. Only the index page will likely function.", file=sys.stderr)


# Global variables for storing temporary dataframes
result = pd.DataFrame() 
df = pd.DataFrame()

# --- Flask Application Setup ---
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', languages=Lang, titles=Titles)

@app.route('/about', methods=['POST'])
def getvalue():
    if cosine_sim2 is None:
        # Handle case where data failed to load at startup
        return "Error: Recommendation data is unavailable. Check server logs.", 503

    global df
    movienames = request.form.getlist('titles')
    languages = request.form.getlist('languages')
    
    # Clear the results from the previous search
    df = pd.DataFrame() 

    for moviename in movienames:
        get_recommendations(moviename, cosine_sim2)
        
        # Filter results by language preference and concatenate to the final list
        for language in languages:
            # Use .astype(str) to prevent errors if 'Languages' contains non-string types
            df = pd.concat([result[result['Languages'].astype(str).str.contains(language, case=False, na=False)], df], ignore_index=True)

    # Clean and sort final recommendations
    df.drop_duplicates(subset=['Title'], keep='first', inplace=True)
    df.sort_values(by='IMDb Score', ascending=False, inplace=True)
    
    images = df['Image'].tolist()
    titles = df['Title'].tolist()
    
    return render_template('result.html', titles=titles, images=images)

@app.route('/moviepage/<name>')
def movie_details(name):
    global df
    
    # Safely look for the movie in the filtered results
    details_list = df[df['Title'].astype(str) == name].to_numpy().tolist()
    
    if not details_list:
        # Fallback: look in the full dataset
        try:
            # Match the case used in the index
            full_data_row = netflix_data.loc[name.title()] 
            # Convert single row to a list of lists format required by the template
            details_list = full_data_row.to_frame().T.to_numpy().tolist()
        except KeyError:
            # Movie not found in either dataset
            return "Movie details not found.", 404

    return render_template('moviepage.html', details=details_list[0])


if __name__ == '__main__':
    # When running locally, the dev server handles static files and templates differently
    app.run(debug=True)
