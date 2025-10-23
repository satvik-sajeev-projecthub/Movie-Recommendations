import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load datasets
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Step 2: Merge datasets
movies = movies.merge(credits, on='title')

# Step 3: Select relevant columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Step 4: Handle missing data
movies.dropna(inplace=True)

# Step 5: Convert JSON-like strings into lists
import ast

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

def convert_cast(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert_cast)

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

# Step 6: Convert overview to list of words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Step 7: Remove spaces in names
def collapse(L):
    return [i.replace(" ", "") for i in L]

movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)

# Step 8: Create a single column for tags
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags']]

# Step 9: Convert list to string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Step 10: Vectorize the text
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Step 11: Compute similarity
similarity = cosine_similarity(vectors)

# Step 12: Recommendation function
def recommend(movie):
    movie = movie.lower()
    if movie not in new_df['title'].str.lower().values:
        print("❌ Movie not found in the database.")
        return

    index = new_df[new_df['title'].str.lower() == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    print(f"\n🎬 Movies similar to '{new_df.iloc[index].title}':\n")
    for i in distances[1:6]:
        print("➡️", new_df.iloc[i[0]].title)

# Step 13: Test it!
recommend("Avatar")

