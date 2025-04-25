import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import re

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.write("Search for movies by title, actor, or director!")

# Load data function
@st.cache_data
def load_data():
    try:
        data_movie = pd.read_csv('tmdb_5000_movies.csv')
        data_credit = pd.read_csv('tmdb_5000_credits.csv')
        
        # Merge the datasets
        data = data_movie.merge(data_credit)
        
        # Drop rows with missing values
        data.dropna(inplace=True)
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Process data function
@st.cache_data
def process_data(data):
    try:
        # Helper function to convert JSON string to list of names
        def convert(feature):
            if isinstance(feature, list):
                return feature
            L = []
            for i in ast.literal_eval(feature):
                L.append(i['name'])
            return L
        
        # Extract up to 3 cast members
        def cast_convert(feature):
            if isinstance(feature, list):
                return feature
            L = []
            c = 0
            for i in ast.literal_eval(feature):
                if c < 3:
                    L.append(i["name"])
                    c += 1
                else:
                    break
            return L
        
        # Extract directors
        def fetch_director(feature):
            if isinstance(feature, list):
                return feature
            L = []
            for i in ast.literal_eval(feature):
                if i['job'] == 'Director':
                    L.append(i['name'])
            return L
        
        # Remove spaces from names
        def collapse(L):
            L1 = []
            for i in L:
                L1.append(i.replace(" ", ""))
            return L1
        
        # Apply transformations
        data['keywords'] = data['keywords'].apply(convert)
        data['cast'] = data['cast'].apply(cast_convert)
        data['crew'] = data['crew'].apply(fetch_director)
        data['cast'] = data['cast'].apply(collapse)
        data['crew'] = data['crew'].apply(collapse)
        data['genres'] = data['genres'].apply(convert)
        data['genres'] = data['genres'].apply(collapse)
        data['keywords'] = data['keywords'].apply(collapse)
        data['overview'] = data['overview'].apply(lambda x: x.split())
        
        # Create tags for content-based filtering
        data['tags'] = data['overview'] + data['genres'] + data['keywords']
        
        # Create new dataframe with selected columns
        new_df = data[['movie_id', 'title', 'cast', 'crew', 'tags']]
        new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
        
        # Create lowercase version of title for searching
        new_df['title_lower'] = new_df['title'].str.lower()
        
        return new_df
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return None

# Create similarity matrix function
@st.cache_data
def create_similarity_matrix(df):
    try:
        # Create count matrix from tags
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vector = cv.fit_transform(df['tags']).toarray()
        
        # Calculate cosine similarity
        similarity = cosine_similarity(vector)
        
        return similarity
    except Exception as e:
        st.error(f"Error creating similarity matrix: {e}")
        return None

# Recommendation function
def recommend(query, df, similarity):
    try:
        # Extract keywords from the query using simpler patterns
        if "acted by" in query.lower():
            name = query.lower().split("acted by")[1].strip()
            search_by = 'cast'
        elif "movies of" in query.lower():
            name = query.lower().split("movies of")[1].strip()
            search_by = 'crew'
        elif "movie" in query.lower() and not ("acted by" in query.lower() or "movies of" in query.lower()):
            name = query.lower().split("movie")[1].strip()
            search_by = 'title'
        else:
            name = query.lower()
            search_by = 'title'

        if search_by == 'title':
            # Simple title search
            matches = df[df['title_lower'].str.contains(name, na=False)]
        elif search_by == 'cast':
            # Safer cast search with error handling
            matches = df[df['cast'].apply(lambda x:
                isinstance(x, list) and
                any(name in str(i).lower() for i in x) if isinstance(x, list) else False
            )]
        elif search_by == 'crew':
            # Safer crew search with error handling
            matches = df[df['crew'].apply(lambda x:
                isinstance(x, list) and
                any(name in str(i).lower() for i in x) if isinstance(x, list) else False
            )]

        if len(matches) == 0:
            return None

        index = matches.index[0]

        # Get recommendations
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        
        # Return top 5 recommendations
        recommendations = [(df.iloc[i[0]].title, round(i[1] * 100, 2)) for i in distances[1:6]]
        return recommendations, matches.iloc[0]
    
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return None, None

# Main function
def main():
    # Add sidebar with information
    st.sidebar.title("About")
    st.sidebar.info(
        "This app recommends movies based on content similarity. "
        "You can search by movie title, actor, or director."
    )
    
    st.sidebar.title("Search Examples")
    st.sidebar.markdown(
        """
        - **Movie Title**: "Avatar", "The Dark Knight"
        - **Actor**: "Movies acted by KeanuReeves"
        - **Director**: "Movies of ChristopherNolan"
        """
    )
    
    # Load and process data
    with st.spinner("Loading movie data..."):
        data = load_data()
        if data is not None:
            df = process_data(data)
            if df is not None:
                similarity = create_similarity_matrix(df)
                if similarity is not None:
                    st.success("Data loaded successfully!")
                else:
                    st.error("Failed to create similarity matrix")
                    return
            else:
                st.error("Failed to process data")
                return
        else:
            st.error("Failed to load data")
            return
    
    # Create search box
    query = st.text_input("What kind of movies are you looking for?", placeholder="e.g., 'Avatar' or 'Movies acted by KeanuReeves'")
    
    # Create search button
    search_button = st.button("Search")
    
    # Show recommendations if search is clicked
    if search_button and query:
        with st.spinner("Searching for recommendations..."):
            results, match = recommend(query, df, similarity)
            
            if results:
                # Display the matched movie
                st.subheader(f"Matched: {match['title']}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Cast:**")
                    cast_names = [name for name in match['cast']] if isinstance(match['cast'], list) else []
                    st.write(", ".join([re.sub(r'([a-z])([A-Z])', r'\1 \2', name) for name in cast_names]))
                
                with col2:
                    st.write("**Director:**")
                    director_names = [name for name in match['crew']] if isinstance(match['crew'], list) else []
                    st.write(", ".join([re.sub(r'([a-z])([A-Z])', r'\1 \2', name) for name in director_names]))
                
                # Display recommendations
                st.subheader("Recommendations for you:")
                
                # Display as cards in columns
                cols = st.columns(5)
                for i, (title, score) in enumerate(results):
                    with cols[i]:
                        st.markdown(f"""
                        <div style="
                            padding: 10px; 
                            border-radius: 5px; 
                            border: 1px solid #ddd;
                            min-height: 150px;
                            text-align: center;
                            display: flex;
                            flex-direction: column;
                            justify-content: space-between;
                        ">
                            <h4 style="margin-top: 0;">{title}</h4>
                            <div style="margin-top: auto;">
                                <div style="background-color: {'#5cb85c' if score > 70 else '#f0ad4e'}; color: white; padding: 3px 8px; border-radius: 10px; display: inline-block;">
                                    {score}% match
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning(f"No movies found matching '{query}'. Try a different search term.")

if __name__ == "__main__":
    main()
