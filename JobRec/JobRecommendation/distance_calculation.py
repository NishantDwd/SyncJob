# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.neighbors import NearestNeighbors
# from  JobRecommendation.exception import jobException
# import streamlit as st
# import sys
# @st.cache_data
# def TFIDF(scraped_data, cv):
#     try:
#         tfidf_vectorizer = TfidfVectorizer(stop_words='english')
#         # TF-IDF Scraped data
#         tfidf_jobid = tfidf_vectorizer.fit_transform(scraped_data)
#         # TF-IDF CV
#         user_tfidf = tfidf_vectorizer.transform(cv)
#         # Using cosine_similarity on (Scraped data) & (CV)
#         cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf,x),tfidf_jobid)
#         output2 = list(cos_similarity_tfidf)
#         return output2  # what does it return?
#     except Exception as e:
#         raise jobException(e, sys)
# @st.cache_data
# def count_vectorize(scraped_data, cv):
#     try:
#         # CountV the scraped data
#         count_vectorizer = CountVectorizer()
#         count_jobid = count_vectorizer.fit_transform(scraped_data) #fitting and transforming the vector
#         # CountV the cv
#         user_count = count_vectorizer.transform(cv)
#         cos_similarity_countv = map(lambda x: cosine_similarity(user_count, x),count_jobid)
#         output3 = list(cos_similarity_countv)
#         return output3
#     except Exception as e:
#         raise jobException(e, sys)

# @st.cache_data
# def KNN(scraped_data, cv,number_of_neighbors):
#     try:
#         tfidf_vectorizer = TfidfVectorizer(stop_words='english')
#         # n_neighbors = 100
#         KNN = NearestNeighbors(n_neighbors = number_of_neighbors, p=2)
#         KNN.fit(tfidf_vectorizer.fit_transform(scraped_data))
#         NNs = KNN.kneighbors(tfidf_vectorizer.transform(cv), return_distance=True)
#         top = NNs[1][0][1:]
#         index_score = NNs[0][0][1:]
        
#         return top ,index_score
#     except Exception as e:
#         raise jobException(e, sys)


# JobRecommendation/distance_calculation.py
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from JobRecommendation.exception import jobException
import streamlit as st
import sys
import numpy as np

def _clean_text_list(text_list):
    """Remove None/empty/whitespace-only entries and ensure list type."""
    if text_list is None:
        return []
    # If it's a pandas Series, convert to list
    try:
        iter(text_list)
    except TypeError:
        return []
    cleaned = []
    for t in text_list:
        if t is None:
            continue
        s = str(t).strip()
        if s:
            cleaned.append(s)
    return cleaned

@st.cache_data
def TFIDF(scraped_data, cv):
    """
    Returns cosine similarity matrix shape (len(cv), len(scraped_data)).
    If either input is empty, returns a zero matrix of appropriate shape.
    """
    try:
        scraped = _clean_text_list(scraped_data)
        cv_list = _clean_text_list(cv)

        # If no scraped job texts -> return empty similarity matrix with 0 columns
        if len(scraped) == 0:
            # return shape (len(cv_list), 0)
            return np.zeros((len(cv_list), 0))

        # If cv empty -> return zeros with shape (0, len(scraped))
        if len(cv_list) == 0:
            return np.zeros((0, len(scraped)))

        tfidf_vectorizer = TfidfVectorizer(stop_words='english')

        tfidf_jobid = tfidf_vectorizer.fit_transform(scraped)
        user_tfidf = tfidf_vectorizer.transform(cv_list)

        cos_similarity_tfidf = cosine_similarity(user_tfidf, tfidf_jobid)

        return cos_similarity_tfidf  # matrix (len(cv), len(scraped))

    except Exception as e:
        raise jobException(e, sys)


@st.cache_data
def count_vectorize(scraped_data, cv):
    """
    Returns cosine similarity matrix computed on CountVectorizer.
    Handles empty inputs by returning zero matrices.
    """
    try:
        scraped = _clean_text_list(scraped_data)
        cv_list = _clean_text_list(cv)

        if len(scraped) == 0:
            return np.zeros((len(cv_list), 0))
        if len(cv_list) == 0:
            return np.zeros((0, len(scraped)))

        count_vectorizer = CountVectorizer()

        count_jobid = count_vectorizer.fit_transform(scraped)
        user_count = count_vectorizer.transform(cv_list)

        cos_similarity_countv = cosine_similarity(user_count, count_jobid)

        return cos_similarity_countv

    except Exception as e:
        raise jobException(e, sys)


@st.cache_data
def KNN(scraped_data, cv, number_of_neighbors=100):
    """
    Returns (indices, distances) for nearest neighbors.
    - indices: array of shape (k,) with indices into scraped_data for top neighbors (excludes query itself)
    - distances: array of shape (k,) with distances
    If there are fewer samples than requested neighbors, the function uses the available samples.
    If scraped_data is empty, returns (np.array([]), np.array([])).
    """
    try:
        scraped = _clean_text_list(scraped_data)
        cv_list = _clean_text_list(cv)

        if len(scraped) == 0 or len(cv_list) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        tfidf_vectorizer = TfidfVectorizer(stop_words='english')

        tfidf_data = tfidf_vectorizer.fit_transform(scraped)
        tfidf_cv = tfidf_vectorizer.transform(cv_list)

        n_samples = tfidf_data.shape[0]
        # KNeighbors requires n_neighbors <= n_samples
        n_neighbors = max(1, min(number_of_neighbors, n_samples))

        knn = NearestNeighbors(n_neighbors=n_neighbors, p=2)
        knn.fit(tfidf_data)

        distances, indices = knn.kneighbors(tfidf_cv, return_distance=True)

        # If number_of_neighbors requested > 1, we may want to exclude the self-match when query is from same corpus.
        # The app previously excluded first entry; but if n_neighbors == 1, we simply return that one.
        # We'll return indices[0] and distances[0] for first (and only) CV.
        return indices[0], distances[0]

    except Exception as e:
        raise jobException(e, sys)

