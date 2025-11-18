# # 👨‍🎓 I AM A CANDIDATE.py  (updated)
# import streamlit as st
# import pandas as pd
# import numpy as np
# import re
# import plotly.express as px
# import time, datetime
# import base64, random
# import geopy
# from geopy.geocoders import Nominatim
# from geopy.extra.rate_limiter import RateLimiter
# import folium
# from folium.plugins import FastMarkerCluster
# from streamlit_folium import folium_static
# from pyresparser import ResumeParser
# import os, sys
# import pymongo
# from JobRecommendation.animation import load_lottieurl
# from streamlit_lottie import st_lottie, st_lottie_spinner
# from JobRecommendation.side_logo import add_logo
# from JobRecommendation.sidebar import sidebar
# from JobRecommendation import utils, MongoDB_function
# from JobRecommendation import text_preprocessing, distance_calculation
# from JobRecommendation.exception import jobException

# dataBase = "Job-Recomendation"
# collection1 = "preprocessed_jobs_Data"
# collection2 = "Resume_from_CANDIDATE"
# collection3 = "all_locations_Data"

# st.set_page_config(layout="wide", page_icon='logo/logo2.png', page_title="CANDIDATE")

# url = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_x62chJ.json")
# add_logo()
# sidebar()

# def safe_build_all_column(df):
#     """
#     Ensure df has an 'All' column. If missing, construct by concatenating available text fields.
#     """
#     if 'All' in df.columns:
#         return df

#     # candidate source columns (some may not exist)
#     candidate_cols = [
#         'title', 'job highlights', 'job description', 'company overview',
#         'industry', 'description', 'positionName', 'company'
#     ]

#     # normalize columns to lower for matching
#     lower_map = {c.lower(): c for c in df.columns}
#     chosen = []
#     for c in candidate_cols:
#         if c in lower_map:
#             chosen.append(lower_map[c])
#     # if nothing matched, try to use every object dtype/text-like column
#     if not chosen:
#         text_cols = [c for c, dt in df.dtypes.items() if dt == object]
#         chosen = text_cols

#     if not chosen:
#         # cannot build meaningful 'All' column
#         df['All'] = ""  # keep as empty strings to avoid crashes later
#         return df

#     def join_row_fields(row):
#         parts = []
#         for col in chosen:
#             val = row.get(col, "")
#             if pd.notna(val):
#                 parts.append(str(val))
#         return " ".join(parts)

#     df['All'] = df.apply(join_row_fields, axis=1)
#     return df

# def app():
#     st.title('Job Recommendation')
#     c1, c2 = st.columns((3,2))

#     cv = c1.file_uploader('Upload your CV', type='pdf')

#     # load locations (with CSV fallback)
#     import os
#     csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "concatenated_data", "all_locations.csv"))

#     job_loc = MongoDB_function.get_collection_as_dataframe(dataBase, collection3)
#     if job_loc is None or not hasattr(job_loc, "columns"):
#         st.warning("DB did not return a DataFrame. Falling back to CSV: " + csv_path)
#         if os.path.exists(csv_path):
#             job_loc = pd.read_csv(csv_path)
#         else:
#             job_loc = pd.DataFrame(columns=["location"])

#     if "location" not in job_loc.columns:
#         st.warning("'location' column missing in DB/CSV. No locations available.")
#         all_locations = []
#     else:
#         all_locations = list(job_loc["location"].dropna().unique())

#     RL = c2.multiselect('Filter', all_locations )

#     no_of_jobs = st.slider('Number of Job Recommendations:', min_value=1, max_value=100, step=10)

#     if cv is not None:
#         if st.button('Proceed Further !! '):
#             with st_lottie_spinner(url, key="download", reverse=True, speed=1, loop=True, quality='high'):
#                 time.sleep(3)
#                 try:
#                     cv_text = utils.extract_data(cv)  # OCR

#                     if not cv_text or cv_text.strip() == "":
#                         st.error("Your resume text could not be extracted. Please upload a clearer PDF or ensure the resume has readable text.")
#                         return
                    
#                     cv_text_list = [cv_text]

#                     encoded_pdf = utils.pdf_to_base64(cv)
#                     resume_data = ResumeParser(cv).get_extracted_data()
#                     resume_data["pdf_to_base64"] = encoded_pdf

#                     timestamp = utils.generateUniqueFileName()
#                     save = {timestamp: resume_data}
#                     MongoDB_function.resume_store(save, dataBase, collection2)

#                     try:
#                         NLP_Processed_CV = text_preprocessing.nlp(cv_text)
#                     except Exception as e:
#                         st.error('Error during CV NLP processing: ' + str(e))
#                         NLP_Processed_CV = []

#                     # put CV's keywords into dataframe (single-row)
#                     # df2 = pd.DataFrame({
#                     #     'title': ["I"],
#                     #     'job highlights': ["I"],
#                     #     'job description': ["I"],
#                     #     'company overview': ["I"],
#                     #     'industry': ["I"]
#                     # })
#                     # df2['All'] = " ".join(NLP_Processed_CV)

#                     # df2 = pd.DataFrame({'All': [" ".join(NLP_Processed_CV)]})

#                     cv_combined = " ".join(NLP_Processed_CV).strip()
#                     if not cv_combined:
#                         st.error("No meaningful text extracted from resume after NLP preprocessing. Please upload a clearer resume.")
#                         return
                    
#                     df2 = pd.DataFrame({'All': [cv_combined]})


#                     # load jobs dataset
#                     df = MongoDB_function.get_collection_as_dataframe(dataBase, collection1)
#                     # fallback: try to read from CSV in package if DB fails (optional)
#                     if df is None or not hasattr(df, "columns"):
#                         st.warning("Jobs DB collection returned nothing. Please check DB or provide fallback CSV if available.")
#                         df = pd.DataFrame()  # empty but safe

#                     # Ensure critical columns exist and build 'All' if missing
#                     df = safe_build_all_column(df)

#                     if df is None or not hasattr(df, "columns") or df.shape[0] == 0:
#                         st.error("No job data available. Jobs DB collection returned empty. Please check DB or provide a jobs CSV fallback.")
#                         return

#                         # Ensure 'All' column exists and has at least one non-empty row
#                     df = safe_build_all_column(df)
#                     if 'All' not in df.columns or df['All'].dropna().astype(str).str.strip().eq('').all():
#                         st.error("Job text ('All' column) is empty for all jobs. Please check scraped job data.")
#                         return

#                     # Check that 'All' exists and is non-empty for vectorizers
#                     # if 'All' not in df.columns:
#                     #     raise jobException("'All' column missing and could not be constructed from job data.", sys)

#                     # Recommendation functions
#                     @st.cache_data
#                     def get_recommendation(top, df_all, scores):
#                         try:
#                             recommendation = pd.DataFrame(columns=['positionName', 'company', "location", 'JobID', 'description', 'score'])
#                             count = 0
#                             for i in top:
#                                 recommendation.at[count, 'positionName'] = df_all.loc[i, 'positionName'] if 'positionName' in df_all.columns else df_all.loc[i, 'title'] if 'title' in df_all.columns else "Not Provided"
#                                 recommendation.at[count, 'company'] = df_all.loc[i, 'company'] if 'company' in df_all.columns else "Not Provided"
#                                 recommendation.at[count, 'location'] = df_all.loc[i, 'location'] if 'location' in df_all.columns else "Not Provided"
#                                 recommendation.at[count, 'JobID'] = df_all.index[i]
#                                 recommendation.at[count, 'description'] = df_all.loc[i, 'description'] if 'description' in df_all.columns else ""
#                                 recommendation.at[count, 'score'] = scores[count]
#                                 count += 1
#                             return recommendation
#                         except Exception as e:
#                             raise jobException(e, sys)

#                     # run similarity functions (they expect pd.Series inputs)
#                     output2 = distance_calculation.TFIDF(df['All'], df2['All'])
#                     top = sorted(range(len(output2)), key=lambda i: float(np.squeeze(output2[i])), reverse=True)[:1000]
#                     # list_scores = [output2[i][0][0] for i in top]
#                     list_scores = [float(np.squeeze(output2[i])) for i in top]
#                     TF = get_recommendation(top, df, list_scores)

#                     print(np.array(output2).shape)
#                     print(np.array(output3).shape)

#                     output3 = distance_calculation.count_vectorize(df['All'], df2['All'])
#                     top = sorted(range(len(output3)), key=lambda i: float(np.squeeze(output3[i])), reverse=True)[:1000]
#                     list_scores = [float(np.squeeze(output3[i])) for i in top]
#                     cv = get_recommendation(top, df, list_scores)

#                     top, index_score = distance_calculation.KNN(df['All'], df2['All'], number_of_neighbors=100)
#                     knn = get_recommendation(top, df, index_score)

#                     # Combine results
#                     merge1 = knn[['JobID', 'positionName', 'score']].merge(TF[['JobID', 'score']], on="JobID")
#                     final = merge1.merge(cv[['JobID', 'score']], on="JobID")
#                     final = final.rename(columns={"score_x": "KNN", "score_y": "TF-IDF", "score": "CV"})
#                     from sklearn.preprocessing import MinMaxScaler
#                     slr = MinMaxScaler()
#                     final[["KNN", "TF-IDF", 'CV']] = slr.fit_transform(final[["KNN", "TF-IDF", 'CV']])

#                     final['KNN'] = (1 - final['KNN']) / 3
#                     final['TF-IDF'] = final['TF-IDF'] / 3
#                     final['CV'] = final['CV'] / 3
#                     final['Final'] = final['KNN'] + final['TF-IDF'] + final['CV']

#                     final2 = final.sort_values(by="Final", ascending=False).copy()
#                     final_df = df.merge(final2, on="JobID")
#                     final_df = final_df.sort_values(by="Final", ascending=False)
#                     final_df.fillna('Not Available', inplace=True)

#                     result_jd = final_df
#                     if len(RL) != 0:
#                         result_jd = result_jd[result_jd["location"].isin(list(RL))]

#                     final_jobrecomm = result_jd.head(no_of_jobs)

#                     # Visualization and mapping
#                     df3 = final_jobrecomm.copy()
#                     rec_loc = df3.location.value_counts()
#                     locations_df = pd.DataFrame(rec_loc)
#                     locations_df.reset_index(inplace=True)
#                     locations_df['index'] = locations_df['index'].apply(lambda x: x.replace("Area", "") if isinstance(x, str) and "Area" in x else x)

#                     locator = Nominatim(user_agent="myGeocoder")
#                     geocode = RateLimiter(locator.geocode, min_delay_seconds=1)

#                     # Attempt geocoding but catch exceptions so app doesn't fail
#                     locations_df['loc_geo'] = locations_df['index'].apply(lambda x: geocode(x) if pd.notna(x) else None)
#                     locations_df['point'] = locations_df['loc_geo'].apply(lambda loc: tuple(loc.point) if loc else None)
#                     locations_df[['latitude', 'longitude', 'altitude']] = pd.DataFrame(locations_df['point'].tolist(), index=locations_df.index)
#                     locations_df.dropna(subset=['latitude', 'longitude'], inplace=True)

#                     folium_map = folium.Map(location=[12.9767936, 77.590082], zoom_start=11, tiles="openstreetmap")
#                     for lat, lon, ind, job_no in zip(locations_df['latitude'], locations_df['longitude'], locations_df['index'], locations_df[locations_df.columns[1]]):
#                         label = folium.Popup("Area: " + str(ind) + "<br> Number of Jobs: " + str(job_no), max_width=500)
#                         folium.CircleMarker([lat, lon], radius=10, popup=label, fill=True, icon_size=(150, 150)).add_to(folium_map)

#                     db_expander = st.expander(label='CV dashboard:')
#                     with db_expander:
#                         available_locations = df3.location.value_counts().sum()
#                         all_locations = df3.location.value_counts().sum() + df3.location.isnull().sum()
#                         st.write(" **JOB LOCATIONS FROM**", available_locations, "**OF**", all_locations, "**JOBS**")
#                         folium_static(folium_map, width=1380)

#                         chart2, chart3, chart1 = st.columns(3)
#                         with chart3:
#                             st.write("<p style='font-size:17px;font-family: Verdana, sans-serif'> RATINGS W.R.T Company</p>", unsafe_allow_html=True)
#                             rating_count = final_jobrecomm[["rating", "company"]]
#                             fig = px.pie(rating_count, values="rating", names="company", width=600)
#                             fig.update_layout(showlegend=True)
#                             st.plotly_chart(fig, use_container_width=True)

#                         with chart2:
#                             st.write("<p style='font-size:17px;font-family: Verdana, sans-serif'> REVIEWS COUNT W.R.T Company</p>", unsafe_allow_html=True)
#                             review_count = final_jobrecomm[["reviewsCount", "company"]]
#                             fig = px.pie(review_count, values="reviewsCount", names="company", width=600)
#                             fig.update_layout(showlegend=True)
#                             st.plotly_chart(fig, use_container_width=True)

#                         with chart1:
#                             final_salary = final_jobrecomm.copy()
#                             col = final_salary["salary"].dropna().to_list()
#                             y, m = utils.get_monthly_yearly_salary(col)
#                             yearly_salary_range = utils.salary_converter(y)
#                             monthly_salary_to_yearly = utils.salary_converter(m)
#                             final_salary_vals = yearly_salary_range + monthly_salary_to_yearly
#                             salary_df = pd.DataFrame(final_salary_vals, columns=['Salary Range'])
#                             sal_count = salary_df['Salary Range'].count()
#                             st.write(" **SALARY RANGE FROM** ", sal_count, "**SALARY VALUES PROVIDED**")
#                             fig2 = px.box(salary_df, y="Salary Range", width=500, title="Salary Range For The Given Job Profile")
#                             fig2.update_yaxes(showticklabels=True, title="Salary Range in Rupees")
#                             fig2.update_xaxes(visible=True, showticklabels=True)
#                             st.write(fig2)

#                     # Job recommendations table
#                     db_expander = st.expander(label='Job Recommendations:')
#                     final_jobrecomm = final_jobrecomm.replace(np.nan, "Not Provided")

#                     @st.cache_data
#                     def make_clickable(link):
#                         text = 'more details'
#                         return f'<a target="_blank" href="{link}">{text}</a>'

#                     with db_expander:
#                         def convert_df(df_in):
#                             try:
#                                 return df_in.to_csv(index=False).encode('utf-8')
#                             except Exception as e:
#                                 raise jobException(e, sys)

#                         if 'externalApplyLink' in final_jobrecomm.columns:
#                             final_jobrecomm['externalApplyLink'] = final_jobrecomm['externalApplyLink'].apply(lambda x: make_clickable(x) if pd.notna(x) and x != "Not Provided" else "Not Provided")
#                         if 'url' in final_jobrecomm.columns:
#                             final_jobrecomm['url'] = final_jobrecomm['url'].apply(lambda x: make_clickable(x) if pd.notna(x) and x != "Not Provided" else "Not Provided")

#                         final_df = final_jobrecomm[['company', 'positionName_x' if 'positionName_x' in final_jobrecomm.columns else 'positionName',
#                                                     'description', 'location', 'salary', 'rating', 'reviewsCount', 'externalApplyLink', 'url']].copy()
#                         final_df.rename({'company': 'Company', 'positionName_x': 'Position Name', 'positionName': 'Position Name',
#                                          'description': 'Job Description', 'location': 'Location', 'salary': 'Salary',
#                                          'rating': 'Company Rating', 'reviewsCount': 'Company ReviewCount',
#                                          'externalApplyLink': 'Web Apply Link', 'url': 'Indeed Apply Link'}, axis=1, inplace=True)

#                         show_df = final_df.to_html(escape=False)
#                         st.write(show_df, unsafe_allow_html=True)

#                     csv = convert_df(final_df)
#                     st.download_button("Press to Download", csv, "file.csv", "text/csv", key='download-csv')
#                     st.balloons()

#                 except Exception as e:
#                     # Show a helpful error and raise a jobException so your outer error-handling can still catch it
#                     st.error("Error occurred: " + str(e))
#                     raise jobException(e, sys)

# if __name__ == '__main__':
#     app()

# 👨‍🎓 I AM A CANDIDATE.py  (final corrected)
import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import time, datetime
import base64, random
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
from folium.plugins import FastMarkerCluster
from streamlit_folium import folium_static
from pyresparser import ResumeParser
import os, sys
import pymongo
from JobRecommendation.animation import load_lottieurl
from streamlit_lottie import st_lottie, st_lottie_spinner
from JobRecommendation.side_logo import add_logo
from JobRecommendation.sidebar import sidebar
from JobRecommendation import utils, MongoDB_function
from JobRecommendation import text_preprocessing, distance_calculation
from JobRecommendation.exception import jobException

# Basic config
dataBase = "Job-Recomendation"
collection1 = "preprocessed_jobs_Data"
collection2 = "Resume_from_CANDIDATE"
collection3 = "all_locations_Data"

st.set_page_config(layout="wide", page_icon='logo/logo2.png', page_title="CANDIDATE")

url = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_x62chJ.json")
add_logo()
sidebar()


def safe_build_all_column(df):
    """
    Ensure df has an 'All' column. If missing, construct by concatenating available text fields.
    """
    if df is None or not hasattr(df, "columns"):
        return pd.DataFrame()

    if 'All' in df.columns:
        return df

    candidate_cols = [
        'title', 'job highlights', 'job description', 'company overview',
        'industry', 'description', 'positionName', 'company'
    ]

    lower_map = {c.lower(): c for c in df.columns}
    chosen = []
    for c in candidate_cols:
        if c in lower_map:
            chosen.append(lower_map[c])

    if not chosen:
        text_cols = [c for c, dt in df.dtypes.items() if dt == object]
        chosen = text_cols

    if not chosen:
        df['All'] = ""
        return df

    def join_row_fields(row):
        parts = []
        for col in chosen:
            try:
                val = row.get(col, "")
            except Exception:
                val = row[col] if col in row else ""
            if pd.notna(val):
                parts.append(str(val))
        return " ".join(parts)

    df['All'] = df.apply(join_row_fields, axis=1)
    return df


def app():
    st.title('Job Recommendation')
    c1, c2 = st.columns((3, 2))

    cv = c1.file_uploader('Upload your CV', type='pdf')

    # load locations (with CSV fallback)
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "concatenated_data", "all_locations.csv"))

    job_loc = MongoDB_function.get_collection_as_dataframe(dataBase, collection3)
    if job_loc is None or not hasattr(job_loc, "columns"):
        st.warning("DB did not return a DataFrame. Falling back to CSV: " + csv_path)
        if os.path.exists(csv_path):
            job_loc = pd.read_csv(csv_path)
        else:
            job_loc = pd.DataFrame(columns=["location"])

    if "location" not in job_loc.columns:
        st.warning("'location' column missing in DB/CSV. No locations available.")
        all_locations = []
    else:
        all_locations = list(job_loc["location"].dropna().unique())

    RL = c2.multiselect('Filter', all_locations)

    no_of_jobs = st.slider('Number of Job Recommendations:', min_value=1, max_value=100, step=10)

    if cv is not None:
        if st.button('Proceed Further !! '):
            with st_lottie_spinner(url, key="download", reverse=True, speed=1, loop=True, quality='high'):
                time.sleep(1.2)
                try:
                    # --- Extract resume text ---
                    cv_text = utils.extract_data(cv)  # OCR
                    if not cv_text or cv_text.strip() == "":
                        st.error("Your resume text could not be extracted. Please upload a clearer PDF or ensure the resume has readable text.")
                        return

                    encoded_pdf = utils.pdf_to_base64(cv)
                    try:
                        resume_data = ResumeParser(cv).get_extracted_data()
                    except Exception:
                        resume_data = {}
                    resume_data["pdf_to_base64"] = encoded_pdf

                    timestamp = utils.generateUniqueFileName()
                    save = {timestamp: resume_data}
                    MongoDB_function.resume_store(save, dataBase, collection2)

                    # NLP process the CV to extract keywords/text
                    try:
                        NLP_Processed_CV = text_preprocessing.nlp(cv_text)
                    except Exception as e:
                        st.error('Error during CV NLP processing: ' + str(e))
                        NLP_Processed_CV = []

                    cv_combined = " ".join(NLP_Processed_CV).strip()
                    if not cv_combined:
                        st.error("No meaningful text extracted from resume after NLP preprocessing. Please upload a clearer resume.")
                        return
                    df2 = pd.DataFrame({'All': [cv_combined]})

                    # --- Load jobs dataset ---
                    df = MongoDB_function.get_collection_as_dataframe(dataBase, collection1)
                    if df is None or not hasattr(df, "columns") or df.shape[0] == 0:
                        st.error("No job data available. Jobs DB collection returned empty. Please check DB or provide a jobs CSV fallback.")
                        return

                    df = safe_build_all_column(df)
                    if 'All' not in df.columns or df['All'].dropna().astype(str).str.strip().eq('').all():
                        st.error("Job text ('All' column) is empty for all jobs. Please check scraped job data.")
                        return

                    # --- Helper: create recommendation DataFrame ---
                    @st.cache_data
                    def get_recommendation(top, df_all, scores):
                        try:
                            recommendation = pd.DataFrame(columns=['positionName', 'company', "location", 'JobID', 'description', 'score'])
                            count = 0
                            for idx, i in enumerate(top):
                                # ensure i is within df_all index range
                                if i < 0 or i >= df_all.shape[0]:
                                    continue
                                recommendation.at[count, 'positionName'] = df_all.iloc[i]['positionName'] if 'positionName' in df_all.columns else (df_all.iloc[i]['title'] if 'title' in df_all.columns else "Not Provided")
                                recommendation.at[count, 'company'] = df_all.iloc[i]['company'] if 'company' in df_all.columns else "Not Provided"
                                recommendation.at[count, 'location'] = df_all.iloc[i]['location'] if 'location' in df_all.columns else "Not Provided"
                                recommendation.at[count, 'JobID'] = df_all.index[i]
                                recommendation.at[count, 'description'] = df_all.iloc[i]['description'] if 'description' in df_all.columns else ""
                                # score may be missing if lengths mismatch, guard it
                                try:
                                    recommendation.at[count, 'score'] = float(scores[count])
                                except Exception:
                                    # fallback: try to take score by index i if available
                                    try:
                                        recommendation.at[count, 'score'] = float(scores[idx])
                                    except Exception:
                                        recommendation.at[count, 'score'] = 0.0
                                count += 1
                            return recommendation
                        except Exception as e:
                            raise jobException(e, sys)

                    # --- Run similarity functions and ensure they return 1-D score arrays when single CV is passed ---
                    # TF-IDF
                    output2 = distance_calculation.TFIDF(df['All'], df2['All'])
                    # output2 should be either:
                    #  - 1D array of length n_jobs (recommended), or
                    #  - 2D array (n_cv, n_jobs). If 1 CV passed, we take row 0.
                    out2_arr = np.asarray(output2)
                    if out2_arr.ndim == 2 and out2_arr.shape[0] == 1:
                        scores_tfidf = out2_arr[0]
                    elif out2_arr.ndim == 1:
                        scores_tfidf = out2_arr
                    else:
                        # If it's 2D with multiple rows, take the mean across queries as aggregate
                        scores_tfidf = out2_arr.mean(axis=0)

                    # Defensive: ensure shape matches number of jobs
                    n_jobs = df.shape[0]
                    if scores_tfidf.shape[0] != n_jobs:
                        # Try to flatten any extra dims, or set zeros
                        try:
                            scores_tfidf = np.asarray(scores_tfidf).reshape(n_jobs)
                        except Exception:
                            scores_tfidf = np.zeros(n_jobs)

                    # sort top indices by score
                    top = sorted(range(len(scores_tfidf)), key=lambda i: float(scores_tfidf[i]), reverse=True)[:1000]
                    list_scores = [float(scores_tfidf[i]) for i in top]
                    TF = get_recommendation(top, df, list_scores)

                    # Count Vectorizer
                    output3 = distance_calculation.count_vectorize(df['All'], df2['All'])
                    out3_arr = np.asarray(output3)
                    if out3_arr.ndim == 2 and out3_arr.shape[0] == 1:
                        scores_count = out3_arr[0]
                    elif out3_arr.ndim == 1:
                        scores_count = out3_arr
                    else:
                        scores_count = out3_arr.mean(axis=0)

                    if scores_count.shape[0] != n_jobs:
                        try:
                            scores_count = np.asarray(scores_count).reshape(n_jobs)
                        except Exception:
                            scores_count = np.zeros(n_jobs)

                    top = sorted(range(len(scores_count)), key=lambda i: float(scores_count[i]), reverse=True)[:1000]
                    list_scores = [float(scores_count[i]) for i in top]
                    cv = get_recommendation(top, df, list_scores)

                    # KNN (this returns indices and distances — already handled in distance module)
                    top_knn, index_score = distance_calculation.KNN(df['All'], df2['All'], number_of_neighbors=100)
                    # KNN may return numpy arrays; ensure they're lists/1D
                    if isinstance(top_knn, np.ndarray):
                        top_knn = top_knn.tolist()
                    if isinstance(index_score, np.ndarray):
                        index_score = index_score.tolist()
                    knn = get_recommendation(top_knn, df, index_score)

                    # --- Combine results ---
                    # Ensure the 'JobID' column exists for merges:
                    # In get_recommendation we set JobID as df_all.index[i], so merges should work.
                    # Keep only top overlaps to merge safely.
                    merge1 = knn[['JobID', 'positionName', 'score']].merge(TF[['JobID', 'score']], on="JobID")
                    final = merge1.merge(cv[['JobID', 'score']], on="JobID")
                    final = final.rename(columns={"score_x": "KNN", "score_y": "TF-IDF", "score": "CV"})

                    from sklearn.preprocessing import MinMaxScaler
                    slr = MinMaxScaler()
                    # Protect scaling step if final has less than 1 row
                    if final.shape[0] == 0:
                        st.warning("No overlapping recommendations found from similarity methods.")
                        return

                    # Ensure numeric columns exist before scaling
                    for col in ["KNN", "TF-IDF", "CV"]:
                        if col not in final.columns:
                            final[col] = 0.0

                    final[["KNN", "TF-IDF", 'CV']] = slr.fit_transform(final[["KNN", "TF-IDF", 'CV']])

                    final['KNN'] = (1 - final['KNN']) / 3
                    final['TF-IDF'] = final['TF-IDF'] / 3
                    final['CV'] = final['CV'] / 3
                    final['Final'] = final['KNN'] + final['TF-IDF'] + final['CV']

                    final2 = final.sort_values(by="Final", ascending=False).copy()
                    final_df = df.merge(final2, on="JobID")
                    final_df = final_df.sort_values(by="Final", ascending=False)
                    final_df.fillna('Not Available', inplace=True)

                    result_jd = final_df
                    if len(RL) != 0:
                        result_jd = result_jd[result_jd["location"].isin(list(RL))]

                    final_jobrecomm = result_jd.head(no_of_jobs)

                    # --- Visualization and mapping ---
                    df3 = final_jobrecomm.copy()
                    rec_loc = df3.location.value_counts()
                    locations_df = pd.DataFrame(rec_loc)
                    locations_df.reset_index(inplace=True)
                    locations_df['index'] = locations_df['index'].apply(lambda x: x.replace("Area", "") if isinstance(x, str) and "Area" in x else x)

                    locator = Nominatim(user_agent="myGeocoder")
                    geocode = RateLimiter(locator.geocode, min_delay_seconds=1)

                    # Attempt geocoding but catch exceptions so app doesn't fail
                    locations_df['loc_geo'] = locations_df['index'].apply(lambda x: geocode(x) if pd.notna(x) else None)
                    locations_df['point'] = locations_df['loc_geo'].apply(lambda loc: tuple(loc.point) if loc else None)
                    if 'point' in locations_df.columns:
                        locations_df[['latitude', 'longitude', 'altitude']] = pd.DataFrame(locations_df['point'].tolist(), index=locations_df.index)
                        locations_df.dropna(subset=['latitude', 'longitude'], inplace=True)
                    else:
                        locations_df[['latitude', 'longitude', 'altitude']] = pd.DataFrame(columns=['latitude', 'longitude', 'altitude'])

                    folium_map = folium.Map(location=[12.9767936, 77.590082], zoom_start=11, tiles="openstreetmap")
                    if not locations_df.empty and 'latitude' in locations_df.columns:
                        for lat, lon, ind, job_no in zip(locations_df['latitude'], locations_df['longitude'], locations_df['index'], locations_df[locations_df.columns[1]]):
                            label = folium.Popup("Area: " + str(ind) + "<br> Number of Jobs: " + str(job_no), max_width=500)
                            folium.CircleMarker([lat, lon], radius=10, popup=label, fill=True, icon_size=(150, 150)).add_to(folium_map)

                    db_expander = st.expander(label='CV dashboard:')
                    with db_expander:
                        available_locations = df3.location.value_counts().sum() if 'location' in df3.columns else 0
                        all_locations = available_locations + (df3.location.isnull().sum() if 'location' in df3.columns else 0)
                        st.write(" **JOB LOCATIONS FROM**", available_locations, "**OF**", all_locations, "**JOBS**")
                        folium_static(folium_map, width=1380)

                        chart2, chart3, chart1 = st.columns(3)
                        with chart3:
                            st.write("<p style='font-size:17px;font-family: Verdana, sans-serif'> RATINGS W.R.T Company</p>", unsafe_allow_html=True)
                            rating_count = final_jobrecomm[["rating", "company"]] if {'rating', 'company'}.issubset(final_jobrecomm.columns) else pd.DataFrame(columns=['rating','company'])
                            if not rating_count.empty:
                                fig = px.pie(rating_count, values="rating", names="company", width=600)
                                fig.update_layout(showlegend=True)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.write("No rating data available")

                        with chart2:
                            st.write("<p style='font-size:17px;font-family: Verdana, sans-serif'> REVIEWS COUNT W.R.T Company</p>", unsafe_allow_html=True)
                            review_count = final_jobrecomm[["reviewsCount", "company"]] if {'reviewsCount', 'company'}.issubset(final_jobrecomm.columns) else pd.DataFrame(columns=['reviewsCount','company'])
                            if not review_count.empty:
                                fig = px.pie(review_count, values="reviewsCount", names="company", width=600)
                                fig.update_layout(showlegend=True)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.write("No reviewsCount data available")

                        with chart1:
                            final_salary = final_jobrecomm.copy()
                            col = final_salary["salary"].dropna().to_list() if 'salary' in final_salary.columns else []
                            y, m = utils.get_monthly_yearly_salary(col)
                            yearly_salary_range = utils.salary_converter(y)
                            monthly_salary_to_yearly = utils.salary_converter(m)
                            final_salary_vals = yearly_salary_range + monthly_salary_to_yearly
                            salary_df = pd.DataFrame(final_salary_vals, columns=['Salary Range']) if final_salary_vals else pd.DataFrame(columns=['Salary Range'])
                            sal_count = salary_df['Salary Range'].count() if not salary_df.empty else 0
                            st.write(" **SALARY RANGE FROM** ", sal_count, "**SALARY VALUES PROVIDED**")
                            if not salary_df.empty:
                                fig2 = px.box(salary_df, y="Salary Range", width=500, title="Salary Range For The Given Job Profile")
                                fig2.update_yaxes(showticklabels=True, title="Salary Range in Rupees")
                                fig2.update_xaxes(visible=True, showticklabels=True)
                                st.write(fig2)
                            else:
                                st.write("No salary data available")

                    # --- Job recommendations table ---
                    db_expander = st.expander(label='Job Recommendations:')
                    final_jobrecomm = final_jobrecomm.replace(np.nan, "Not Provided") if 'final_jobrecomm' in locals() else pd.DataFrame()

                    @st.cache_data
                    def make_clickable(link):
                        text = 'more details'
                        return f'<a target="_blank" href="{link}">{text}</a>'

                    with db_expander:
                        def convert_df(df_in):
                            try:
                                return df_in.to_csv(index=False).encode('utf-8')
                            except Exception as e:
                                raise jobException(e, sys)

                        if not final_jobrecomm.empty:
                            if 'externalApplyLink' in final_jobrecomm.columns:
                                final_jobrecomm['externalApplyLink'] = final_jobrecomm['externalApplyLink'].apply(lambda x: make_clickable(x) if pd.notna(x) and x != "Not Provided" else "Not Provided")
                            if 'url' in final_jobrecomm.columns:
                                final_jobrecomm['url'] = final_jobrecomm['url'].apply(lambda x: make_clickable(x) if pd.notna(x) and x != "Not Provided" else "Not Provided")

                            final_df = final_jobrecomm[['company', 'positionName_x' if 'positionName_x' in final_jobrecomm.columns else 'positionName',
                                                        'description', 'location', 'salary', 'rating', 'reviewsCount', 'externalApplyLink', 'url']].copy()
                            final_df.rename({'company': 'Company', 'positionName_x': 'Position Name', 'positionName': 'Position Name',
                                             'description': 'Job Description', 'location': 'Location', 'salary': 'Salary',
                                             'rating': 'Company Rating', 'reviewsCount': 'Company ReviewCount',
                                             'externalApplyLink': 'Web Apply Link', 'url': 'Indeed Apply Link'}, axis=1, inplace=True)

                            show_df = final_df.to_html(escape=False)
                            st.write(show_df, unsafe_allow_html=True)

                            csv = convert_df(final_df)
                            st.download_button("Press to Download", csv, "file.csv", "text/csv", key='download-csv')
                        else:
                            st.write("No job recommendations to show.")

                    st.balloons()

                except Exception as e:
                    st.error("Error occurred: " + str(e))
                    raise jobException(e, sys)


if __name__ == '__main__':
    app()
