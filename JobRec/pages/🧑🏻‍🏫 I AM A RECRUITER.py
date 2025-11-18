# import streamlit as st
# import pandas as pd
# import numpy as np
# import base64
# import os,sys
# import pymongo
# from  JobRecommendation.exception import jobException
# from JobRecommendation.side_logo import add_logo
# from JobRecommendation.sidebar import sidebar
# from JobRecommendation import utils ,MongoDB_function
# from JobRecommendation import text_preprocessing,distance_calculation


# dataBase = "Job-Recomendation"
# collection = "Resume_Data"



# st.set_page_config(layout="wide", page_icon='logo/logo2.png', page_title="RECRUITER")



# add_logo()
# sidebar()


   
# def app():
#     st.title('Candidate Recommendation')
#     c1, c2 = st.columns((3,2))
#     # number of cv recommend slider------------------display##
#     no_of_cv = c2.slider('Number of CV Recommendations:', min_value=0, max_value=6, step=1)
#     jd = c1.text_area("PASTE YOUR JOB DESCRIPTION HERE")
        
#     if len(jd) >=1:


#         NLP_Processed_JD=text_preprocessing.nlp(jd)   # caling (NLP funtion) for text processing

#         jd_df=pd.DataFrame()
#         jd_df['jd']=[' '.join(NLP_Processed_JD)]

#         @st.cache_data
#         def get_recommendation(top, df_all, scores):
#             try:
#                 recommendation = pd.DataFrame(columns = ['name', 'degree',"email",'Unnamed: 0','mobile_number','skills','no_of_pages','score'])
#                 count = 0
#                 for i in top:
                    
                    
#                     recommendation.at[count, 'name'] = df['name'][i]
#                     recommendation.at[count, 'degree'] = df['degree'][i]
#                     recommendation.at[count, 'email'] = df['email'][i]
#                     recommendation.at[count, 'Unnamed: 0'] = df.index[i]
#                     recommendation.at[count, 'mobile_number'] = df['mobile_number'][i]
#                     recommendation.at[count, 'skills'] = df['skills'][i]
#                     recommendation.at[count, 'no_of_pages'] = df['no_of_pages'][i]
#                     recommendation.at[count, 'score'] =  scores[count]
#                     count += 1
#                 return recommendation
#             except Exception as e:
#                 raise jobException(e, sys)



#         df = MongoDB_function.get_collection_as_dataframe(dataBase,collection)

#         cv_data=[]
#         for i in range(len(df["All"])):
#             NLP_Processed_cv=text_preprocessing.nlp(df["All"].values[i])
#             cv_data.append(NLP_Processed_cv)

#         cv_=[]
#         for i in cv_data:
#             cv_.append([' '.join(i)])

#         df["clean_all"]=pd.DataFrame(cv_)



#         # TfidfVectorizer  function

#         tf = distance_calculation.TFIDF(df['clean_all'],jd_df['jd'])   

#         top = sorted(range(len(tf)), key=lambda i: tf[i], reverse=True)[:100]
#         list_scores = [tf[i][0][0] for i in top]
#         TF=get_recommendation(top,df, list_scores)


#         # Count Vectorizer function
#         countv = distance_calculation.count_vectorize(df['clean_all'],jd_df['jd'])
#         top = sorted(range(len(countv)), key=lambda i: countv[i], reverse=True)[:100]
#         list_scores = [countv[i][0][0] for i in top]
#         cv=get_recommendation(top, df, list_scores)

#         # KNN function
        
#         top, index_score = distance_calculation.KNN(df['clean_all'], jd_df['jd'],number_of_neighbors=19)
#         knn = get_recommendation(top, df, index_score)

#         merge1 = knn[['Unnamed: 0','name', 'score']].merge(TF[['Unnamed: 0','score']], on= "Unnamed: 0")
#         final = merge1.merge(cv[['Unnamed: 0','score']], on = 'Unnamed: 0')
#         final = final.rename(columns={"score_x": "KNN", "score_y": "TF-IDF","score": "CV"})

#         # Scale it
#         from sklearn.preprocessing import MinMaxScaler
#         slr = MinMaxScaler()
#         final[["KNN", "TF-IDF", 'CV']] = slr.fit_transform(final[["KNN", "TF-IDF", 'CV']])

#         # Multiply by weights
#         final['KNN'] = (1-final['KNN'])/3
#         final['TF-IDF'] = final['TF-IDF']/3
#         final['CV'] = final['CV']/3
#         final['Final'] = final['KNN']+final['TF-IDF']+final['CV']


#         final =final.sort_values(by="Final", ascending=False)
#         final1 = final.sort_values(by="Final", ascending=False).copy()
#         final_df = df.merge(final1, on='Unnamed: 0')
#         final_df = final_df.sort_values(by="Final", ascending=False)
#         final_df = final_df.reset_index(drop=True)  # reset the index
#         final_df = final_df.head(no_of_cv)
#         #st.dataframe(final_df)
        
        
#         db_expander = st.expander(label='CV recommendations:')
#         with db_expander:        
#             no_of_cols=3
#             cols=st.columns(no_of_cols)
#             for i in range(0, no_of_cv):
#                 cols[i%no_of_cols].text(f"CV ID: {final_df['Unnamed: 0'][i]}")
#                 cols[i%no_of_cols].text(f"Name: {final_df['name_x'][i]}")
#                 cols[i%no_of_cols].text(f"Phone no.: {final_df['mobile_number'][i]}")
#                 cols[i%no_of_cols].text(f"Skills: {final_df['skills'][i]}")
#                 cols[i%no_of_cols].text(f"Degree: {final_df['degree'][i]}")
#                 cols[i%no_of_cols].text(f"No. of Pages Resume: {final_df['no_of_pages'][i]}")
#                 cols[i%no_of_cols].text(f"Email: {final_df['email'][i]}")
#                 encoded_pdf=final_df['pdf_to_base64'][i]
#                 cols[i%no_of_cols].markdown(f'<a href="data:application/octet-stream;base64,{encoded_pdf}" download="resume.pdf"><button style="background-color:GreenYellow;">Download Resume</button></a>', unsafe_allow_html=True)
#                 embed_code = utils.show_pdf(encoded_pdf)
#                 cvID=final1['Unnamed: 0'][i]
#                 show_pdf=cols[i%no_of_cols].button(f"{cvID}.pdf")
#                 if show_pdf:
#                     st.markdown(embed_code, unsafe_allow_html=True)
                
                
                            
            
                
                
#                 cols[i%no_of_cols].text('___________________________________________________')

            
#     else:
#         st.write("<p style='font-size:15px;'>Please Provide The Job Discription </p>",unsafe_allow_html=True)



# if __name__ == '__main__':
#         app()

import streamlit as st
import pandas as pd
import numpy as np
import base64
import os,sys
import pymongo

from JobRecommendation.exception import jobException
from JobRecommendation.side_logo import add_logo
from JobRecommendation.sidebar import sidebar
from JobRecommendation import utils ,MongoDB_function
from JobRecommendation import text_preprocessing,distance_calculation


dataBase = "Job-Recomendation"
collection = "Resume_Data"

st.set_page_config(layout="wide", page_icon='logo/logo2.png', page_title="RECRUITER")
add_logo()
sidebar()


def to_scalar(x):
    arr = np.asarray(x, dtype=float)
    return float(arr.ravel()[0])


def app():
    st.title('Candidate Recommendation')
    c1, c2 = st.columns((3,2))
    no_of_cv = c2.slider('Number of CV Recommendations:', min_value=1, max_value=10, step=1)
    jd = c1.text_area("PASTE YOUR JOB DESCRIPTION HERE")
        
    if len(jd) < 1:
        st.write("<p style='font-size:15px;'>Please Provide The Job Description </p>",unsafe_allow_html=True)
        return
    

    NLP_Processed_JD = text_preprocessing.nlp(jd)
    jd_df = pd.DataFrame({'jd': [' '.join(NLP_Processed_JD)]})


    @st.cache_data
    def get_recommendation(top, df_all, scores):
        try:
            recommendation = pd.DataFrame(columns=['cv_id','name', 'degree',"email",'mobile_number','skills','no_of_pages','score'])
            for idx, i in enumerate(top):
                recommendation.loc[idx] = [
                    df_all['cv_id'][i],
                    df_all['name'][i],
                    df_all['degree'][i],
                    df_all['email'][i],
                    df_all['mobile_number'][i],
                    df_all['skills'][i],
                    df_all['no_of_pages'][i],
                    scores[idx]
                ]
            return recommendation
        except Exception as e:
            raise jobException(e, sys)


    # ✅ Load Resume DB
    df = MongoDB_function.get_collection_as_dataframe(dataBase, collection)

    # ✅ Fix index to use as ID
    if 'cv_id' not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={"index": "cv_id"}, inplace=True)

    if "All" not in df.columns:
        st.error("❌ Column 'All' missing in dataset. Recheck stored resume data.")
        st.stop()

    # ✅ Preprocess Resume Text
    df["clean_all"] = df["All"].apply(lambda x: ' '.join(text_preprocessing.nlp(x)))


    # ✅ TF-IDF Similarity
    tf = distance_calculation.TFIDF(df['clean_all'], jd_df['jd'])
    scores_tf = [to_scalar(tf[i]) for i in range(len(tf))]
    top_tf = sorted(range(len(scores_tf)), key=lambda i: scores_tf[i], reverse=True)[:100]
    TF = get_recommendation(top_tf, df, [scores_tf[i] for i in top_tf])


    # ✅ Count Vector Similarity
    countv = distance_calculation.count_vectorize(df['clean_all'], jd_df['jd'])
    scores_cv = [to_scalar(countv[i]) for i in range(len(countv))]
    top_cv = sorted(range(len(scores_cv)), key=lambda i: scores_cv[i], reverse=True)[:100]
    cv = get_recommendation(top_cv, df, [scores_cv[i] for i in top_cv])


    # ✅ KNN
    top_knn, index_score = distance_calculation.KNN(df['clean_all'], jd_df['jd'], number_of_neighbors=19)
    index_score = [to_scalar(s) for s in index_score]
    knn = get_recommendation(top_knn, df, index_score)


    # ✅ Merge Scores
    final = knn[['cv_id','score']] \
        .merge(TF[['cv_id','score']], on="cv_id", suffixes=("_knn","_tfidf")) \
        .merge(cv[['cv_id','score']], on="cv_id")

    final = final.rename(columns={"score": "cv_score", "score_knn": "KNN", "score_tfidf": "TF-IDF"})


    # ✅ Normalize
    from sklearn.preprocessing import MinMaxScaler
    slr = MinMaxScaler()
    final[["KNN", "TF-IDF", "cv_score"]] = slr.fit_transform(final[["KNN", "TF-IDF", "cv_score"]])

    final['KNN'] = (1 - final['KNN']) / 3
    final['TF-IDF'] = final['TF-IDF'] / 3
    final['CV'] = final['cv_score'] / 3
    final['Final'] = final['KNN'] + final['TF-IDF'] + final['CV']

    final = final.sort_values(by="Final", ascending=False)


    # ✅ Attach to full dataframe
    final_df = df.merge(final, on='cv_id').sort_values(by="Final", ascending=False).reset_index(drop=True)


    # ✅ Display
    available = len(final_df)
    if available == 0:
        st.warning("⚠️ No matching candidates found. Try improving JD.")
        return

    limit = min(no_of_cv, available)
    final_df = final_df.head(limit)


    exp = st.expander(label='CV Recommendations:')
    with exp:
        no_of_cols = 3
        cols = st.columns(no_of_cols)

        for i in range(limit):
            row = final_df.iloc[i]
            col = cols[i % no_of_cols]

            col.text(f"CV ID: {row['cv_id']}")
            col.text(f"Name: {row['name']}")
            col.text(f"Phone: {row['mobile_number']}")
            col.text(f"Skills: {row['skills']}")
            col.text(f"Degree: {row['degree']}")
            col.text(f"Pages: {row['no_of_pages']}")
            col.text(f"Email: {row['email']}")

            encoded_pdf = row['pdf_to_base64']
            col.markdown(
                f'<a href="data:application/octet-stream;base64,{encoded_pdf}" download="resume.pdf">'
                '<button style="background-color:GreenYellow;">Download Resume</button></a>',
                unsafe_allow_html=True
            )

            if col.button(f"View {row['cv_id']}.pdf"):
                st.markdown(utils.show_pdf(encoded_pdf), unsafe_allow_html=True)

            col.text("---------------------------------------------------")


if __name__ == '__main__':
    app()



