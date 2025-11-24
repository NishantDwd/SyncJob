import streamlit as st
import pandas as pd
import numpy as np
import base64
import os,sys
import pymongo

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('words', quiet=True)

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
    """Convert any array-like to scalar float"""
    try:
        arr = np.asarray(x, dtype=float)
        return float(arr.ravel()[0])
    except:
        return float(x)


def app():
    st.title('🧑‍💼 AI-Powered Candidate Recommendation')
    st.markdown("Get the best matching candidates for your job description using advanced ML algorithms")
    
    c1, c2 = st.columns((3,2))
    no_of_cv = c2.slider('Number of CV Recommendations:', min_value=1, max_value=20, step=1, value=10)
    jd = c1.text_area("📝 PASTE YOUR JOB DESCRIPTION HERE", height=300)
        
    if len(jd) < 10:
        st.info("👆 Please provide a detailed job description to get candidate recommendations")
        st.markdown("""
        ### Sample Job Description Format:
        ```
        Job Title: Senior Data Scientist
        
        Requirements:
        - 5+ years of experience in Machine Learning
        - Strong Python, TensorFlow, PyTorch skills
        - Experience with NLP and Computer Vision
        - PhD or Masters in Computer Science/Statistics
        
        Responsibilities:
        - Build and deploy ML models
        - Lead data science projects
        - Mentor junior team members
        ```
        """)
        return
    
    if st.button('🔍 Find Best Candidates'):
        with st.spinner('🤖 Analyzing candidates using AI...'):
            try:
                # Process Job Description
                NLP_Processed_JD = text_preprocessing.nlp(jd)
                if not NLP_Processed_JD or len(NLP_Processed_JD) == 0:
                    st.error("❌ Could not process job description. Please provide a clearer description.")
                    return
                    
                jd_df = pd.DataFrame({'jd': [' '.join(NLP_Processed_JD)]})

                # Helper function to create recommendation DataFrame
                def get_recommendation(top_indices, df_all, scores):
                    """Create recommendation dataframe from top matching indices"""
                    try:
                        recommendation_data = []
                        
                        for idx, i in enumerate(top_indices):
                            if i >= len(df_all):
                                continue
                                
                            rec_dict = {
                                'cv_id': df_all.iloc[i]['cv_id'] if 'cv_id' in df_all.columns else i,
                                'name': df_all.iloc[i]['name'] if 'name' in df_all.columns else 'N/A',
                                'degree': df_all.iloc[i]['degree'] if 'degree' in df_all.columns else 'N/A',
                                'email': df_all.iloc[i]['email'] if 'email' in df_all.columns else 'N/A',
                                'mobile_number': df_all.iloc[i]['mobile_number'] if 'mobile_number' in df_all.columns else 'N/A',
                                'skills': df_all.iloc[i]['skills'] if 'skills' in df_all.columns else [],
                                'no_of_pages': df_all.iloc[i]['no_of_pages'] if 'no_of_pages' in df_all.columns else 0,
                                'score': float(scores[idx]) if idx < len(scores) else 0.0
                            }
                            recommendation_data.append(rec_dict)
                        
                        return pd.DataFrame(recommendation_data)
                    except Exception as e:
                        st.error(f"Error creating recommendations: {str(e)}")
                        raise jobException(e, sys)

                # Load Resume Database
                df = MongoDB_function.get_collection_as_dataframe(dataBase, collection)
                
                if df is None or df.empty:
                    st.error("❌ No resume data found in database. Please upload resumes first.")
                    st.info("💡 Use the data_dump3.py script to load resume data into MongoDB")
                    return

                st.success(f"✅ Loaded {len(df)} candidate resumes from database")

                # Create cv_id if not exists
                if 'cv_id' not in df.columns:
                    if 'Unnamed: 0' in df.columns:
                        df['cv_id'] = df['Unnamed: 0']
                    else:
                        df.reset_index(inplace=True)
                        df.rename(columns={"index": "cv_id"}, inplace=True)

                # Check for 'All' column (resume text)
                if "All" not in df.columns:
                    st.error("❌ Column 'All' missing in resume dataset. Please check data format.")
                    st.info("The 'All' column should contain the full resume text")
                    return

                # Check if we have valid resume text
                valid_resumes = df['All'].notna() & (df['All'].astype(str).str.strip() != '')
                if not valid_resumes.any():
                    st.error("❌ No valid resume text found in database")
                    return

                # Filter to only valid resumes
                df = df[valid_resumes].copy()
                df.reset_index(drop=True, inplace=True)
                st.info(f"📊 Processing {len(df)} valid resumes...")

                # Preprocess Resume Text
                st.write("🔄 Step 1/4: Preprocessing resume text...")
                clean_all_list = []
                for idx, resume_text in enumerate(df["All"]):
                    try:
                        processed = text_preprocessing.nlp(str(resume_text))
                        clean_all_list.append(' '.join(processed))
                    except:
                        clean_all_list.append('')
                
                df["clean_all"] = clean_all_list
                
                # Remove any resumes that became empty after cleaning
                df = df[df["clean_all"].str.strip() != ''].copy()
                df.reset_index(drop=True, inplace=True)
                
                if len(df) == 0:
                    st.error("❌ No valid resumes after preprocessing")
                    return

                st.write(f"✅ Preprocessed {len(df)} resumes")

                # TF-IDF Similarity
                st.write("🔄 Step 2/4: Calculating TF-IDF similarity...")
                tf = distance_calculation.TFIDF(df['clean_all'], jd_df['jd'])
                scores_tf = [to_scalar(tf[i]) for i in range(len(tf))]
                
                # Get top matches (up to 100 or all available)
                num_candidates = min(100, len(df))
                top_tf = sorted(range(len(scores_tf)), key=lambda i: scores_tf[i], reverse=True)[:num_candidates]
                TF = get_recommendation(top_tf, df, [scores_tf[i] for i in top_tf])

                # Count Vector Similarity
                st.write("🔄 Step 3/4: Calculating Count Vector similarity...")
                countv = distance_calculation.count_vectorize(df['clean_all'], jd_df['jd'])
                scores_cv = [to_scalar(countv[i]) for i in range(len(countv))]
                top_cv = sorted(range(len(scores_cv)), key=lambda i: scores_cv[i], reverse=True)[:num_candidates]
                CV = get_recommendation(top_cv, df, [scores_cv[i] for i in top_cv])

                # KNN Similarity
                st.write("🔄 Step 4/4: Calculating KNN similarity...")
                # Adjust neighbors based on dataset size
                n_neighbors = min(19, len(df) - 1)
                if n_neighbors < 1:
                    n_neighbors = 1
                    
                top_knn, index_score = distance_calculation.KNN(df['clean_all'], jd_df['jd'], number_of_neighbors=n_neighbors)
                index_score = [to_scalar(s) for s in index_score]
                KNN = get_recommendation(top_knn, df, index_score)

                st.success("✅ All similarity calculations completed!")

                # Merge Scores
                st.write("🔄 Combining results from all algorithms...")
                
                # Merge all three methods
                try:
                    final = KNN[['cv_id','score']].copy()
                    final.rename(columns={'score': 'score_knn'}, inplace=True)
                    
                    # Merge TF-IDF
                    tf_merge = TF[['cv_id','score']].copy()
                    tf_merge.rename(columns={'score': 'score_tfidf'}, inplace=True)
                    final = final.merge(tf_merge, on="cv_id", how='outer')
                    
                    # Merge Count Vector
                    cv_merge = CV[['cv_id','score']].copy()
                    cv_merge.rename(columns={'score': 'score_cv'}, inplace=True)
                    final = final.merge(cv_merge, on="cv_id", how='outer')
                    
                    # Fill NaN with 0
                    final.fillna(0, inplace=True)
                    
                except Exception as e:
                    st.error(f"Error merging results: {str(e)}")
                    return

                # Rename for clarity
                final.rename(columns={
                    "score_knn": "KNN",
                    "score_tfidf": "TF-IDF",
                    "score_cv": "CV"
                }, inplace=True)

                # Normalize scores
                from sklearn.preprocessing import MinMaxScaler
                slr = MinMaxScaler()
                
                # Check if we have valid data to scale
                if len(final) == 0:
                    st.error("❌ No matching candidates found")
                    return
                
                final[["KNN", "TF-IDF", "CV"]] = slr.fit_transform(final[["KNN", "TF-IDF", "CV"]])

                # Calculate weighted final score
                # KNN: inverted distance (lower is better), so we use (1-score)
                final['KNN'] = (1 - final['KNN']) / 3
                final['TF-IDF'] = final['TF-IDF'] / 3
                final['CV'] = final['CV'] / 3
                final['Final'] = final['KNN'] + final['TF-IDF'] + final['CV']

                # Sort by final score
                final = final.sort_values(by="Final", ascending=False)

                # Merge with original dataframe to get full details
                final_df = df.merge(final, on='cv_id', how='inner')
                final_df = final_df.sort_values(by="Final", ascending=False).reset_index(drop=True)

                if len(final_df) == 0:
                    st.warning("⚠️ No matching candidates found. Try a different job description.")
                    return

                st.success(f"🎯 Found {len(final_df)} matching candidates!")

                # Get top N candidates
                limit = min(no_of_cv, len(final_df))
                final_df = final_df.head(limit)

                # Display Results
                st.markdown("---")
                st.header(f"🏆 Top {limit} Recommended Candidates")
                
                # Show score distribution
                st.subheader("📊 Match Score Distribution")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Best Match", f"{final_df['Final'].iloc[0]:.2%}")
                with col2:
                    st.metric("Average Match", f"{final_df['Final'].mean():.2%}")
                with col3:
                    st.metric("Candidates Found", len(final_df))

                # Detailed candidate cards
                exp = st.expander(label=f'📋 View All {limit} Candidate Details', expanded=True)
                with exp:
                    no_of_cols = 3
                    cols = st.columns(no_of_cols)

                    for i in range(limit):
                        row = final_df.iloc[i]
                        col = cols[i % no_of_cols]
                        
                        # Card styling
                        match_score = row['Final']
                        if match_score > 0.7:
                            badge_color = "#1ed760"
                            badge_text = "Excellent Match"
                        elif match_score > 0.5:
                            badge_color = "#fba171"
                            badge_text = "Good Match"
                        else:
                            badge_color = "#fabc10"
                            badge_text = "Fair Match"
                        
                        col.markdown(f"""
                        <div style='padding: 15px; border: 2px solid {badge_color}; border-radius: 10px; margin-bottom: 10px;'>
                            <h3 style='color: {badge_color}; margin: 0;'>Candidate #{i+1}</h3>
                            <p style='color: {badge_color}; font-size: 14px; margin: 5px 0;'>{badge_text}: {match_score:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col.text(f"🆔 CV ID: {row['cv_id']}")
                        col.text(f"👤 Name: {row['name']}")
                        col.text(f"📧 Email: {row['email']}")
                        col.text(f"📱 Phone: {row['mobile_number']}")
                        col.text(f"🎓 Degree: {row['degree']}")
                        col.text(f"📄 Pages: {row['no_of_pages']}")
                        
                        # Display skills in a better format
                        if 'skills' in row and row['skills']:
                            skills_str = str(row['skills'])
                            if len(skills_str) > 100:
                                skills_str = skills_str[:100] + "..."
                            col.text(f"💼 Skills: {skills_str}")
                        
                        # Show algorithm scores
                        col.markdown(f"""
                        **Algorithm Scores:**
                        - KNN: {row['KNN']*3:.2%}
                        - TF-IDF: {row['TF-IDF']*3:.2%}
                        - Count Vector: {row['CV']*3:.2%}
                        """)
                        
                        # Download resume button
                        if 'pdf_to_base64' in row and pd.notna(row['pdf_to_base64']):
                            encoded_pdf = row['pdf_to_base64']
                            col.markdown(
                                f'<a href="data:application/octet-stream;base64,{encoded_pdf}" download="resume_{row["cv_id"]}.pdf">'
                                '<button style="background-color:#1ed760; color:white; padding:8px 16px; border:none; border-radius:5px; cursor:pointer;">📥 Download Resume</button></a>',
                                unsafe_allow_html=True
                            )
                            
                            if col.button(f"👁️ Preview Resume #{i+1}", key=f"view_{i}"):
                                st.markdown(utils.show_pdf(encoded_pdf), unsafe_allow_html=True)
                        
                        col.markdown("---")

                # Export options
                st.markdown("---")
                st.subheader("📥 Export Results")
                
                # Prepare export dataframe
                export_df = final_df[['cv_id', 'name', 'email', 'mobile_number', 'degree', 'skills', 'Final', 'KNN', 'TF-IDF', 'CV']].copy()
                export_df['Final'] = export_df['Final'].apply(lambda x: f"{x:.2%}")
                export_df['KNN'] = export_df['KNN'].apply(lambda x: f"{x*3:.2%}")
                export_df['TF-IDF'] = export_df['TF-IDF'].apply(lambda x: f"{x*3:.2%}")
                export_df['CV'] = export_df['CV'].apply(lambda x: f"{x*3:.2%}")
                
                export_df.rename(columns={
                    'cv_id': 'CV ID',
                    'name': 'Name',
                    'email': 'Email',
                    'mobile_number': 'Phone',
                    'degree': 'Degree',
                    'skills': 'Skills',
                    'Final': 'Match Score',
                    'KNN': 'KNN Score',
                    'TF-IDF': 'TF-IDF Score',
                    'CV': 'Count Vector Score'
                }, inplace=True)
                
                # Download as CSV
                csv = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Download Results as CSV",
                    data=csv,
                    file_name=f"candidate_recommendations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
                
                st.balloons()

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                import traceback
                with st.expander("🔍 Error Details (for debugging)"):
                    st.code(traceback.format_exc())
                raise jobException(e, sys)


if __name__ == '__main__':
    app()