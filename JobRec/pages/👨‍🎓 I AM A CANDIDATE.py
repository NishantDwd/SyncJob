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

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('words', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

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


# Custom CSS for modern table and modal
st.markdown("""
<style>
    /* Modern Table Styles */
    .job-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-radius: 8px;
        overflow: hidden;
    }
    
    .job-table thead {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .job-table th {
        padding: 15px;
        text-align: left;
        font-weight: 600;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .job-table td {
        padding: 12px 15px;
        border-bottom: 1px solid #e0e0e0;
        font-size: 14px;
    }
    
    .job-table tbody tr {
        transition: all 0.3s ease;
        background-color: white;
    }
    
    .job-table tbody tr:hover {
        background-color: #f5f7ff;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .job-table tbody tr:last-child td {
        border-bottom: none;
    }
    
    /* Match Score Badge */
    .match-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 12px;
        color: white;
    }
    
    .match-excellent { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .match-good { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .match-fair { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
    .match-low { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color: #333; }
    
    /* View Details Button */
    .view-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 12px;
        font-weight: 600;
        text-decoration: none;
        display: inline-block;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    .view-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    
    /* Apply Link Button */
    .apply-btn {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: none;
        padding: 6px 12px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 11px;
        font-weight: 600;
        text-decoration: none;
        display: inline-block;
        margin-right: 5px;
        transition: all 0.3s ease;
    }
    
    .apply-btn:hover {
        transform: scale(1.05);
        box-shadow: 0 3px 8px rgba(0,0,0,0.2);
    }
    
    /* Company Rating */
    .rating-box {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 8px;
        background: #fff3cd;
        border-radius: 4px;
        font-weight: 600;
        color: #856404;
    }
    
    /* Location Badge */
    .location-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        color: #666;
        font-size: 13px;
    }
    
    /* Salary Badge */
    .salary-badge {
        display: inline-block;
        padding: 4px 10px;
        background: #d4edda;
        color: #155724;
        border-radius: 4px;
        font-weight: 600;
        font-size: 12px;
    }
    
    /* Job Card Alternative */
    .job-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border-left: 4px solid #667eea;
    }
    
    .job-card:hover {
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        transform: translateY(-4px);
    }
    
    .job-card-header {
        display: flex;
        justify-content: space-between;
        align-items: start;
        margin-bottom: 15px;
    }
    
    .job-title {
        font-size: 18px;
        font-weight: 700;
        color: #2c3e50;
        margin: 0 0 5px 0;
    }
    
    .company-name {
        font-size: 16px;
        color: #667eea;
        font-weight: 600;
    }
    
    .job-details {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        margin: 15px 0;
    }
    
    .detail-item {
        display: flex;
        align-items: center;
        gap: 5px;
        color: #666;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


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
            val = row.get(col, "")
            if pd.notna(val):
                parts.append(str(val))
        return " ".join(parts)

    df['All'] = df.apply(join_row_fields, axis=1)
    return df


def format_job_description(description):
    """Format job description with proper structure"""
    if pd.isna(description) or description == "Not Provided" or description == "Not Available":
        return "<p>No description available</p>"
    
    # Clean and structure the description
    desc = str(description)
    
    # Split by common section headers
    sections = []
    keywords = ['responsibilities:', 'requirements:', 'qualifications:', 'skills:', 
                'experience:', 'about', 'duties:', 'benefits:', 'what you']
    
    lines = desc.split('\n')
    current_section = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        is_header = any(keyword in line.lower() for keyword in keywords)
        
        if is_header and current_section:
            sections.append('<br>'.join(current_section))
            current_section = [f"<strong>{line}</strong>"]
        elif is_header:
            current_section = [f"<strong>{line}</strong>"]
        else:
            # Format bullet points
            if line.startswith('-') or line.startswith('•'):
                current_section.append(f"&nbsp;&nbsp;• {line[1:].strip()}")
            else:
                current_section.append(line)
    
    if current_section:
        sections.append('<br>'.join(current_section))
    
    # If no sections found, just split by paragraphs
    if not sections:
        sections = [line for line in lines if line.strip()]
    
    formatted = '<br><br>'.join(sections)
    
    # Limit length for display
    if len(formatted) > 2000:
        formatted = formatted[:2000] + "..."
    
    return formatted


def create_job_cards_view(df):
    """Create modern card-based view for jobs"""
    for idx, row in df.iterrows():
        # Calculate match score badge
        match_score = row.get('Final', 0)
        if match_score >= 0.75:
            badge_class = "match-excellent"
            badge_text = "Excellent Match"
        elif match_score >= 0.60:
            badge_class = "match-good"
            badge_text = "Good Match"
        elif match_score >= 0.40:
            badge_class = "match-fair"
            badge_text = "Fair Match"
        else:
            badge_class = "match-low"
            badge_text = "Low Match"
        
        # Build job details HTML
        details_html = []
        
        # Location
        location = row.get('location', 'Not Specified')
        if location and location != 'Not Available':
            details_html.append(f'<div class="detail-item"><span>📍</span><span>{location}</span></div>')
        
        # Salary
        salary = row.get('salary', '')
        if pd.notna(salary) and salary not in ['Not Provided', 'Not Disclosed', 'Not Available', '']:
            details_html.append(f'<div class="detail-item"><span>💰</span><span class="salary-badge">{salary}</span></div>')
        
        # Rating
        rating = row.get('rating', '')
        reviews = row.get('reviewsCount', 0)
        if pd.notna(rating) and rating not in ['Not Available', '']:
            details_html.append(f'<div class="detail-item"><span>⭐</span><span class="rating-box">{rating} ({reviews} reviews)</span></div>')
        
        details_str = ''.join(details_html)
        
        # Job card HTML
        card_html = f"""
        <div class="job-card">
            <div class="job-card-header">
                <div>
                    <h3 class="job-title">{row.get('positionName', row.get('positionName_x', 'Position Not Specified'))}</h3>
                    <p class="company-name">{row.get('company', 'Company Not Specified')}</p>
                </div>
                <span class="match-badge {badge_class}">{badge_text}<br>{match_score:.1%}</span>
            </div>
            <div class="job-details">
                {details_str}
            </div>
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)
        
        # Expandable section for job description
        with st.expander(f"📄 View Full Job Description & Apply", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("### Job Description")
                description = format_job_description(row.get('description', 'Not Provided'))
                st.markdown(f'<div style="text-align: justify; line-height: 1.6;">{description}</div>', 
                           unsafe_allow_html=True)
            
            with col2:
                st.markdown("### Quick Actions")
                
                # Apply buttons
                external_link = row.get('externalApplyLink', '')
                if pd.notna(external_link) and external_link not in ['Not Provided', 'Not Available', '']:
                    st.markdown(f'<a href="{external_link}" target="_blank" class="view-btn" style="display: block; text-align: center; margin: 10px 0;">🌐 Apply on Company Site</a>', 
                               unsafe_allow_html=True)
                
                url_link = row.get('url', '')
                if pd.notna(url_link) and url_link not in ['Not Provided', 'Not Available', '']:
                    st.markdown(f'<a href="{url_link}" target="_blank" class="view-btn" style="display: block; text-align: center; margin: 10px 0;">📝 Apply on Indeed</a>', 
                               unsafe_allow_html=True)
                
                # Additional info
                st.markdown("---")
                st.markdown(f"**Job ID:** {row.get('JobID', 'N/A')}")
                st.markdown(f"**Match Scores:**")
                st.markdown(f"- KNN: {row.get('KNN', 0)*3:.1%}")
                st.markdown(f"- TF-IDF: {row.get('TF-IDF', 0)*3:.1%}")
                st.markdown(f"- Count Vec: {row.get('CV', 0)*3:.1%}")


def create_table_view(df):
    """Create clean table view with action buttons"""
    # Prepare display dataframe
    display_df = df.copy()
    
    # Select and rename columns
    column_mapping = {
        'company': 'Company',
        'positionName': 'Position',
        'positionName_x': 'Position',
        'location': 'Location',
        'salary': 'Salary',
        'rating': 'Rating',
        'Final': 'Match Score'
    }
    
    # Keep only relevant columns that exist
    available_cols = []
    for old_col, new_col in column_mapping.items():
        if old_col in display_df.columns:
            available_cols.append(old_col)
            if old_col != new_col.lower():
                display_df.rename(columns={old_col: new_col}, inplace=True)
    
    # Format match score
    if 'Match Score' in display_df.columns:
        display_df['Match Score'] = display_df['Match Score'].apply(lambda x: f"{x:.1%}")
    
    # Create HTML table
    html_table = '<table class="job-table"><thead><tr>'
    
    # Table headers
    headers = ['#', 'Company', 'Position', 'Location', 'Salary', 'Rating', 'Match Score', 'Actions']
    for header in headers:
        html_table += f'<th>{header}</th>'
    html_table += '</tr></thead><tbody>'
    
    # Table rows
    for idx, row in display_df.head(50).iterrows():  # Limit to 50 for performance
        match_score = row.get('Match Score', '0%')
        match_val = float(match_score.strip('%')) / 100 if isinstance(match_score, str) else match_score
        
        if match_val >= 0.75:
            badge_class = "match-excellent"
        elif match_val >= 0.60:
            badge_class = "match-good"
        elif match_val >= 0.40:
            badge_class = "match-fair"
        else:
            badge_class = "match-low"
        
        html_table += '<tr>'
        html_table += f'<td>{idx + 1}</td>'
        html_table += f'<td><strong>{row.get("Company", "N/A")}</strong></td>'
        html_table += f'<td>{row.get("Position", "N/A")}</td>'
        html_table += f'<td><span class="location-badge">📍 {row.get("Location", "N/A")}</span></td>'
        
        # Salary with proper escaping
        salary = row.get('Salary', 'Not Disclosed')
        if salary != "Not Disclosed" and pd.notna(salary):
            salary_badge = '<span class="salary-badge">' + str(salary) + '</span>'
            html_table += f'<td>{salary_badge}</td>'
        else:
            html_table += '<td>Not Disclosed</td>'
        
        # Rating with proper escaping
        rating = row.get('Rating', 'N/A')
        if pd.notna(rating) and rating != "N/A":
            rating_badge = '<span class="rating-box">⭐ ' + str(rating) + '</span>'
            html_table += f'<td>{rating_badge}</td>'
        else:
            html_table += '<td>N/A</td>'
        
        html_table += f'<td><span class="match-badge {badge_class}">{match_score}</span></td>'
        html_table += '<td><button class="view-btn" onclick="alert(\'Use the expandable cards below to view full details\')">View Details</button></td>'
        html_table += '</tr>'
    
    html_table += '</tbody></table>'
    
    st.markdown(html_table, unsafe_allow_html=True)


def app():
    st.title('🎯 AI-Powered Job Recommendation')
    st.markdown("Upload your resume and discover personalized job matches using advanced machine learning")
    
    c1, c2 = st.columns((3, 2))

    cv = c1.file_uploader('📄 Upload your CV (PDF)', type='pdf')

    # Load locations
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "concatenated_data", "all_locations.csv"))

    job_loc = MongoDB_function.get_collection_as_dataframe(dataBase, collection3)
    if job_loc is None or not hasattr(job_loc, "columns"):
        if os.path.exists(csv_path):
            job_loc = pd.read_csv(csv_path)
        else:
            job_loc = pd.DataFrame(columns=["location"])

    if "location" not in job_loc.columns:
        all_locations = []
    else:
        all_locations = list(job_loc["location"].dropna().unique())

    RL = c2.multiselect('🗺️ Filter by Location (Optional)', all_locations)
    no_of_jobs = st.slider('📊 Number of Job Recommendations:', min_value=5, max_value=100, step=5, value=20)

    if cv is not None:
        if st.button('🚀 Find My Perfect Jobs', type="primary"):
            with st_lottie_spinner(url, key="download", reverse=True, speed=1, loop=True, quality='high'):
                time.sleep(1.2)
                try:
                    # Extract resume text
                    cv_text = utils.extract_data(cv)
                    if not cv_text or cv_text.strip() == "":
                        st.error("❌ Could not extract text from your resume. Please upload a clearer PDF.")
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

                    # NLP processing
                    try:
                        NLP_Processed_CV = text_preprocessing.nlp(cv_text)
                    except Exception as e:
                        st.error(f'❌ Error during CV processing: {str(e)}')
                        NLP_Processed_CV = []

                    cv_combined = " ".join(NLP_Processed_CV).strip()
                    if not cv_combined:
                        st.error("❌ No meaningful text extracted from resume. Please upload a clearer resume.")
                        return
                    
                    df2 = pd.DataFrame({'All': [cv_combined]})

                    # Load jobs dataset
                    df = MongoDB_function.get_collection_as_dataframe(dataBase, collection1)
                    if df is None or not hasattr(df, "columns") or df.shape[0] == 0:
                        st.error("❌ No job data available. Please contact administrator.")
                        return

                    df = safe_build_all_column(df)
                    if 'All' not in df.columns or df['All'].dropna().astype(str).str.strip().eq('').all():
                        st.error("❌ Job database is empty. Please check data source.")
                        return

                    # Progress indicator
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Recommendation function
                    @st.cache_data
                    def get_recommendation(top, df_all, scores):
                        try:
                            recommendation = pd.DataFrame(columns=['positionName', 'company', "location", 'JobID', 'description', 'score'])
                            count = 0
                            for idx, i in enumerate(top):
                                if i < 0 or i >= df_all.shape[0]:
                                    continue
                                recommendation.at[count, 'positionName'] = df_all.iloc[i]['positionName'] if 'positionName' in df_all.columns else (df_all.iloc[i]['title'] if 'title' in df_all.columns else "Not Provided")
                                recommendation.at[count, 'company'] = df_all.iloc[i]['company'] if 'company' in df_all.columns else "Not Provided"
                                recommendation.at[count, 'location'] = df_all.iloc[i]['location'] if 'location' in df_all.columns else "Not Provided"
                                recommendation.at[count, 'JobID'] = df_all.index[i]
                                recommendation.at[count, 'description'] = df_all.iloc[i]['description'] if 'description' in df_all.columns else ""
                                try:
                                    recommendation.at[count, 'score'] = float(scores[count])
                                except Exception:
                                    try:
                                        recommendation.at[count, 'score'] = float(scores[idx])
                                    except Exception:
                                        recommendation.at[count, 'score'] = 0.0
                                count += 1
                            return recommendation
                        except Exception as e:
                            raise jobException(e, sys)

                    # TF-IDF Similarity
                    status_text.text("🔄 Calculating TF-IDF similarity...")
                    progress_bar.progress(25)
                    
                    output2 = distance_calculation.TFIDF(df['All'], df2['All'])
                    n_jobs = df.shape[0]
                    
                    out2_arr = np.asarray(output2)
                    if out2_arr.ndim == 2 and out2_arr.shape[0] == 1:
                        scores_tfidf = out2_arr[0]
                    elif out2_arr.ndim == 1:
                        scores_tfidf = out2_arr
                    else:
                        scores_tfidf = out2_arr.mean(axis=0)

                    if scores_tfidf.shape[0] != n_jobs:
                        try:
                            scores_tfidf = np.asarray(scores_tfidf).reshape(n_jobs)
                        except Exception:
                            scores_tfidf = np.zeros(n_jobs)

                    top = sorted(range(len(scores_tfidf)), key=lambda i: float(scores_tfidf[i]), reverse=True)[:1000]
                    list_scores = [float(scores_tfidf[i]) for i in top]
                    TF = get_recommendation(top, df, list_scores)

                    # Count Vectorizer
                    status_text.text("🔄 Calculating Count Vector similarity...")
                    progress_bar.progress(50)
                    
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

                    # KNN
                    status_text.text("🔄 Calculating KNN similarity...")
                    progress_bar.progress(75)
                    
                    top_knn, index_score = distance_calculation.KNN(df['All'], df2['All'], number_of_neighbors=100)
                    if isinstance(top_knn, np.ndarray):
                        top_knn = top_knn.tolist()
                    if isinstance(index_score, np.ndarray):
                        index_score = index_score.tolist()
                    knn = get_recommendation(top_knn, df, index_score)

                    # Combine results
                    status_text.text("🔄 Combining results...")
                    progress_bar.progress(90)
                    
                    merge1 = knn[['JobID', 'positionName', 'score']].merge(TF[['JobID', 'score']], on="JobID")
                    final = merge1.merge(cv[['JobID', 'score']], on="JobID")
                    final = final.rename(columns={"score_x": "KNN", "score_y": "TF-IDF", "score": "CV"})

                    from sklearn.preprocessing import MinMaxScaler
                    slr = MinMaxScaler()
                    
                    if final.shape[0] == 0:
                        st.warning("⚠️ No overlapping recommendations found.")
                        return

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

                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()

                    # Display Results
                    st.success(f"🎉 Found {len(final_jobrecomm)} perfect job matches for you!")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Jobs Found", len(final_jobrecomm))
                    with col2:
                        st.metric("Best Match Score", f"{final_jobrecomm['Final'].iloc[0]:.1%}")
                    with col3:
                        st.metric("Avg Match Score", f"{final_jobrecomm['Final'].mean():.1%}")
                    with col4:
                        unique_companies = final_jobrecomm['company'].nunique()
                        st.metric("Companies", unique_companies)

                    st.markdown("---")

                    # View selection tabs
                    view_tab1, view_tab2, view_tab3 = st.tabs(["📋 Card View", "📊 Analytics", "💾 Download Data"])
                    
                    with view_tab1:
                        st.markdown("### 🎯 Your Personalized Job Matches")
                        create_job_cards_view(final_jobrecomm)
                    
                    with view_tab2:
                        # Visualizations
                        df3 = final_jobrecomm.copy()
                        
                        chart_col1, chart_col2 = st.columns(2)
                        
                        with chart_col1:
                            if 'rating' in df3.columns and 'company' in df3.columns:
                                st.markdown("#### ⭐ Company Ratings Distribution")
                                rating_count = df3[["rating", "company"]].dropna()
                                if not rating_count.empty:
                                    fig = px.pie(rating_count, values="rating", names="company", 
                                               title="Ratings by Company")
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        with chart_col2:
                            if 'salary' in df3.columns:
                                st.markdown("#### 💰 Salary Range Distribution")
                                col = df3["salary"].dropna().to_list()
                                y, m = utils.get_monthly_yearly_salary(col)
                                yearly_salary_range = utils.salary_converter(y)
                                monthly_salary_to_yearly = utils.salary_converter(m)
                                final_salary_vals = yearly_salary_range + monthly_salary_to_yearly
                                
                                if final_salary_vals:
                                    salary_df = pd.DataFrame(final_salary_vals, columns=['Salary Range'])
                                    fig2 = px.box(salary_df, y="Salary Range", 
                                                title="Salary Range Distribution")
                                    st.plotly_chart(fig2, use_container_width=True)
                        
                        # Location map
                        if 'location' in df3.columns:
                            st.markdown("#### 🗺️ Job Locations Map")
                            rec_loc = df3.location.value_counts()
                            locations_df = pd.DataFrame(rec_loc).reset_index()
                            locations_df.rename(columns={'index': 'location', 'location': 'count'}, inplace=True)
                            
                            locator = Nominatim(user_agent="JobRecommendation")
                            geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
                            
                            locations_df['loc_geo'] = locations_df['location'].apply(
                                lambda x: geocode(x) if pd.notna(x) else None
                            )
                            locations_df['point'] = locations_df['loc_geo'].apply(
                                lambda loc: tuple(loc.point) if loc else None
                            )
                            
                            if 'point' in locations_df.columns:
                                locations_df[['latitude', 'longitude', 'altitude']] = pd.DataFrame(
                                    locations_df['point'].tolist(), index=locations_df.index
                                )
                                locations_df.dropna(subset=['latitude', 'longitude'], inplace=True)
                                
                                if not locations_df.empty:
                                    folium_map = folium.Map(
                                        location=[locations_df['latitude'].mean(), 
                                                locations_df['longitude'].mean()],
                                        zoom_start=5
                                    )
                                    
                                    for _, row in locations_df.iterrows():
                                        folium.CircleMarker(
                                            [row['latitude'], row['longitude']],
                                            radius=10,
                                            popup=f"<b>{row['location']}</b><br>Jobs: {row['count']}",
                                            color='#667eea',
                                            fill=True,
                                            fillColor='#764ba2'
                                        ).add_to(folium_map)
                                    
                                    folium_static(folium_map, width=1200, height=400)
                    
                    with view_tab3:
                        st.markdown("### 📥 Export Your Job Recommendations")
                        
                        # Prepare download data
                        @st.cache_data
                        def convert_df(df_in):
                            try:
                                return df_in.to_csv(index=False).encode('utf-8')
                            except Exception as e:
                                raise jobException(e, sys)
                        
                        # Determine which position column exists
                        position_col = None
                        if 'positionName' in final_jobrecomm.columns:
                            position_col = 'positionName'
                        elif 'positionName_x' in final_jobrecomm.columns:
                            position_col = 'positionName_x'
                        elif 'title' in final_jobrecomm.columns:
                            position_col = 'title'
                        
                        # Build export columns list
                        export_cols = ['company']
                        if position_col:
                            export_cols.append(position_col)
                        export_cols.extend(['location', 'salary', 'rating', 'description', 'Final'])
                        
                        # Filter only existing columns
                        export_cols_exist = [col for col in export_cols if col in final_jobrecomm.columns]
                        
                        export_df = final_jobrecomm[export_cols_exist].copy()
                        
                        # Rename columns
                        rename_dict = {
                            'company': 'Company',
                            'location': 'Location',
                            'salary': 'Salary',
                            'rating': 'Rating',
                            'description': 'Description',
                            'Final': 'Match Score'
                        }
                        
                        if position_col:
                            rename_dict[position_col] = 'Position'
                        
                        export_df.rename(columns=rename_dict, inplace=True)
                        
                        export_df['Match Score'] = export_df['Match Score'].apply(lambda x: f"{x:.2%}")
                        
                        csv = convert_df(export_df)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="📄 Download as CSV",
                                data=csv,
                                file_name=f"job_recommendations_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                            )
                        
                        with col2:
                            st.info(f"📊 Total jobs in export: {len(export_df)}")
                        
                        # Preview
                        st.markdown("#### Preview Export Data")
                        st.dataframe(export_df.head(10), use_container_width=True)

                    st.balloons()

                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    import traceback
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())
                    raise jobException(e, sys)
    else:
        # Show helpful information when no CV is uploaded
        st.info("👆 Upload your resume to get started!")
        
        st.markdown("""
        ### 🎯 How It Works:
        
        1. **Upload Resume**: Upload your CV in PDF format
        2. **Select Preferences**: Choose preferred locations and number of recommendations
        3. **AI Analysis**: Our algorithms analyze your resume using:
           - TF-IDF similarity matching
           - Count Vectorization
           - K-Nearest Neighbors (KNN)
        4. **Get Results**: Receive personalized job recommendations with match scores
        
        ### ✨ Features:
        - 🎯 **Smart Matching**: AI-powered job recommendations
        - 📊 **Visual Analytics**: Charts and maps for better insights
        - 💾 **Export Options**: Download your results as CSV
        - 🗺️ **Location Filter**: Find jobs in your preferred locations
        
        ### 📋 Tips for Best Results:
        - Ensure your resume has clear sections (Experience, Skills, Education)
        - Use relevant keywords for your target role
        - Keep your PDF text-searchable (not scanned images)
        """)


if __name__ == '__main__':
    app()