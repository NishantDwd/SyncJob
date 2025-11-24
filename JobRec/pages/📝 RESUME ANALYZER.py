import streamlit as st
import pandas as pd
import base64, random
import time, datetime

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('words', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)

from pyresparser import ResumeParser
import io, random
from streamlit_tags import st_tags
from PIL import Image
import pymongo
import plotly.express as px
from JobRecommendation.side_logo import add_logo
from JobRecommendation.sidebar import sidebar
from JobRecommendation.courses import ds_course, web_course, android_course, ios_course, uiux_course, ds_keyword, web_keyword, android_keyword, ios_keyword, uiux_keyword
from JobRecommendation import utils, MongoDB_function
import re
from collections import Counter

dataBase = "Job-Recomendation"
collection = "Resume_from_RESUME_ANALYZER"
st.set_page_config(layout="wide", page_icon='logo/logo2.png', page_title="RESUME ANALYZER")

add_logo()
sidebar()


def extract_contact_info(resume_text):
    """Enhanced contact information extraction"""
    contact_info = {
        'name': None,
        'email': None,
        'mobile_number': None
    }
    
    # Extract email with improved pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, resume_text)
    if emails:
        contact_info['email'] = emails[0]
    
    # Extract phone number (Indian and international formats)
    phone_patterns = [
        r'\+?91[-\s]?[6-9]\d{9}',  # Indian format with +91
        r'\b[6-9]\d{9}\b',  # Indian 10-digit
        r'\+\d{1,3}[-\s]?\d{10}',  # International format
        r'\(\d{3}\)[-\s]?\d{3}[-\s]?\d{4}'  # US format
    ]
    
    for pattern in phone_patterns:
        phones = re.findall(pattern, resume_text)
        if phones:
            # Clean phone number
            phone = re.sub(r'[-\s()]', '', phones[0])
            contact_info['mobile_number'] = phone
            break
    
    # Extract name from first 500 characters (usually at top)
    lines = resume_text[:500].split('\n')
    for line in lines[:10]:
        line = line.strip()
        # Name is usually 2-4 words, capitalized, at the start
        if len(line) > 5 and len(line) < 50:
            words = line.split()
            if 2 <= len(words) <= 4 and all(word[0].isupper() for word in words if word):
                # Avoid common headers
                if not any(header in line.lower() for header in ['resume', 'cv', 'curriculum', 'profile', 'objective']):
                    contact_info['name'] = line
                    break
    
    return contact_info


def course_recommender(course_list):
    st.subheader("📚 Recommended Courses")
    rec_course = []
    no_of_reco = st.slider('Number of Courses to Recommend:', 1, 10, 5)
    random.shuffle(course_list)
    
    for c_name, c_link in course_list[:no_of_reco]:
        st.markdown(f"👉 [{c_name}]({c_link})")
        rec_course.append(c_name)
    
    return rec_course


def extract_sections(resume_text):
    """Extract different sections from resume with improved logic"""
    sections = {
        'experience': '',
        'education': '',
        'projects': '',
        'skills': '',
        'certifications': '',
        'summary': ''
    }
    
    # Common section headers with better patterns
    section_patterns = {
        'experience': r'(?i)(work\s+experience|professional\s+experience|experience|employment\s+history|work\s+history|career\s+history)',
        'education': r'(?i)(education|academic\s+background|qualifications|academic\s+details|educational\s+qualifications)',
        'projects': r'(?i)(projects|personal\s+projects|academic\s+projects|key\s+projects|project\s+work)',
        'skills': r'(?i)(skills|technical\s+skills|core\s+competencies|technologies|technical\s+expertise|proficiencies)',
        'certifications': r'(?i)(certifications?|certificates?|achievements?|awards?|accomplishments?|honors?)',
        'summary': r'(?i)(summary|objective|profile|about\s+me|career\s+objective|professional\s+summary)'
    }
    
    lines = resume_text.split('\n')
    current_section = None
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Check if this line is a section header
        matched_section = None
        for section_name, pattern in section_patterns.items():
            if re.search(pattern, line_stripped):
                matched_section = section_name
                current_section = section_name
                break
        
        # If not a header and we're in a section, add content
        if not matched_section and current_section and len(line_stripped) > 0:
            # Stop if we hit another section header
            is_new_section = any(re.search(pattern, line_stripped) for pattern in section_patterns.values())
            if not is_new_section:
                sections[current_section] += line + ' '
    
    return sections


def calculate_resume_score(resume_data, resume_text, sections, contact_info):
    """Enhanced scoring system with stricter criteria"""
    score = 0
    max_score = 100
    feedback = []
    
    # 1. Contact Information (15 points total) - STRICT
    if contact_info.get('name') and len(contact_info['name'].split()) >= 2:
        score += 5
        feedback.append(('positive', f"✅ Full name found: {contact_info['name']}"))
    else:
        feedback.append(('negative', '❌ Full name not clearly visible at the top'))
    
    if contact_info.get('email') and '@' in contact_info['email']:
        score += 5
        feedback.append(('positive', f"✅ Email found: {contact_info['email']}"))
    else:
        feedback.append(('negative', '❌ Professional email address missing'))
    
    if contact_info.get('mobile_number'):
        score += 5
        feedback.append(('positive', f"✅ Contact number found: {contact_info['mobile_number']}"))
    else:
        feedback.append(('negative', '❌ Contact phone number missing'))
    
    # 2. Professional Summary (8 points) - STRICT
    summary_content = sections.get('summary', '')
    if len(summary_content) > 100:
        score += 8
        feedback.append(('positive', '✅ Strong professional summary/objective (100+ characters)'))
    elif len(summary_content) > 50:
        score += 4
        feedback.append(('warning', '⚠️ Summary present but brief - expand to 2-3 lines'))
    else:
        feedback.append(('negative', '❌ Add a professional summary/objective (2-3 impactful lines)'))
    
    # 3. Work Experience (20 points) - VERY STRICT
    experience_content = sections.get('experience', '')
    has_numbers = bool(re.search(r'\d+[%+]|\d+\s*years?', experience_content))
    has_action_verbs = sum(1 for verb in ['developed', 'managed', 'led', 'created', 'implemented', 
                                           'designed', 'built', 'improved', 'increased', 'reduced'] 
                          if verb in experience_content.lower())
    
    if len(experience_content) > 300 and has_numbers and has_action_verbs >= 3:
        score += 20
        feedback.append(('positive', f'✅ Excellent work experience with quantifiable achievements (found {has_action_verbs} action verbs)'))
    elif len(experience_content) > 150:
        score += 12
        if not has_numbers:
            feedback.append(('warning', '⚠️ Add quantifiable achievements (e.g., "Increased efficiency by 30%")'))
        if has_action_verbs < 3:
            feedback.append(('warning', '⚠️ Use more action verbs (Developed, Managed, Led, etc.)'))
    elif len(experience_content) > 50:
        score += 6
        feedback.append(('warning', '⚠️ Work experience too brief - add detailed responsibilities and achievements'))
    else:
        feedback.append(('negative', '❌ Add comprehensive work experience or internships'))
    
    # 4. Education (12 points) - STRICT
    education_content = sections.get('education', '')
    has_degree = bool(re.search(r'bachelor|master|phd|b\.tech|m\.tech|bsc|msc|mba|be|me', 
                                education_content.lower()))
    has_year = bool(re.search(r'20\d{2}|19\d{2}', education_content))
    
    if len(education_content) > 100 and has_degree and has_year:
        score += 12
        feedback.append(('positive', '✅ Complete education details with degree and year'))
    elif len(education_content) > 50 or has_degree:
        score += 7
        feedback.append(('warning', '⚠️ Add graduation year, GPA (if good), and major details'))
    else:
        feedback.append(('negative', '❌ Add detailed education information'))
    
    # 5. Projects (15 points) - STRICT
    projects_content = sections.get('projects', '')
    has_github = bool(re.search(r'github|git|portfolio|gitlab', resume_text, re.IGNORECASE))
    project_count = len(re.findall(r'project\s*\d+|•|\*|\d+\)', projects_content, re.IGNORECASE))
    
    if len(projects_content) > 200 and (has_github or project_count >= 2):
        score += 15
        feedback.append(('positive', f'✅ Strong projects section with {project_count}+ projects'))
    elif len(projects_content) > 100:
        score += 9
        if not has_github:
            feedback.append(('warning', '⚠️ Add GitHub/portfolio links to showcase your work'))
    elif len(projects_content) > 30:
        score += 4
        feedback.append(('warning', '⚠️ Add more project descriptions with technologies used'))
    else:
        feedback.append(('negative', '❌ Add 2-3 detailed projects with GitHub links'))
    
    # 6. Skills (12 points) - STRICT
    skills_list = resume_data.get('skills', [])
    skills_content = sections.get('skills', '')
    total_skills = len(skills_list) if skills_list else len(skills_content.split(','))
    
    if total_skills >= 15:
        score += 12
        feedback.append(('positive', f'✅ Comprehensive skills list ({total_skills} skills)'))
    elif total_skills >= 10:
        score += 8
        feedback.append(('warning', f'⚠️ Good skills ({total_skills}) - add 5 more relevant ones'))
    elif total_skills >= 5:
        score += 4
        feedback.append(('warning', f'⚠️ Limited skills ({total_skills}) - add 10 more technical skills'))
    else:
        feedback.append(('negative', '❌ Add at least 10-15 relevant technical skills'))
    
    # 7. Certifications/Achievements (8 points) - STRICT
    cert_content = sections.get('certifications', '')
    cert_count = len(re.findall(r'certificate|certified|certification|award|achievement', 
                                cert_content, re.IGNORECASE))
    
    if len(cert_content) > 100 or cert_count >= 2:
        score += 8
        feedback.append(('positive', f'✅ Certifications/achievements section found ({cert_count}+ items)'))
    elif cert_count >= 1:
        score += 4
        feedback.append(('warning', '⚠️ Add more certifications or professional achievements'))
    else:
        feedback.append(('negative', '❌ Add relevant certifications, awards, or achievements'))
    
    # 8. Resume Length (5 points)
    pages = resume_data.get('no_of_pages', 1)
    if pages == 1 or pages == 2:
        score += 5
        feedback.append(('positive', f'✅ Optimal resume length ({pages} page{"s" if pages > 1 else ""})'))
    elif pages > 2:
        score += 2
        feedback.append(('warning', f'⚠️ Resume is {pages} pages - consider condensing to 2 pages'))
    
    # 9. Formatting Quality (5 points) - Check for consistency
    bullet_points = len(re.findall(r'[•\-\*]', resume_text))
    if bullet_points >= 5:
        score += 5
        feedback.append(('positive', '✅ Good use of bullet points for readability'))
    else:
        feedback.append(('warning', '⚠️ Use bullet points to organize information'))
    
    return min(score, max_score), feedback


def determine_career_level(resume_data, resume_text, sections):
    """More accurate career level determination"""
    experience_content = sections.get('experience', '')
    
    # Extract years of experience
    years_patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)',
        r'(\d{4})\s*-\s*(\d{4}|present|current)',
    ]
    
    total_years = 0
    for pattern in years_patterns:
        matches = re.findall(pattern, experience_content.lower())
        if matches:
            if isinstance(matches[0], tuple):
                # Date range
                for match in matches:
                    try:
                        start = int(match[0])
                        end = int(match[1]) if match[1].isdigit() else datetime.datetime.now().year
                        total_years += max(0, end - start)
                    except:
                        pass
            else:
                # Years mentioned
                total_years = max(total_years, int(matches[0]))
    
    # Count job positions
    job_titles = len(re.findall(r'(?i)(engineer|developer|manager|lead|senior|analyst|consultant|specialist)', 
                                experience_content))
    
    # Check education level
    education_content = sections.get('education', '')
    has_phd = bool(re.search(r'phd|doctorate', education_content.lower()))
    has_masters = bool(re.search(r'master|m\.tech|msc|mba|ms\s', education_content.lower()))
    
    # Determine level
    if total_years >= 5 or job_titles >= 5 or has_phd:
        return "Experienced (5+ years)", "#10b981"
    elif total_years >= 2 or job_titles >= 3 or (has_masters and len(experience_content) > 200):
        return "Intermediate (2-5 years)", "#3b82f6"
    elif total_years >= 1 or len(experience_content) > 150:
        return "Junior (1-2 years)", "#8b5cf6"
    else:
        return "Fresher / Entry Level", "#f59e0b"


def analyze_skills_and_recommend(resume_data, resume_text, sections):
    """Improved field detection with better accuracy - Now includes AI & ML as distinct"""
    skills = resume_data.get('skills', [])
    skills_lower = [skill.lower() for skill in skills] if skills else []
    
    # Get focused text
    skills_section = sections.get('skills', '').lower()
    experience_section = sections.get('experience', '').lower()
    projects_section = sections.get('projects', '').lower()
    
    focused_text = f"{skills_section} {experience_section} {projects_section}"
    
    # Field scoring with improved logic
    field_scores = {}
    
    # Data Science keywords (weighted)
    ds_high_value = ['data science', 'data analysis', 'statistics', 'statistical analysis', 
                     'data visualization', 'tableau', 'power bi', 'regression', 'clustering']
    ds_medium_value = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'excel', 'sql', 
                       'data mining', 'exploratory data analysis', 'eda']
    
    ds_score = 0
    for kw in ds_high_value:
        if any(kw in skill.lower() for skill in skills_lower) or kw in focused_text:
            ds_score += 3
    for kw in ds_medium_value:
        if any(kw in skill.lower() for skill in skills_lower) or kw in focused_text:
            ds_score += 1
    field_scores['Data Science'] = ds_score
    
    # AI & Machine Learning keywords (weighted) - NEW DISTINCT FIELD
    aiml_high_value = ['machine learning', 'deep learning', 'artificial intelligence', 'ai', 
                       'tensorflow', 'keras', 'pytorch', 'neural network', 'nlp', 
                       'natural language processing', 'computer vision', 'cnn', 'rnn', 'lstm',
                       'transformers', 'bert', 'gpt', 'reinforcement learning']
    aiml_medium_value = ['scikit-learn', 'sklearn', 'model training', 'ml model', 
                         'classification', 'random forest', 'svm', 'gradient boosting',
                         'xgboost', 'feature engineering', 'hyperparameter tuning']
    
    aiml_score = 0
    for kw in aiml_high_value:
        if any(kw in skill.lower() for skill in skills_lower) or kw in focused_text:
            aiml_score += 3
    for kw in aiml_medium_value:
        if any(kw in skill.lower() for skill in skills_lower) or kw in focused_text:
            aiml_score += 1
    field_scores['AI & Machine Learning'] = aiml_score
    
    # Web Development keywords
    web_high_value = ['react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring boot', 
                      'full stack', 'frontend', 'backend']
    web_medium_value = ['javascript', 'html', 'css', 'typescript', 'mongodb', 'postgresql', 
                        'rest api', 'graphql', 'docker']
    
    web_score = 0
    for kw in web_high_value:
        if any(kw in skill.lower() for skill in skills_lower) or kw in focused_text:
            web_score += 3
    for kw in web_medium_value:
        if any(kw in skill.lower() for skill in skills_lower) or kw in focused_text:
            web_score += 1
    field_scores['Web Development'] = web_score
    
    # Android Development
    android_keywords = ['android', 'kotlin', 'java', 'android studio', 'firebase', 'xml', 'gradle']
    android_score = sum(2 if any(kw in skill.lower() for skill in skills_lower) else 
                       1 if kw in focused_text else 0 for kw in android_keywords)
    field_scores['Android Development'] = android_score
    
    # iOS Development
    ios_keywords = ['swift', 'ios', 'xcode', 'swiftui', 'uikit', 'objective-c']
    ios_score = sum(2 if any(kw in skill.lower() for skill in skills_lower) else 
                    1 if kw in focused_text else 0 for kw in ios_keywords)
    field_scores['iOS Development'] = ios_score
    
    # UI/UX Design
    uiux_keywords = ['figma', 'adobe xd', 'sketch', 'ui/ux', 'user experience', 'wireframe', 
                     'prototype', 'design']
    uiux_score = sum(2 if any(kw in skill.lower() for skill in skills_lower) else 
                     1 if kw in focused_text else 0 for kw in uiux_keywords)
    field_scores['UI-UX Development'] = uiux_score
    
    # Determine primary field
    max_score = max(field_scores.values()) if field_scores else 0
    
    if max_score < 3:
        return None, [], [], field_scores
    
    primary_field = max(field_scores, key=field_scores.get)
    
    # Map to courses and skills (Updated with AI & ML)
    field_mapping = {
        'Data Science': {'skills': ds_high_value + ds_medium_value, 'courses': ds_course},
        'AI & Machine Learning': {'skills': aiml_high_value + aiml_medium_value, 'courses': ds_course},  # Using ds_course for now
        'Web Development': {'skills': web_high_value + web_medium_value, 'courses': web_course},
        'Android Development': {'skills': android_keywords, 'courses': android_course},
        'iOS Development': {'skills': ios_keywords, 'courses': ios_course},
        'UI-UX Development': {'skills': uiux_keywords, 'courses': uiux_course}
    }
    
    return (primary_field, 
            field_mapping[primary_field]['skills'], 
            field_mapping[primary_field]['courses'], 
            field_scores)


def run():
    st.title("🎯 AI-Powered Resume Analyzer")
    st.markdown("Get comprehensive analysis and personalized recommendations for your resume")
    
    pdf_file = st.file_uploader("📄 Upload Your Resume (PDF)", type=["pdf"])
    
    if pdf_file is not None:
        with st.spinner('🔍 Analyzing your resume...'):
            try:
                # Save and encode PDF
                encoded_pdf = utils.pdf_to_base64(pdf_file)
                
                # Extract text
                resume_text = utils.pdf_reader(pdf_file)
                
                if not resume_text or len(resume_text.strip()) < 100:
                    st.error("❌ Unable to extract sufficient text. Ensure PDF is text-based, not scanned.")
                    return
                
                # Parse resume with pyresparser
                try:
                    resume_data = ResumeParser(pdf_file).get_extracted_data()
                    if not resume_data:
                        resume_data = {}
                except Exception as e:
                    st.warning(f"⚠️ Parser had issues: {str(e)}. Using fallback extraction.")
                    resume_data = {}
                
                # Enhanced contact extraction - ALWAYS EXTRACT FROM TEXT
                contact_info = extract_contact_info(resume_text)
                
                # FORCE USE EXTRACTED CONTACT INFO - Override parser data
                resume_data['name'] = contact_info.get('name') or resume_data.get('name', 'Not Detected')
                resume_data['email'] = contact_info.get('email') or resume_data.get('email', 'Not Detected')
                resume_data['mobile_number'] = contact_info.get('mobile_number') or resume_data.get('mobile_number', 'Not Detected')
                
                resume_data["pdf_to_base64"] = encoded_pdf
                
                # Extract sections
                sections = extract_sections(resume_text)
                
                st.success("✅ Analysis Complete!")
                st.markdown("---")
                
                # Display Profile
                st.header("👤 Candidate Profile")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    # Use the EXTRACTED contact info, not parser data
                    name_value = resume_data.get('name', 'Not Detected')
                    if name_value and name_value != 'Not Detected':
                        st.metric("Name", name_value)
                    else:
                        st.metric("Name", "⚠️ Not Found")
                
                with col2:
                    email_value = resume_data.get('email', 'Not Detected')
                    if email_value and email_value != 'Not Detected' and '@' in str(email_value):
                        st.metric("Email", email_value)
                    else:
                        st.metric("Email", "⚠️ Not Found")
                
                with col3:
                    phone_value = resume_data.get('mobile_number', 'Not Detected')
                    if phone_value and phone_value != 'Not Detected':
                        st.metric("Contact", phone_value)
                    else:
                        st.metric("Contact", "⚠️ Not Found")
                
                col1, col2 = st.columns(2)
                with col1:
                    pages = resume_data.get('no_of_pages', 1)
                    st.metric("Resume Pages", pages)
                with col2:
                    cand_level, level_color = determine_career_level(resume_data, resume_text, sections)
                    st.markdown(f"""
                    <div style='padding: 15px; background: linear-gradient(135deg, {level_color}20, {level_color}10); 
                                border-left: 4px solid {level_color}; border-radius: 8px;'>
                        <h4 style='color: {level_color}; margin: 0; font-size: 18px;'>
                            Career Level: {cand_level}
                        </h4>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Resume Score
                st.header("📊 Resume Score & Detailed Feedback")
                
                resume_score, feedback = calculate_resume_score(resume_data, resume_text, sections, contact_info)
                
                # Score visualization
                if resume_score >= 80:
                    score_color = "#10b981"
                    score_emoji = "🎉"
                    score_message = "Outstanding Resume!"
                elif resume_score >= 65:
                    score_color = "#3b82f6"
                    score_emoji = "👍"
                    score_message = "Good Resume"
                elif resume_score >= 50:
                    score_color = "#f59e0b"
                    score_emoji = "⚠️"
                    score_message = "Needs Improvement"
                else:
                    score_color = "#ef4444"
                    score_emoji = "❗"
                    score_message = "Requires Significant Work"
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    progress = st.progress(0)
                    for i in range(resume_score + 1):
                        time.sleep(0.01)
                        progress.progress(i)
                
                with col2:
                    st.markdown(f"""
                    <div style='text-align: center;'>
                        <h1 style='color: {score_color}; font-size: 48px; margin: 0;'>{resume_score}</h1>
                        <p style='color: {score_color}; margin: 0;'>out of 100</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"<h2 style='text-align: center; color: {score_color};'>{score_emoji} {score_message}</h2>", 
                           unsafe_allow_html=True)
                
                # Detailed Feedback
                st.subheader("📋 Detailed Analysis")
                
                positive = [f for t, f in feedback if t == 'positive']
                warnings = [f for t, f in feedback if t == 'warning']
                negative = [f for t, f in feedback if t == 'negative']
                
                if positive:
                    with st.expander(f"✅ Strengths ({len(positive)})", expanded=True):
                        for msg in positive:
                            st.markdown(f"- {msg}")
                
                if warnings:
                    with st.expander(f"⚠️ Areas for Improvement ({len(warnings)})", expanded=True):
                        for msg in warnings:
                            st.markdown(f"- {msg}")
                
                if negative:
                    with st.expander(f"❌ Critical Missing Elements ({len(negative)})", expanded=True):
                        for msg in negative:
                            st.markdown(f"- {msg}")
                
                st.markdown("---")
                
                # Skills & Career Recommendations
                st.header("🎯 Career Path & Skills Analysis")
                
                current_skills = resume_data.get('skills', [])
                if current_skills:
                    st.subheader("Your Current Skills")
                    st_tags(label='', text='Detected skills', value=current_skills, key='current_skills')
                else:
                    st.warning("⚠️ No skills clearly detected. Add a 'Skills' section with bullet points.")
                
                # Analyze career field
                field, recommended_skills, courses, field_scores = analyze_skills_and_recommend(
                    resume_data, resume_text, sections
                )
                
                if field_scores:
                    st.subheader("Career Field Match Analysis")
                    
                    # Visualization
                    score_df = pd.DataFrame({
                        'Field': list(field_scores.keys()),
                        'Match Score': list(field_scores.values())
                    }).sort_values('Match Score', ascending=False)
                    
                    fig = px.bar(score_df, x='Match Score', y='Field', orientation='h',
                                color='Match Score', color_continuous_scale='viridis',
                                title='Your Profile Match by Career Field')
                    st.plotly_chart(fig, use_container_width=True)
                
                if field:
                    st.success(f"🎯 **Best Match: {field}**")
                    st.info(f"Based on your skills and experience, you're well-suited for **{field}** roles.")
                    
                    st.subheader(f"Recommended Skills for {field}")
                    
                    # Find missing skills
                    current_lower = [s.lower() for s in current_skills]
                    missing_skills = [s for s in recommended_skills if s.lower() not in current_lower][:15]
                    
                    if missing_skills:
                        st_tags(label='', text='Add these skills', value=missing_skills, key='recommended')
                        st.markdown("<p style='color: #10b981;'>💡 <b>Adding these skills will significantly improve your profile!</b></p>", 
                                   unsafe_allow_html=True)
                    else:
                        st.success("🎉 Great! You have most recommended skills for this field.")
                    
                    # Course recommendations
                    if courses:
                        st.markdown("---")
                        rec_courses = course_recommender(courses)
                else:
                    st.warning("💡 Add more technical skills to get targeted recommendations.")
                
                st.markdown("---")
                
                # Tips
                with st.expander("📚 Resume Best Practices", expanded=False):
                    st.markdown("""
                    ### Professional Resume Guidelines:
                    1. **Contact Info**: Name, phone, email, LinkedIn at the top
                    2. **Summary**: 2-3 lines highlighting your expertise
                    3. **Action Verbs**: Start with Developed, Managed, Led, Implemented
                    4. **Quantify**: Include numbers (e.g., "Improved efficiency by 30%")
                    5. **Projects**: 2-3 with GitHub links and tech stack
                    6. **Skills**: 10-15 relevant technical skills
                    7. **Length**: 1-2 pages maximum
                    8. **Format**: Clear sections, consistent style
                    9. **Keywords**: Match job descriptions
                    10. **Proofread**: Zero typos
                    """)
                
                # Save to MongoDB
                timestamp = utils.generateUniqueFileName()
                save_data = {
                    timestamp: {
                        **resume_data,
                        'analysis': {
                            'career_level': cand_level,
                            'recommended_field': field,
                            'resume_score': resume_score,
                            'field_scores': field_scores,
                            'analyzed_at': datetime.datetime.now().isoformat()
                        }
                    }
                }
                MongoDB_function.resume_store(save_data, dataBase, collection)
                st.success("✅ Analysis saved!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    
    else:
        st.info("👆 Upload your resume to get started")
        
        with st.expander("📖 What You'll Get"):
            st.markdown("""
            ### Comprehensive Analysis:
            - ✅ **Accurate Score** (0-100) based on industry standards
            - 📊 **Career Level** detection
            - 🎯 **Career Field** recommendations (including AI & ML as distinct from Data Science)
            - 💼 **Skills Gap** analysis
            - 📚 **Course Recommendations**
            - 💡 **Actionable Feedback**
            - 📈 **Visual Analytics**
            """)

run()