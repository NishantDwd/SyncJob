import streamlit as st
import pandas as pd
import base64,random
import time,datetime
from pyresparser import ResumeParser
import io,random
from streamlit_tags import st_tags
from PIL import Image
import pymongo
import plotly.express as px
from JobRecommendation.side_logo import add_logo
from JobRecommendation.sidebar import sidebar
from JobRecommendation.courses import ds_course,web_course,android_course,ios_course,uiux_course,ds_keyword,web_keyword,android_keyword,ios_keyword,uiux_keyword
from JobRecommendation import utils ,MongoDB_function
import re
from collections import Counter

dataBase = "Job-Recomendation"
collection = "Resume_from_RESUME_ANALYZER"
st.set_page_config(layout="wide", page_icon='logo/logo2.png', page_title="RESUME ANALYZER")


add_logo()
sidebar()


def course_recommender(course_list):
    st.subheader("**Courses & Certificates🎓 Recommendations**")
    c = 0
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 4)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course


def extract_sections(resume_text):
    """Extract different sections from resume"""
    sections = {
        'experience': '',
        'education': '',
        'projects': '',
        'skills': '',
        'certifications': ''
    }
    
    # Common section headers
    experience_headers = r'(work experience|professional experience|experience|employment history|work history)'
    education_headers = r'(education|academic background|qualifications|academic details)'
    projects_headers = r'(projects|personal projects|academic projects|key projects)'
    skills_headers = r'(skills|technical skills|core competencies|technologies|technical expertise)'
    cert_headers = r'(certifications|certificates|achievements|awards|accomplishments)'
    
    lines = resume_text.split('\n')
    current_section = None
    
    for line in lines:
        line_lower = line.lower().strip()
        
        if re.search(experience_headers, line_lower):
            current_section = 'experience'
        elif re.search(education_headers, line_lower):
            current_section = 'education'
        elif re.search(projects_headers, line_lower):
            current_section = 'projects'
        elif re.search(skills_headers, line_lower):
            current_section = 'skills'
        elif re.search(cert_headers, line_lower):
            current_section = 'certifications'
        elif current_section and len(line.strip()) > 0:
            sections[current_section] += line + ' '
    
    return sections


def calculate_resume_score(resume_data, resume_text, sections):
    """Calculate resume score based on actual content analysis"""
    score = 0
    feedback = []
    
    # Check for basic information (20 points)
    if resume_data.get('name'):
        score += 5
        feedback.append(('positive', 'Contact: Name is present'))
    else:
        feedback.append(('negative', 'Add your full name prominently at the top'))
        
    if resume_data.get('email'):
        score += 5
        feedback.append(('positive', 'Contact: Email address is present'))
    else:
        feedback.append(('negative', 'Add a professional email address'))
        
    if resume_data.get('mobile_number'):
        score += 5
        feedback.append(('positive', 'Contact: Phone number is present'))
    else:
        feedback.append(('negative', 'Add a contact phone number'))
        
    if resume_data.get('skills') and len(resume_data['skills']) > 0:
        score += 5
        feedback.append(('positive', 'Skills section is present'))
    else:
        feedback.append(('negative', 'Add a dedicated skills section'))
    
    # Check for Objective/Summary (10 points)
    summary_keywords = ['objective', 'summary', 'profile', 'about me', 'career objective', 'professional summary']
    if any(keyword in resume_text.lower()[:500] for keyword in summary_keywords):
        score += 10
        feedback.append(('positive', '✅ Professional summary/objective found'))
    else:
        feedback.append(('negative', '❌ Add a career objective or professional summary (2-3 lines about your career goals)'))
    
    # Check for Experience (20 points)
    experience_content = sections.get('experience', '')
    if len(experience_content) > 100:
        score += 20
        # Check for measurable achievements
        if any(char.isdigit() for char in experience_content):
            feedback.append(('positive', '✅ Work experience with quantifiable achievements detected'))
        else:
            feedback.append(('positive', '✅ Work experience section found'))
            feedback.append(('warning', '⚠️ Add quantifiable achievements (e.g., "Increased sales by 30%")'))
    elif len(experience_content) > 20:
        score += 10
        feedback.append(('warning', '⚠️ Work experience is too brief - add more details about your roles and achievements'))
    else:
        feedback.append(('negative', '❌ Add work experience or internships with detailed responsibilities'))
    
    # Check for Education (15 points)
    education_content = sections.get('education', '')
    if resume_data.get('degree') or len(education_content) > 50:
        score += 15
        feedback.append(('positive', '✅ Education details are comprehensive'))
    elif len(education_content) > 20:
        score += 10
        feedback.append(('warning', '⚠️ Add more education details (graduation year, major, GPA if good)'))
    else:
        feedback.append(('negative', '❌ Add detailed education information'))
    
    # Check for Projects (15 points)
    projects_content = sections.get('projects', '')
    github_mentioned = bool(re.search(r'github|git|portfolio|gitlab', resume_text, re.IGNORECASE))
    
    if len(projects_content) > 100 or github_mentioned:
        score += 15
        if github_mentioned:
            feedback.append(('positive', '✅ Excellent! Projects section with GitHub/portfolio links'))
        else:
            feedback.append(('positive', '✅ Projects section is detailed'))
    elif len(projects_content) > 20:
        score += 8
        feedback.append(('warning', '⚠️ Add more project details and include GitHub links'))
    else:
        feedback.append(('negative', '❌ Add 2-3 projects with descriptions and GitHub/portfolio links'))
    
    # Check for Skills quality (10 points)
    skills_list = resume_data.get('skills', [])
    if len(skills_list) >= 10:
        score += 10
        feedback.append(('positive', f'✅ Excellent! {len(skills_list)} skills listed'))
    elif len(skills_list) >= 5:
        score += 6
        feedback.append(('warning', f'⚠️ Good start with {len(skills_list)} skills - add 5-10 more relevant skills'))
    elif len(skills_list) > 0:
        score += 3
        feedback.append(('warning', f'⚠️ Only {len(skills_list)} skills listed - add at least 8-10 relevant skills'))
    
    # Check for Certifications/Achievements (10 points)
    cert_content = sections.get('certifications', '')
    cert_keywords = ['certificate', 'certification', 'certified', 'achievement', 'award', 'honor', 'recognition']
    has_certs = len(cert_content) > 30 or any(keyword in resume_text.lower() for keyword in cert_keywords)
    
    if has_certs:
        score += 10
        feedback.append(('positive', '✅ Certifications/achievements section found'))
    else:
        feedback.append(('negative', '❌ Add certifications, awards, or achievements to stand out'))
    
    # Resume length check (5 points)
    pages = resume_data.get('no_of_pages', 1)
    if pages == 1 or pages == 2:
        score += 5
        feedback.append(('positive', f'✅ Optimal resume length ({pages} page{"s" if pages > 1 else ""})'))
    elif pages > 2:
        feedback.append(('warning', f'⚠️ Resume is {pages} pages - try to keep it concise (1-2 pages)'))
    
    # Check for action verbs in experience
    action_verbs = ['developed', 'managed', 'led', 'created', 'implemented', 'designed', 'built', 
                   'improved', 'increased', 'reduced', 'achieved', 'delivered']
    has_action_verbs = sum(1 for verb in action_verbs if verb in resume_text.lower())
    
    if has_action_verbs >= 5:
        feedback.append(('positive', f'✅ Strong action verbs used ({has_action_verbs} found)'))
    elif has_action_verbs >= 2:
        feedback.append(('warning', '⚠️ Use more action verbs (Developed, Managed, Led, Implemented, etc.)'))
    
    return min(score, 100), feedback


def determine_career_level(resume_data, resume_text, sections):
    """Determine candidate level based on resume content"""
    experience_content = sections.get('experience', '')
    
    # More sophisticated experience detection
    years_pattern = r'(\d+)\+?\s*(?:years?|yrs?)'
    years_matches = re.findall(years_pattern, experience_content.lower())
    total_years = sum(int(year) for year in years_matches) if years_matches else 0
    
    # Count job positions
    job_indicators = len(re.findall(r'(intern|engineer|developer|manager|lead|senior|junior|associate)', 
                                     experience_content.lower()))
    
    # Check skills count
    skill_count = len(resume_data.get('skills', []))
    
    # Check education level
    education_content = sections.get('education', '') + str(resume_data.get('degree', ''))
    has_masters = bool(re.search(r'master|m\.tech|msc|mba|ms\s', education_content.lower()))
    has_phd = bool(re.search(r'phd|doctorate', education_content.lower()))
    
    # Determine level
    if total_years >= 5 or job_indicators >= 4 or has_phd:
        return "Experienced (5+ years)", "#fba171"
    elif total_years >= 2 or job_indicators >= 2 or (has_masters and skill_count >= 8):
        return "Intermediate (2-5 years)", "#1ed760"
    elif total_years >= 1 or len(experience_content) > 200:
        return "Junior (1-2 years)", "#51c4d3"
    else:
        return "Fresher", "#d73b5c"


def analyze_skills_and_recommend(resume_data, resume_text, sections):
    """Enhanced skill analysis with weighted scoring"""
    skills = resume_data.get('skills', [])
    skills_lower = [skill.lower() for skill in skills]
    resume_text_lower = resume_text.lower()
    
    # Get skills and experience sections for more focused analysis
    skills_section = sections.get('skills', '').lower()
    experience_section = sections.get('experience', '').lower()
    projects_section = sections.get('projects', '').lower()
    
    # Combine sections with weights
    focused_text = skills_section + ' ' + experience_section + ' ' + projects_section
    
    # Calculate weighted scores for each field
    field_scores = {}
    
    # Data Science scoring with weights
    ds_matches = []
    for kw in ds_keyword:
        kw_lower = kw.lower()
        # Higher weight for skills in skills section
        skill_match = kw_lower in skills_lower
        text_match = kw_lower in focused_text
        
        if skill_match:
            ds_matches.append(2)  # Weight: 2 for direct skill match
        elif text_match:
            ds_matches.append(1)  # Weight: 1 for text mention
    field_scores['Data Science'] = sum(ds_matches)
    
    # Web Development scoring
    web_matches = []
    for kw in web_keyword:
        kw_lower = kw.lower()
        skill_match = kw_lower in skills_lower
        text_match = kw_lower in focused_text
        
        if skill_match:
            web_matches.append(2)
        elif text_match:
            web_matches.append(1)
    field_scores['Web Development'] = sum(web_matches)
    
    # Android Development scoring
    android_matches = []
    for kw in android_keyword:
        kw_lower = kw.lower()
        skill_match = kw_lower in skills_lower
        text_match = kw_lower in focused_text
        
        if skill_match:
            android_matches.append(2)
        elif text_match:
            android_matches.append(1)
    field_scores['Android Development'] = sum(android_matches)
    
    # iOS Development scoring
    ios_matches = []
    for kw in ios_keyword:
        kw_lower = kw.lower()
        skill_match = kw_lower in skills_lower
        text_match = kw_lower in focused_text
        
        if skill_match:
            ios_matches.append(2)
        elif text_match:
            ios_matches.append(1)
    field_scores['iOS Development'] = sum(ios_matches)
    
    # UI-UX Development scoring
    uiux_matches = []
    for kw in uiux_keyword:
        kw_lower = kw.lower()
        skill_match = kw_lower in skills_lower
        text_match = kw_lower in focused_text
        
        if skill_match:
            uiux_matches.append(2)
        elif text_match:
            uiux_matches.append(1)
    field_scores['UI-UX Development'] = sum(uiux_matches)
    
    # Get top fields (can be multiple if close scores)
    max_score = max(field_scores.values()) if field_scores else 0
    
    if max_score == 0:
        return None, [], [], field_scores
    
    # Allow multiple fields if scores are close (within 30%)
    threshold = max_score * 0.7
    recommended_fields = [field for field, score in field_scores.items() if score >= threshold]
    
    # Sort by score
    recommended_fields.sort(key=lambda x: field_scores[x], reverse=True)
    primary_field = recommended_fields[0] if recommended_fields else None
    
    # Map field to skills and courses
    field_mapping = {
        'Data Science': {
            'skills': ['TensorFlow','Keras','PyTorch','Scikit-learn','Pandas','NumPy','Matplotlib','Seaborn',
                      'Data Visualization','Statistical Modeling','Machine Learning','Deep Learning',
                      'Natural Language Processing','Computer Vision','SQL','Big Data','Apache Spark',
                      'Data Mining','Time Series Analysis','Feature Engineering'],
            'courses': ds_course
        },
        'Web Development': {
            'skills': ['React','Angular','Vue.js','Node.js','Express.js','Django','Flask','Spring Boot',
                      'MongoDB','PostgreSQL','MySQL','Redis','GraphQL','REST API','Docker','Kubernetes',
                      'AWS','Azure','Git','CI/CD','TypeScript','Webpack','Jest','Microservices'],
            'courses': web_course
        },
        'Android Development': {
            'skills': ['Kotlin','Java','Android SDK','Jetpack Compose','MVVM','Room','Retrofit',
                      'Coroutines','LiveData','ViewModel','Material Design','Firebase','Git',
                      'RESTful APIs','SQLite','Gradle','Android Studio','Unit Testing','RxJava'],
            'courses': android_course
        },
        'iOS Development': {
            'skills': ['Swift','SwiftUI','UIKit','Xcode','Core Data','Alamofire','Combine',
                      'MVVM','MVC','Grand Central Dispatch','Auto Layout','TestFlight',
                      'Core Animation','AVFoundation','MapKit','Push Notifications','Git'],
            'courses': ios_course
        },
        'UI-UX Development': {
            'skills': ['Figma','Adobe XD','Sketch','InVision','Prototyping','Wireframing',
                      'User Research','Usability Testing','Information Architecture','Interaction Design',
                      'Visual Design','Typography','Color Theory','Design Systems','Responsive Design',
                      'Adobe Illustrator','Adobe Photoshop','User Personas','A/B Testing'],
            'courses': uiux_course
        }
    }
    
    if primary_field:
        return primary_field, field_mapping[primary_field]['skills'], field_mapping[primary_field]['courses'], field_scores
    
    return None, [], [], field_scores


def run():
    st.title("AI-Powered Resume Analyzer 🤖")
    st.markdown("Get intelligent insights and personalized recommendations for your resume")

    pdf_file = st.file_uploader("📄 Upload Your Resume (PDF)", type=["pdf"])
    
    if pdf_file is not None:
        with st.spinner('🔍 Analyzing your resume with AI...'):
            try:
                # Extract resume data
                encoded_pdf = utils.pdf_to_base64(pdf_file)
                
                try:
                    resume_data = ResumeParser(pdf_file).get_extracted_data()
                except:
                    resume_data = {}
                
                resume_data["pdf_to_base64"] = encoded_pdf
                
                # Get full resume text
                resume_text = utils.pdf_reader(pdf_file)

                if not resume_text or resume_text.strip() == "":
                    st.error("❌ Unable to extract text from resume. Please ensure the PDF is readable and not image-based.")
                    st.info("💡 Try converting your resume to a text-based PDF or use OCR tools.")
                    return

                # Extract sections
                sections = extract_sections(resume_text)

                st.success("✅ Resume analysis completed!")
                st.markdown("---")

                # Display basic info
                st.header("👤 Candidate Profile")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Name", resume_data.get('name', 'Not Found'))
                with col2:
                    st.metric("Email", resume_data.get('email', 'Not Found'))
                with col3:
                    st.metric("Contact", resume_data.get('mobile_number', 'Not Found'))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Resume Pages", resume_data.get('no_of_pages', 'N/A'))
                with col2:
                    # Determine career level
                    cand_level, level_color = determine_career_level(resume_data, resume_text, sections)
                    st.markdown(f"<div style='padding: 10px; background-color: {level_color}20; border-left: 4px solid {level_color}; border-radius: 5px;'>"
                               f"<h4 style='color: {level_color}; margin: 0;'>Career Level: {cand_level}</h4></div>", 
                               unsafe_allow_html=True)

                st.markdown("---")

                # Resume Score and Feedback
                st.header("📊 Resume Score & Analysis")
                
                resume_score, feedback = calculate_resume_score(resume_data, resume_text, sections)
                
                # Score display with color coding
                if resume_score >= 80:
                    score_color = "#1ed760"
                    score_emoji = "🎉"
                    score_message = "Excellent Resume!"
                elif resume_score >= 60:
                    score_color = "#fba171"
                    score_emoji = "👍"
                    score_message = "Good Resume"
                elif resume_score >= 40:
                    score_color = "#fabc10"
                    score_emoji = "⚠️"
                    score_message = "Needs Improvement"
                else:
                    score_color = "#d73b5c"
                    score_emoji = "❗"
                    score_message = "Significant Improvements Needed"
                
                # Animated progress bar
                col1, col2 = st.columns([3, 1])
                with col1:
                    progress_bar = st.progress(0)
                    for percent_complete in range(resume_score + 1):
                        time.sleep(0.01)
                        progress_bar.progress(percent_complete)
                
                with col2:
                    st.markdown(f"<h1 style='text-align: center; color: {score_color};'>{resume_score}</h1>", 
                               unsafe_allow_html=True)
                
                st.markdown(f"<h2 style='text-align: center; color: {score_color};'>{score_emoji} {score_message}</h2>", 
                           unsafe_allow_html=True)

                # Detailed feedback
                st.subheader("📋 Detailed Feedback")
                
                positive_feedback = [f for f_type, f in feedback if f_type == 'positive']
                warning_feedback = [f for f_type, f in feedback if f_type == 'warning']
                negative_feedback = [f for f_type, f in feedback if f_type == 'negative']
                
                if positive_feedback:
                    with st.expander(f"✅ Strengths ({len(positive_feedback)})", expanded=True):
                        for msg in positive_feedback:
                            st.markdown(f"- {msg}")
                
                if warning_feedback:
                    with st.expander(f"⚠️ Areas for Improvement ({len(warning_feedback)})", expanded=True):
                        for msg in warning_feedback:
                            st.markdown(f"- {msg}")
                
                if negative_feedback:
                    with st.expander(f"❌ Critical Missing Elements ({len(negative_feedback)})", expanded=True):
                        for msg in negative_feedback:
                            st.markdown(f"- {msg}")

                st.markdown("---")

                # Skills Analysis and Recommendations
                st.header("🎯 Career Path & Skills Recommendations")
                
                # Show current skills
                current_skills = resume_data.get('skills', [])
                if current_skills:
                    st.subheader("Your Current Skills")
                    keywords = st_tags(
                        label='',
                        text='Skills extracted from your resume',
                        value=current_skills,
                        key='current_skills'
                    )
                else:
                    st.warning("⚠️ No skills detected. Ensure your resume has a clear 'Skills' section.")

                # Analyze and recommend career field
                recommended_field, recommended_skills, course_list, field_scores = analyze_skills_and_recommend(resume_data, resume_text, sections)
                
                # Show all field scores for transparency
                st.subheader("Career Field Analysis")
                
                # Create visualization of field scores
                if field_scores:
                    score_df = pd.DataFrame({
                        'Field': list(field_scores.keys()),
                        'Match Score': list(field_scores.values())
                    }).sort_values('Match Score', ascending=False)
                    
                    fig = px.bar(score_df, x='Match Score', y='Field', orientation='h',
                                color='Match Score', color_continuous_scale='viridis',
                                title='Your Resume Match Score by Career Field')
                    st.plotly_chart(fig, use_container_width=True)
                
                if recommended_field:
                    st.success(f"🎯 **Primary Recommendation: {recommended_field}**")
                    st.info(f"Based on your skills and experience, you are best suited for **{recommended_field}** roles.")
                    
                    st.subheader(f"Recommended Skills for {recommended_field}")
                    
                    # Filter out skills already present (case-insensitive)
                    current_skills_lower = [s.lower() for s in current_skills]
                    skills_to_add = [skill for skill in recommended_skills 
                                    if skill.lower() not in current_skills_lower]
                    
                    if skills_to_add:
                        # Prioritize skills not in resume
                        recommended_keywords = st_tags(
                            label='',
                            text='Add these skills to strengthen your profile',
                            value=skills_to_add[:15],
                            key='recommended_skills'
                        )
                        st.markdown("<p style='color: #1ed760;'>💡 <b>Adding these skills will significantly boost your job prospects!</b></p>", 
                                   unsafe_allow_html=True)
                    else:
                        st.success("🎉 Excellent! You have most of the recommended skills for this field.")
                    
                    # Course recommendations
                    if course_list:
                        st.markdown("---")
                        rec_course = course_recommender(course_list)
                else:
                    st.warning("💡 Your resume needs more specific technical skills to get targeted recommendations.")
                    st.info("Add skills related to your target field (e.g., Python, React, AWS, Machine Learning, etc.)")

                st.markdown("---")

                # Additional Tips
                with st.expander("📚 General Resume Tips", expanded=False):
                    st.markdown("""
                    ### Resume Best Practices:
                    1. **Use Action Verbs**: Start bullet points with strong verbs (Developed, Managed, Led, Implemented)
                    2. **Quantify Achievements**: Include numbers and metrics (e.g., "Increased efficiency by 30%")
                    3. **Tailor to Job**: Customize your resume for each application
                    4. **Keep it Concise**: 1-2 pages maximum
                    5. **Use Professional Format**: Clear headings, consistent formatting, readable fonts
                    6. **Include Keywords**: Use industry-specific terminology from job descriptions
                    7. **Proofread**: Zero typos and grammatical errors
                    8. **Add Links**: Include LinkedIn, GitHub, portfolio links
                    9. **Update Regularly**: Keep skills and experience current
                    10. **Use White Space**: Don't overcrowd - make it easy to scan
                    """)

                # Store in MongoDB
                timestamp = utils.generateUniqueFileName()
                save = {
                    timestamp: {
                        **resume_data,
                        'analysis': {
                            'career_level': cand_level,
                            'recommended_field': recommended_field,
                            'resume_score': resume_score,
                            'field_scores': field_scores,
                            'analyzed_at': datetime.datetime.now().isoformat()
                        }
                    }
                }
                MongoDB_function.resume_store(save, dataBase, collection)
                
                st.success("✅ Analysis saved to your profile!")

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.info("Please ensure your resume is a valid, text-based PDF.")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())

    else:
        st.info("👆 Upload your resume in PDF format to get started")
        
        # Show sample analysis
        with st.expander("📖 What You'll Get"):
            st.markdown("""
            ### Comprehensive Resume Analysis:
            - ✅ **Resume Score** (0-100) based on completeness and quality
            - 📊 **Career Level** detection (Fresher/Junior/Intermediate/Experienced)
            - 🎯 **Career Field** recommendations with match scores
            - 💼 **Skills Gap** analysis
            - 📚 **Course Recommendations** tailored to your profile
            - 💡 **Actionable Feedback** for improvement
            - 📈 **Visual Analytics** of your profile strengths
            """)


run()