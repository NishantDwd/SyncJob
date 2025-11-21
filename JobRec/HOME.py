# # Core Pkgs
# import streamlit as st 
# from JobRecommendation.side_logo import add_logo
# from JobRecommendation.sidebar import sidebar
# import altair as alt
# import plotly.express as px 
# from streamlit_extras.switch_page_button import switch_page
# from JobRecommendation.lottie_animation import load_lottieurl
# # EDA Pkgs
# import pandas as pd 
# import numpy as np 
# from datetime import datetime
# from streamlit_lottie import st_lottie


# st.set_page_config(layout="centered", page_icon='logo/logo2.png', page_title="HOMEPAGE")



# url = load_lottieurl("https://assets5.lottiefiles.com/private_files/lf30_m075yjya.json")

# add_logo()
# sidebar()


# st.markdown("<h1 style='text-align: center; font-family: Verdana, sans-serif; padding: 20px; border: 2px solid #758283; border-radius: 5px;'>Welcome to Talent Hive !</h1>", unsafe_allow_html=True)

# st.markdown("<div style='background-color: rgba(255, 0, 0, 0); padding: 10px;'>", unsafe_allow_html=True)
# st.markdown("<h2 style='text-align: center; font-family: Verdana, sans-serif; padding: 20px;'>WHAT WE OFFER : </h2>", unsafe_allow_html=True)

# s1,s2, s3 = st.columns(3)
# with s1:

#     candidate = st.button("Job Recomendation")
#     if candidate:
#         switch_page("i am a candidate")
#     st.balloons()

# with s2:
#         analyzer = st.button("Resume Analyzer")
#         if analyzer:
#             switch_page("resume analyzer")
#         st.balloons()

# with s3:
#         recruiter = st.button("Candidate Recomendation")
#         if recruiter:
#             switch_page("i am a recruiter")
#         st.balloons()



# # st.image('logo/TALENTHIVE.png', use_column_width="auto")
# st.markdown("</div>", unsafe_allow_html=True)
# st_lottie(url)

# # Project Description Section
# st.markdown("<h2 style='text-align: center; font-family: Verdana, sans-serif; padding: 20px;'>Why Talent Hive ?</h2>", unsafe_allow_html=True)
# st.write("<p style='font-size:20px;'>Job seekers and recruiters struggle to find the right match for open job positions, leading to a time-consuming and inefficient recruitment process. TalentHive offers a solution to this problem with its advanced technologies that provide personalized job and candidate recommendations based on qualifications and experience.</p>", unsafe_allow_html=True)

# st.markdown("<h2 style='text-align: center; font-family: Verdana, sans-serif; padding: 20px;'>AIM</h2>", unsafe_allow_html=True)
# st.write("<p style='font-size:20px;'>The job search process can be daunting and time-consuming for both job seekers and recruiters. That's where this app comes in!", unsafe_allow_html=True)
# st.write("<p style='font-size:20px;'>This app is designed to assist applicants in searching for potential jobs and to help recruiters find talented candidates. The app offers a user-friendly interface that allows applicants to easily browse and search for job opportunities based on their preferences and qualifications. Users can create a profile, upload their resumes, and set up job alerts to receive notifications about new job postings that match their criteria. The app also provides helpful tips and resources for applicants, such as Resume Analyzer and tips to make your Resume even better !! ", unsafe_allow_html=True)

# # Set footer config

# # # Set footer config
# # footer = "<div style='background-color: #758283; padding: 10px; color: white; text-align: center; position: absolute; bottom: 0; width: 100%;'>© 2023 TalentHive</div>"
# # st.markdown(footer, unsafe_allow_html=True)


# # Create a container for the footer
# footer_container = st.container()

# # Add content to the footer container
# with footer_container:

#     st.write("Github @ <a href='https://github.com/Ryzxxl/Job-Recomendation'>Repository</a>", unsafe_allow_html=True)

import streamlit as st 
from JobRecommendation.side_logo import add_logo
from JobRecommendation.sidebar import sidebar
from streamlit_extras.switch_page_button import switch_page
from JobRecommendation.lottie_animation import load_lottieurl
from streamlit_lottie import st_lottie

st.set_page_config(
    layout="wide", 
    page_icon='🔄', 
    page_title="SyncJob - Home",
    initial_sidebar_state="expanded"
)

# Custom CSS for the entire page
st.markdown("""
    <style>
    /* Main content area */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 60px 40px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 40px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }
    
    .hero-title {
        font-size: 56px;
        font-weight: bold;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        animation: fadeInDown 1s ease-in;
    }
    
    .hero-subtitle {
        font-size: 24px;
        margin-bottom: 10px;
        opacity: 0.95;
    }
    
    .hero-description {
        font-size: 18px;
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.6;
        opacity: 0.9;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 5px 25px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        height: 100%;
        border-left: 5px solid;
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }
    
    .feature-card-candidate {
        border-left-color: #667eea;
    }
    
    .feature-card-analyzer {
        border-left-color: #f093fb;
    }
    
    .feature-card-recruiter {
        border-left-color: #4facfe;
    }
    
    .feature-icon {
        font-size: 48px;
        margin-bottom: 15px;
    }
    
    .feature-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 15px;
        color: #333;
    }
    
    .feature-description {
        font-size: 16px;
        color: #666;
        line-height: 1.6;
        margin-bottom: 20px;
    }
    
    /* Stats section */
    .stats-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px;
        border-radius: 20px;
        margin: 40px 0;
        color: white;
    }
    
    .stat-box {
        text-align: center;
        padding: 20px;
    }
    
    .stat-number {
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .stat-label {
        font-size: 18px;
        opacity: 0.9;
    }
    
    /* Section headers */
    .section-header {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin: 50px 0 30px 0;
        color: #333;
        position: relative;
    }
    
    .section-header::after {
        content: '';
        display: block;
        width: 100px;
        height: 4px;
        background: linear-gradient(to right, #667eea, #764ba2);
        margin: 15px auto;
        border-radius: 2px;
    }
    
    /* Benefits list */
    .benefit-item {
        background: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 3px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .benefit-item:hover {
        transform: translateX(10px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.15);
    }
    
    .benefit-icon {
        font-size: 24px;
        margin-right: 15px;
    }
    
    .benefit-text {
        font-size: 16px;
        color: #333;
    }
    
    /* CTA Button styling */
    .stButton > button {
        width: 100%;
        height: 60px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 30px;
        border: none;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-top: 60px;
    }
    
    .footer-text {
        font-size: 16px;
        margin-bottom: 10px;
    }
    
    .footer-links {
        margin-top: 20px;
    }
    
    .footer-link {
        color: white;
        margin: 0 15px;
        text-decoration: none;
        transition: all 0.3s ease;
    }
    
    .footer-link:hover {
        color: #ffd700;
    }
    </style>
""", unsafe_allow_html=True)

url = load_lottieurl("https://assets5.lottiefiles.com/private_files/lf30_m075yjya.json")

add_logo()
sidebar()

# Hero Section
st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🔄 Welcome to SyncJob</div>
        <div class="hero-subtitle">Where Talent Meets Opportunity</div>
        <div class="hero-description">
            Revolutionizing recruitment with AI-powered matching. Connect job seekers with their dream roles 
            and help recruiters discover exceptional talent—all in one intelligent platform.
        </div>
    </div>
""", unsafe_allow_html=True)

# Stats Section
st.markdown("""
    <div class="stats-container">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
            <div class="stat-box">
                <div class="stat-number">10K+</div>
                <div class="stat-label">Jobs Available</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">95%</div>
                <div class="stat-label">Match Accuracy</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">5K+</div>
                <div class="stat-label">Active Users</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">24/7</div>
                <div class="stat-label">AI Support</div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Main Features Section
st.markdown('<div class="section-header">🚀 Choose Your Journey</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="feature-card feature-card-candidate">
            <div class="feature-icon">👨‍🎓</div>
            <div class="feature-title">Job Seekers</div>
            <div class="feature-description">
                Discover personalized job recommendations powered by AI. 
                Upload your resume and let our smart algorithms find your perfect match.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔍 Find My Dream Job", key="candidate", use_container_width=True):
        switch_page("i am a candidate")
        st.balloons()

with col2:
    st.markdown("""
        <div class="feature-card feature-card-analyzer">
            <div class="feature-icon">📄</div>
            <div class="feature-title">Resume Analyzer</div>
            <div class="feature-description">
                Get instant feedback on your resume. Our AI analyzes content, 
                format, and keywords to help you stand out.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("✨ Analyze My Resume", key="analyzer", use_container_width=True):
        switch_page("resume analyzer")
        st.balloons()

with col3:
    st.markdown("""
        <div class="feature-card feature-card-recruiter">
            <div class="feature-icon">🧑‍💼</div>
            <div class="feature-title">Recruiters</div>
            <div class="feature-description">
                Find top talent effortlessly. Post job descriptions and receive 
                AI-ranked candidate recommendations instantly.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🎯 Find Top Talent", key="recruiter", use_container_width=True):
        switch_page("i am a recruiter")
        st.balloons()

# Lottie Animation
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st_lottie(url, height=300, key="main_animation")

# Why SyncJob Section
st.markdown('<div class="section-header">💡 Why Choose SyncJob?</div>', unsafe_allow_html=True)

benefits = [
    ("🎯", "Smart Matching Algorithm", "Our AI analyzes thousands of data points for perfect matches"),
    ("⚡", "Lightning Fast", "Get results in seconds, not days"),
    ("🔒", "Secure & Private", "Your data is encrypted and protected"),
    ("📊", "Data-Driven Insights", "Make informed decisions with analytics"),
    ("🌐", "Wide Job Network", "Access opportunities from multiple sources"),
    ("🤝", "User-Friendly", "Intuitive interface for seamless experience"),
]

col1, col2 = st.columns(2)
for i, (icon, title, desc) in enumerate(benefits):
    with col1 if i % 2 == 0 else col2:
        st.markdown(f"""
            <div class="benefit-item">
                <span class="benefit-icon">{icon}</span>
                <span class="benefit-text"><strong>{title}:</strong> {desc}</span>
            </div>
        """, unsafe_allow_html=True)

# How It Works Section
st.markdown('<div class="section-header">🔄 How It Works</div>', unsafe_allow_html=True)

steps_col1, steps_col2, steps_col3 = st.columns(3)

with steps_col1:
    st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 48px; margin-bottom: 15px;">1️⃣</div>
            <h3>Upload</h3>
            <p>Submit your resume or job description</p>
        </div>
    """, unsafe_allow_html=True)

with steps_col2:
    st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 48px; margin-bottom: 15px;">2️⃣</div>
            <h3>Analyze</h3>
            <p>AI processes and matches using NLP</p>
        </div>
    """, unsafe_allow_html=True)

with steps_col3:
    st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 48px; margin-bottom: 15px;">3️⃣</div>
            <h3>Connect</h3>
            <p>Get personalized recommendations</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <div class="footer-text">
            <strong>SyncJob</strong> - Powered by AI & Machine Learning
        </div>
        <div class="footer-text" style="font-size: 14px; opacity: 0.8;">
            Connecting talent with opportunity, one match at a time
        </div>
        <div class="footer-links">
            <a href="https://github.com/NishantDwd/SyncJob" target="_blank" class="footer-link">GitHub</a>
            <span style="color: rgba(255,255,255,0.5);">|</span>
            <a href="#" class="footer-link">Documentation</a>
            <span style="color: rgba(255,255,255,0.5);">|</span>
            <a href="#" class="footer-link">Privacy</a>
        </div>
        <div style="margin-top: 20px; font-size: 14px;">
            © 2024 SyncJob. All rights reserved.
        </div>
    </div>
""", unsafe_allow_html=True)