import streamlit as st

def sidebar():
    st.markdown("""
        <style>
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Sidebar content styling */
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 2rem;
        }
        
        /* About section styling */
        .sidebar-about {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 20px;
            margin: 20px 15px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        
        .sidebar-title {
            color: #ffffff;
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 15px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .sidebar-text {
            color: #f0f0f0;
            font-size: 14px;
            line-height: 1.6;
            text-align: center;
        }
        
        /* Link styling */
        .sidebar-link {
            display: block;
            color: #ffffff;
            background: rgba(255, 255, 255, 0.2);
            padding: 12px 15px;
            border-radius: 12px;
            text-decoration: none;
            margin: 10px 0;
            text-align: center;
            transition: all 0.3s ease;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        .sidebar-link:hover {
            background: rgba(255, 255, 255, 0.3);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        
        /* Features list */
        .feature-item {
            color: #ffffff;
            padding: 10px 15px;
            margin: 8px 0;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            border-left: 4px solid #ffd700;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
        <div class="sidebar-about">
            <div class="sidebar-title">🔄 About SyncJob</div>
            <div class="sidebar-text">
                Your AI-powered recruitment companion connecting talent with opportunity.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
        <div class="sidebar-about">
            <div class="sidebar-title">✨ Key Features</div>
            <div class="feature-item">🎯 Smart Job Matching</div>
            <div class="feature-item">📄 Resume Analysis</div>
            <div class="feature-item">🤖 AI Recommendations</div>
            <div class="feature-item">📊 Career Insights</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
        <div class="sidebar-about">
            <div class="sidebar-title">🔗 Connect With Us</div>
            <a href="https://github.com/NishantDwd/SyncJob" target="_blank" class="sidebar-link">
                💻 GitHub Repository
            </a>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
        <div class="sidebar-about">
            <div class="sidebar-text" style="font-size: 12px; margin-top: 20px;">
                © 2025 SyncJob<br>
                Powered by AI & Machine Learning
            </div>
        </div>
    """, unsafe_allow_html=True)