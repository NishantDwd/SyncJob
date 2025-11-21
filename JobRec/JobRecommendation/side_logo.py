import streamlit as st
from PIL import Image

def add_logo(logo_path="logo/image.png", width=300, height=100):
    """Add logo to sidebar with improved positioning"""
    
    st.markdown("""
        <style>
        /* Sidebar gradient background - applied immediately */
        [data-testid="stSidebar"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        }
        
        /* Logo and navigation styling */
        [data-testid="stSidebarNav"] {
            padding-top: 2rem;
            position: relative;
            background: transparent !important;
        }
        
        /* Logo container */
        [data-testid="stSidebarNav"]::before {
            content: "🔄 SYNCJOB";
            display: block;
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            color: #ffffff;
            padding: 20px 10px;
            margin: 0 15px 30px 15px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            border: 2px solid rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
        }
        
        /* Navigation items spacing */
        [data-testid="stSidebarNav"] > ul {
            padding-top: 1rem;
            background: transparent !important;
        }
        
        /* Style navigation links */
        [data-testid="stSidebarNav"] a {
            background: rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            margin: 5px 15px !important;
            padding: 12px 15px !important;
            color: #ffffff !important;
            transition: all 0.3s ease !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
        }
        
        [data-testid="stSidebarNav"] a:hover {
            background: rgba(255, 255, 255, 0.25) !important;
            transform: translateX(5px);
            border-left: 4px solid #ffd700 !important;
        }
        
        /* Active page highlight */
        [data-testid="stSidebarNav"] a[aria-current="page"] {
            background: rgba(255, 215, 0, 0.3) !important;
            border-left: 4px solid #ffd700 !important;
            font-weight: bold;
        }
        
        /* Force sidebar background on all children */
        [data-testid="stSidebar"] > div {
            background: transparent !important;
        }
        
        /* Remove any default backgrounds */
        [data-testid="stSidebarNav"],
        [data-testid="stSidebarNav"] > div,
        [data-testid="stSidebarNav"] ul {
            background: transparent !important;
        }
        </style>
    """, unsafe_allow_html=True)