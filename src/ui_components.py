# modules/ui_components.py
import streamlit as st   

def setup_ui():
    """Configure the UI settings and apply custom CSS"""
    st.set_page_config(page_title="Timetable Generator", layout="wide")
    # Custom CSS to improve the appearance
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            text-align: center;
            margin-bottom: 1rem;
            color: #1E88E5;
        }
        .tab-subheader {
            font-size: 1.8rem;
            margin-bottom: 1rem;
            color: #0D47A1;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            font-size: 1.1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">📅 Timetable Generator </div>', unsafe_allow_html=True)

def create_tabs(tab_names):
    """Create Streamlit tabs with the given names"""
    return st.tabs(tab_names)