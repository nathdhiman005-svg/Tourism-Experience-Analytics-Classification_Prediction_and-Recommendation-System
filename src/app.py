import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Ensure src module is in path so we can import recommenders and custom modules
sys.path.append(str(Path(__file__).parent))
from recommenders import HybridRecommender
from components import ui_helpers
from views import consumer_portal, admin_portal, auth_portal

# Set page config
st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS
ui_helpers.inject_custom_css()

# --- INITIALIZE SESSION STATE FOR AUTH ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'role' not in st.session_state:
    st.session_state.role = None
if 'user_email' not in st.session_state:
    st.session_state.user_email = None

# --- CACHING THE BACKEND ENGINE ---
@st.cache_resource
def load_engine():
    """Loads the HybridRecommender into memory exactly once per session."""
    return HybridRecommender()

@st.cache_data
def get_mappings():
    raw_path = Path(__file__).parent / "../processed_data/raw_merged.csv"
    mod_path = Path(__file__).parent / "../processed_data/modeling_ready_data.csv"
    
    raw = pd.read_csv(raw_path)
    mod = pd.read_csv(mod_path)
    
    continent_map = {}
    for name in raw['UserContinent'].dropna().unique():
        idx = raw[raw['UserContinent'] == name].index[0]
        continent_map[name] = mod.loc[idx, 'UserContinent']
        
    region_map = {}
    for name in raw['UserRegion'].dropna().unique():
        idx = raw[raw['UserRegion'] == name].index[0]
        region_map[name] = mod.loc[idx, 'UserRegion']
        
    country_map = {}
    for name in raw['UserCountry'].dropna().unique():
        idx = raw[raw['UserCountry'] == name].index[0]
        country_map[name] = mod.loc[idx, 'UserCountry']
        
    attraction_map = {}
    if 'Attraction' in raw.columns:
        for att_id in raw['AttractionId'].unique():
            name = raw[raw['AttractionId'] == att_id]['Attraction'].iloc[0]
            attraction_map[att_id] = name
            
    return (continent_map, region_map, country_map, attraction_map)

# --- MAIN APP ROUTING ---
if not st.session_state.authenticated:
    # If not logged in, show ONLY the auth portal
    auth_portal.render()
else:
    # --- LOAD HEAVY ASSETS ONLY IF LOGGED IN ---
    try:
        engine = load_engine()
        mappings = get_mappings()
    except Exception as e:
        st.error(f"Failed to load the Recommendation Engine or Mappings. Error: {e}")
        st.stop()

    # --- PROTECTED SIDEBAR ---
    st.sidebar.title("🌍 Navigation")
    st.sidebar.markdown(f"Logged in as: **{st.session_state.user_email}**")
    st.sidebar.markdown(f"Role: **{st.session_state.role}**")
    
    if st.sidebar.button("Log Out"):
        st.session_state.authenticated = False
        st.session_state.role = None
        st.session_state.user_email = None
        st.rerun()
        
    st.sidebar.markdown("---")
    
    # --- ROLE BASED ROUTING ---
    if st.session_state.role == "Admin":
        admin_portal.render(engine, mappings)
    else:
        consumer_portal.render(engine, mappings)
