import streamlit as st

def render(engine, mappings):
    continent_map, region_map, country_map, attraction_map = mappings
    
    st.title("🗺️ Your Personal Travel Guide")
    st.markdown("Welcome! Let us find the perfect attractions for your next adventure using our Predictive AI.")
    
    st.subheader("👤 Tell us a bit about yourself")
    
    col1, col2 = st.columns(2)
    with col1:
        continent_name = st.selectbox("Continent", options=list(continent_map.keys()))
        region_name = st.selectbox("Region", options=list(region_map.keys()))
        country_name = st.selectbox("Country", options=list(country_map.keys()))
        
    with col2:
        season = st.selectbox("Preferred Season to Travel", options=[1, 2, 3, 4], format_func=lambda x: {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}[x])
        total_visits = st.number_input("How many places have you visited before?", min_value=0, max_value=100, value=0)
        
        if total_visits > 0:
            avg_rating = st.slider("How do you typically rate places?", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
        else:
            avg_rating = 4.0
            
    submit = st.button("✨ Discover My Destinations")
    
    if submit:
        user_profile = {
            'UserContinent': continent_map[continent_name],
            'UserRegion': region_map[region_name],
            'UserCountry': country_map[country_name],
            'VisitSeason': season,
            'UserAvgRating': avg_rating,
            'UserTotalVisits': total_visits
        }
        
        with st.spinner("Analyzing your profile using our Predictive Engine..."):
            recs = engine.get_recommendations(user_profile, past_ratings={}, top_n=5)
            
            st.success("🎯 Here are your Top 5 Personalized Recommendations!")
            
            cols = st.columns(5)
            for i, (att_id, score) in enumerate(recs.items()):
                att_name = attraction_map.get(att_id, f"Place {att_id}")
                with cols[i]:
                    st.markdown(f"<div class='rec-card'><h3>📍 {att_name}</h3></div>", unsafe_allow_html=True)
