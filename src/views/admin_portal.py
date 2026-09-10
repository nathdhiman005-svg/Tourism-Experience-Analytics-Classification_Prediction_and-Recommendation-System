import streamlit as st
import pandas as pd

def render(engine, mappings):
    continent_map, region_map, country_map, attraction_map = mappings
    
    st.title("📊 B2B Analytics Dashboard")
    st.markdown("Welcome to the Admin Portal. Simulate a demographic to predict travel behavior and identify operational bottlenecks.")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Simulate Demographic")
    continent_name = st.sidebar.selectbox("Continent", options=list(continent_map.keys()))
    region_name = st.sidebar.selectbox("Region", options=list(region_map.keys()))
    country_name = st.sidebar.selectbox("Country", options=list(country_map.keys()))
    season = st.sidebar.selectbox("Visit Season", options=[1, 2, 3, 4], format_func=lambda x: {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}[x])
    total_visits = st.sidebar.number_input("How many places have they visited?", min_value=0, max_value=100, value=0)
    
    if total_visits > 0:
        avg_rating = st.sidebar.slider("Historical Avg Rating", min_value=1.0, max_value=5.0, value=4.0)
    else:
        avg_rating = 4.0
        
    user_profile = {
        'UserContinent': continent_map[continent_name],
        'UserRegion': region_map[region_name],
        'UserCountry': country_map[country_name],
        'VisitSeason': season,
        'UserAvgRating': avg_rating,
        'UserTotalVisits': total_visits
    }
    
    # 1. Predict Visit Mode (Classification)
    st.header("1. Marketing & Resource Predictor")
    
    # To get a realistic overall Visit Mode for this demographic, we predict it across all 30 attractions and take the majority vote
    rows = []
    for att_id, row in engine.attractions_meta.iterrows():
        merged_profile = user_profile.copy()
        merged_profile['AttractionType'] = row['AttractionType']
        merged_profile['AttractionAvgRating'] = row['AttractionAvgRating']
        merged_profile['AttractionPopularity'] = row['AttractionPopularity']
        rows.append(merged_profile)
    
    df_cls = pd.DataFrame(rows)
    pred_modes = engine.classifier.predict(df_cls[engine.cls_features])
    pred_mode_id = pd.Series(pred_modes).mode()[0]
    
    # Map ID to actual string
    visit_mode_map = {1: 'Business', 2: 'Couples', 3: 'Family', 4: 'Friends', 5: 'Solo'}
    mode_str = visit_mode_map.get(pred_mode_id, f"Mode {pred_mode_id}")
    
    st.info(f"**Predicted Visitor Type:** {mode_str}")
    
    # Dynamic Actionable Insight
    insights = {
        'Family': "💡 **Actionable Insight:** Deploy family-friendly packages and ensure accommodations have child-friendly amenities.",
        'Solo': "💡 **Actionable Insight:** Promote solo adventure activities, hostel mixers, and safety-focused travel guides.",
        'Couples': "💡 **Actionable Insight:** Highlight romantic getaways, premium dining experiences, and secluded attractions.",
        'Friends': "💡 **Actionable Insight:** Market group discounts, nightlife activities, and multi-person adventure tours.",
        'Business': "💡 **Actionable Insight:** Focus on fast Wi-Fi, proximity to convention centers, and express services."
    }
    st.write(insights.get(mode_str, "💡 **Actionable Insight:** Tailor your marketing campaigns immediately based on this visitor type!"))
    
    st.markdown("---")
    
    # 2. Predict Ratings for Quality Control (Regression)
    st.header("2. Satisfaction & Quality Control")
    st.write("Below are the predicted satisfaction ratings across all 30 attractions for this specific demographic:")
    
    all_ratings = engine._predict_ratings_for_new_user(user_profile)
    
    # Map Attraction IDs to Actual Names for the chart and round to 2 decimal places
    all_ratings_named = all_ratings.rename(index=attraction_map).round(2)
    all_ratings_named.index.name = "Places"
    all_ratings_named.name = "Predicted Rating"
    
    st.bar_chart(all_ratings_named)
    
    low_scoring = all_ratings[all_ratings < 3.5]
    if not low_scoring.empty:
        st.warning("⚠️ **Alert: Low Predicted Satisfaction**")
        st.write("The following attractions are predicted to receive lower ratings from this demographic. Agencies should take corrective actions, such as improving services or better setting user expectations.")
        
        low_scoring_named = low_scoring.rename(index=attraction_map).round(2)
        
        # Convert to DataFrame to properly name columns for the frontend table
        df_low = low_scoring_named.reset_index()
        df_low.columns = ["Places", "Predicted Rating"]
        
        st.dataframe(df_low, hide_index=True)
    else:
        st.success("✅ All attractions are predicted to perform well for this demographic.")
