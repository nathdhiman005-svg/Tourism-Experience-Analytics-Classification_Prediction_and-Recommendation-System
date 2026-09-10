import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridRecommender:
    def __init__(self, data_path="../processed_data/modeling_ready_data.csv", models_dir="../models"):
        self.data_path = (Path(__file__).parent / data_path).resolve()
        self.models_dir = (Path(__file__).parent / models_dir).resolve()
        
        # Load Data
        logger.info("Loading dataset for Recommender Engine...")
        self.df = pd.read_csv(self.data_path)
        
        # Determine the correct VisitMode column name
        self.visit_mode_col = 'VisitModeId' if 'VisitModeId' in self.df.columns else 'VisitMode_id'
        
        # Extract unique attraction metadata for predictions
        self.attractions_meta = self.df[['AttractionId', 'AttractionType', 'AttractionAvgRating', 'AttractionPopularity']].drop_duplicates().set_index('AttractionId')
        
        # Pre-calculate Visit Mode Affinity (Average rating per attraction for each Visit Mode)
        self.mode_affinity = self.df.groupby([self.visit_mode_col, 'AttractionId'])['Rating'].mean().unstack().fillna(0)
        
        # Initialize engines
        self._build_similarity_matrix()
        self._load_models()

    def _build_similarity_matrix(self):
        """Builds the 30x30 Item-Item Cosine Similarity Matrix (Collaborative Filtering Base)"""
        logger.info("Building Item-Item Collaborative Filtering Matrix...")
        # Create User-Item Matrix
        user_item_matrix = self.df.pivot_table(index='UserId', columns='AttractionId', values='Rating').fillna(0)
        
        # Compute Cosine Similarity between Items (Attractions)
        # transpose so attractions are rows
        item_sim_array = cosine_similarity(user_item_matrix.T)
        
        self.item_similarity_df = pd.DataFrame(
            item_sim_array, 
            index=user_item_matrix.columns, 
            columns=user_item_matrix.columns
        )
        logger.info(f"Similarity Matrix built successfully. Shape: {self.item_similarity_df.shape}")

    def _load_models(self):
        """Loads the predictive ML models (Content-Based/Cold Start Base)"""
        logger.info("Loading Predictive ML Models...")
        self.classifier = joblib.load(self.models_dir / "visit_mode_classifier.pkl")
        self.regressor = joblib.load(self.models_dir / "rating_regressor.pkl")
        
        # Store feature structures to ensure perfectly aligned inputs
        self.cls_features = ['UserContinent', 'UserRegion', 'UserCountry', 'AttractionType', 'VisitSeason', 'UserAvgRating', 'UserTotalVisits', 'AttractionAvgRating', 'AttractionPopularity']
        self.reg_features = self.cls_features + [self.visit_mode_col]

    def _predict_ratings_for_new_user(self, user_profile: dict):
        """Uses ML Models to predict ratings for all 30 attractions for a brand new user"""
        
        # Step 1: Prepare data for Classification (Predict VisitMode)
        # We need to predict VisitMode for all 30 attractions, then use that to predict Rating.
        # Create a dataframe with 30 rows (one for each attraction) combining user_profile + attraction_meta
        
        rows = []
        for att_id, row in self.attractions_meta.iterrows():
            merged_profile = user_profile.copy()
            merged_profile['AttractionId'] = att_id
            merged_profile['AttractionType'] = row['AttractionType']
            merged_profile['AttractionAvgRating'] = row['AttractionAvgRating']
            merged_profile['AttractionPopularity'] = row['AttractionPopularity']
            rows.append(merged_profile)
            
        pred_df = pd.DataFrame(rows)
        
        # Predict VisitMode
        pred_df[self.visit_mode_col] = self.classifier.predict(pred_df[self.cls_features])
        
        # Step 2: Prepare data for Regression (Predict Rating)
        predicted_ratings = self.regressor.predict(pred_df[self.reg_features])
        
        # Return a series of predicted ratings indexed by AttractionId
        return pd.Series(predicted_ratings, index=pred_df['AttractionId'])

    def _get_collaborative_scores(self, past_ratings: dict):
        """Calculates collaborative filtering scores based on similarity to past highly-rated items"""
        cf_scores = pd.Series(0.0, index=self.attractions_meta.index)
        
        if not past_ratings:
            return cf_scores
            
        total_similarity = pd.Series(0.0, index=self.attractions_meta.index)
        
        for att_id, rating in past_ratings.items():
            if att_id not in self.item_similarity_df.columns:
                continue
                
            # We only want to find similar items to things they actually LIKED (e.g. 4 or 5 stars)
            # If they gave a 1 star, we don't want to recommend similar things.
            if rating < 3:
                continue
                
            # Get similarity of this attraction to all other attractions
            similarities = self.item_similarity_df[att_id]
            
            # Weighted sum: similarity * user_rating
            cf_scores += similarities * rating
            total_similarity += similarities
            
        # Normalize by total similarity to keep scores on a 1-5 scale
        # Avoid division by zero
        total_similarity = total_similarity.replace(0, 1)
        cf_scores = cf_scores / total_similarity
        
        return cf_scores

    def get_recommendations(self, user_profile: dict, past_ratings: dict = None, top_n: int = 5):
        """
        The Core Hybrid Engine Logic.
        Seamlessly blends Predictive (ML) and Collaborative scores based on user history.
        """
        if past_ratings is None:
            past_ratings = {}
            
        logger.info(f"Generating recommendations... (Past Ratings Found: {len(past_ratings)})")
        
        # 1. Always get Predictive ML Scores (Cold Start baseline)
        ml_scores = self._predict_ratings_for_new_user(user_profile)
        
        # --- DIVERSITY BOOSTER ---
        # 1a. Predict the user's Visit Mode based on their demographics
        dummy_row = user_profile.copy()
        dummy_row['AttractionType'] = 1
        dummy_row['AttractionAvgRating'] = 4.0
        dummy_row['AttractionPopularity'] = 50
        df_cls = pd.DataFrame([dummy_row])
        pred_mode = self.classifier.predict(df_cls[self.cls_features])[0]
        
        # 1b. Inject Demographic Affinity Boost
        if pred_mode in self.mode_affinity.index:
            affinity_boost = self.mode_affinity.loc[pred_mode]
            # Boost the predicted rating by up to 1.5 stars for attractions highly rated by this specific Visit Mode
            ml_scores = ml_scores + ((affinity_boost / 5.0) * 1.5)
            
        # 1c. Penalize Global Popularity to allow niche places to shine
        top_popular = self.attractions_meta.sort_values('AttractionPopularity', ascending=False).head(5).index
        ml_scores.loc[ml_scores.index.isin(top_popular)] -= 0.5
        # -------------------------
        
        # 2. If the user is brand new, rely 100% on ML + Diversity.
        if len(past_ratings) == 0:
            final_scores = ml_scores
            engine_used = "Predictive ML + Diversity Booster (Cold Start)"
            
        # 3. If the user is returning, use the Hybrid Blend!
        else:
            cf_scores = self._get_collaborative_scores(past_ratings)
            
            # Hybrid Blend: 50% Machine Learning Prediction, 50% Collaborative Similarity
            final_scores = (ml_scores * 0.5) + (cf_scores * 0.5)
            engine_used = "Hybrid Engine (50% ML, 50% CF)"
            
        # 4. Remove attractions the user has already visited/rated
        for att_id in past_ratings.keys():
            if att_id in final_scores:
                final_scores.drop(att_id, inplace=True)
                
        # 5. Sort and get Top N
        top_recommendations = final_scores.sort_values(ascending=False).head(top_n)
        
        logger.info(f"Recommendations successfully generated using: {engine_used}")
        return top_recommendations


if __name__ == "__main__":
    recommender = HybridRecommender()
    
    # ---------------------------------------------------------
    # TEST 1: The "Cold Start" New User (Zero History)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("TEST 1: BRAND NEW USER (Cold Start)")
    print("="*50)
    
    # We pass their demographics but NO past ratings
    new_user_profile = {
        'UserContinent': 3,       # e.g., Asia
        'UserRegion': 12,         # e.g., East Asia
        'UserCountry': 45,        # e.g., Japan
        'VisitSeason': 4,         # e.g., Winter
        'UserAvgRating': 4.5,     # They seem generally optimistic
        'UserTotalVisits': 1      # First time!
    }
    
    new_user_recs = recommender.get_recommendations(new_user_profile, past_ratings={})
    print("\nTop 5 Recommendations for New User:")
    for att_id, score in new_user_recs.items():
        print(f"Attraction {att_id}: Predicted Rating {score:.2f} stars")
        
        
    # ---------------------------------------------------------
    # TEST 2: The "Returning" Old User (Rich History)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("TEST 2: RETURNING USER (Hybrid Engine)")
    print("="*50)
    
    # Same user, but now they have visited 3 attractions and rated them highly
    past_history = {
        10: 5.0,  # Loved Attraction 10
        12: 4.0,  # Liked Attraction 12
        25: 1.0   # Hated Attraction 25
    }
    
    returning_user_recs = recommender.get_recommendations(new_user_profile, past_ratings=past_history)
    print("\nTop 5 Recommendations for Returning User:")
    for att_id, score in returning_user_recs.items():
        print(f"Attraction {att_id}: Blended Score {score:.2f}")

    print("\n" + "="*50)
