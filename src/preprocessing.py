import pandas as pd
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    A class to handle data cleaning, feature engineering, and encoding.
    """
    def __init__(self, data_path: str = "../processed_data/raw_merged.csv"):
        self.data_path = (Path(__file__).parent / data_path).resolve()
        
    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.data_path}")
        return pd.read_csv(self.data_path)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles missing values, drops duplicates, and standardizes string formats.
        """
        logger.info(f"Starting data cleaning. Initial shape: {df.shape}")
        
        # 1. Drop Duplicates
        initial_len = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_len:
            logger.info(f"Dropped {initial_len - len(df)} duplicate rows.")
            
        # 2. Handle Missing Values
        # As agreed, we are dropping the ~8 records with missing CityId/UserCity
        missing_count = df.isnull().any(axis=1).sum()
        df = df.dropna()
        logger.info(f"Dropped {missing_count} rows with missing values.")
        
        # 3. Correct String Inconsistencies
        # Strip leading/trailing whitespaces from all object columns
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            df[col] = df[col].astype(str).str.strip()
            
        logger.info(f"Data cleaning complete. Final shape: {df.shape}")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts temporal features and aggregates user/item interaction statistics.
        """
        logger.info("Starting feature engineering...")
        
        # 1. Temporal Features
        # Map VisitMonth to Seasons (Assuming Northern Hemisphere for general grouping)
        # Dec, Jan, Feb = Winter (1); Mar, Apr, May = Spring (2); Jun, Jul, Aug = Summer (3); Sep, Oct, Nov = Fall (4)
        season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                      3: 'Spring', 4: 'Spring', 5: 'Spring',
                      6: 'Summer', 7: 'Summer', 8: 'Summer',
                      9: 'Fall', 10: 'Fall', 11: 'Fall'}
        df['VisitSeason'] = df['VisitMonth'].map(season_map)
        
        # 2. User Interaction Statistics
        # Calculate User's Average Rating
        user_stats = df.groupby('UserId').agg(
            UserAvgRating=('Rating', 'mean'),
            UserTotalVisits=('AttractionId', 'count')
        ).reset_index()
        
        # Merge back to df
        df = df.merge(user_stats, on='UserId', how='left')
        
        # 3. Item (Attraction) Interaction Statistics
        item_stats = df.groupby('AttractionId').agg(
            AttractionAvgRating=('Rating', 'mean'),
            AttractionPopularity=('UserId', 'count')
        ).reset_index()
        
        # Merge back to df
        df = df.merge(item_stats, on='AttractionId', how='left')
        
        logger.info(f"Feature engineering complete. Added {df.shape[1] - (len(df.columns) - 5)} new features. Final shape: {df.shape}")
        return df

    def encode_and_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes categorical variables and scales numerical variables for Machine Learning.
        """
        logger.info("Starting encoding and scaling...")
        df_encoded = df.copy()
        
        # 1. Label Encoding for Categorical Variables
        # We use Label Encoding instead of One-Hot to prevent the dataset from expanding to 5,000+ columns (e.g. UserCity)
        object_cols = df_encoded.select_dtypes(include=['object']).columns
        # Exclude the target variable VisitModeName if it exists, though it's fine to encode it.
        # We already have VisitModeId, so we don't necessarily need to encode VisitModeName, but let's do it for consistency.
        label_encoders = {}
        for col in object_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
            
        # 2. Scaling Numerical Variables
        # We won't scale ID columns or the Target variables (Rating, VisitModeId)
        exclude_cols = ['TransactionId', 'UserId', 'AttractionId', 'VisitMode_id', 'VisitModeId', 'Rating', 'VisitYear', 'VisitMonth', 'ContinentId', 'RegionId', 'CountryId', 'CityId', 'AttractionCityId', 'AttractionTypeId']
        numeric_cols = df_encoded.select_dtypes(include=['int64', 'float64', 'int32']).columns
        cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]
        
        if cols_to_scale:
            scaler = StandardScaler()
            df_encoded[cols_to_scale] = scaler.fit_transform(df_encoded[cols_to_scale])
            
        logger.info(f"Encoding and scaling complete. Scaled {len(cols_to_scale)} numerical features. Final shape: {df_encoded.shape}")
        return df_encoded

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    cleaned_df = preprocessor.clean_data(df)
    
    # Quick sanity check
    assert cleaned_df.isnull().sum().sum() == 0, "There are still missing values!"
    print(f"Sanity check passed. Dataset is perfectly clean with {len(cleaned_df)} rows.")
    
    # Feature Engineering
    engineered_df = preprocessor.engineer_features(cleaned_df)
    
    # Encoding and Scaling (Step 2.4)
    modeling_ready_df = preprocessor.encode_and_scale(engineered_df)
    print(f"Modeling Ready dataset head:\n{modeling_ready_df[['UserId', 'UserAvgRating', 'AttractionId', 'AttractionAvgRating', 'VisitSeason']].head()}")
    
    # Save Final Feature Store (Step 2.5)
    out_path = preprocessor.data_path.parent / "modeling_ready_data.csv"
    logger.info(f"Saving final modeling ready dataset to {out_path}")
    modeling_ready_df.to_csv(out_path, index=False)
    logger.info("Saved successfully.")
