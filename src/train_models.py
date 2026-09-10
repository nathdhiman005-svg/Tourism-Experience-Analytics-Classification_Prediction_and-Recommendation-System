import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import joblib
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, data_path: str = "../processed_data/modeling_ready_data.csv"):
        self.data_path = (Path(__file__).parent / data_path).resolve()
        # Save models directory
        self.models_dir = (Path(__file__).parent / "../models").resolve()
        self.models_dir.mkdir(exist_ok=True)
        
    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading modeling ready dataset from {self.data_path}")
        return pd.read_csv(self.data_path)
        
    def prepare_data_splits(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """
        Splits the dataset into training and testing sets for both Classification and Regression tasks.
        Uses explicit Feature Selection to prevent data leakage and noise memorization.
        """
        logger.info("Preparing data splits...")
        
        # 1. Feature Selection for Classification (Dropping Noise)
        cls_features = [
            'UserContinent', 'UserRegion', 'UserCountry', 
            'AttractionType', 'VisitSeason', 'UserAvgRating', 
            'UserTotalVisits', 'AttractionAvgRating', 'AttractionPopularity'
        ]
        
        X_cls = df[cls_features]
        y_class = df['VisitModeId'] if 'VisitModeId' in df.columns else df['VisitMode_id']
        
        # Split for Classification
        X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
            X_cls, y_class, test_size=test_size, random_state=random_state, stratify=y_class
        )
        
        # 2. Feature Selection for Regression (Dropping Noise, Adding VisitMode)
        reg_features = [
            'UserContinent', 'UserRegion', 'UserCountry', 
            'AttractionType', 'VisitSeason', 'UserAvgRating', 
            'UserTotalVisits', 'AttractionAvgRating', 'AttractionPopularity'
        ]
        # Add VisitMode back in as a predictive behavioral feature for rating
        if 'VisitModeId' in df.columns:
            reg_features.append('VisitModeId')
        elif 'VisitMode_id' in df.columns:
            reg_features.append('VisitMode_id')
            
        X_reg = df[reg_features]
        y_reg = df['Rating']
        
        # Split for Regression
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=test_size, random_state=random_state
        )
        
        return {
            'classification': (X_train_cls, X_test_cls, y_train_cls, y_test_cls),
            'regression': (X_train_reg, X_test_reg, y_train_reg, y_test_reg)
        }

    def train_classification_model(self, X_train, X_test, y_train, y_test):
        logger.info("Starting Hyperparameter Tuning for Classification Model (Random Forest)...")
        
        rf = RandomForestClassifier(class_weight='balanced', random_state=42)
        
        # AGGRESSIVE REGULARIZATION: Restricting max_depth to very small numbers to prevent memorization
        param_dist = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'min_samples_split': [10, 50, 100]
        }
        
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=10,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        best_model = random_search.best_estimator_
        logger.info(f"Best Hyperparameters Found (Classification): {random_search.best_params_}")
        
        # Evaluate on Training Set 
        train_preds = best_model.predict(X_train)
        train_f1 = f1_score(y_train, train_preds, average='weighted')
        
        # Evaluate on Testing Set
        logger.info("Evaluating Best Classification Model on Testing Set...")
        test_preds = best_model.predict(X_test)
        
        test_acc = accuracy_score(y_test, test_preds)
        test_prec = precision_score(y_test, test_preds, average='weighted')
        test_rec = recall_score(y_test, test_preds, average='weighted')
        test_f1 = f1_score(y_test, test_preds, average='weighted')
        
        print("\n" + "="*50)
        print("TUNED CLASSIFICATION MODEL RESULTS")
        print("="*50)
        print(f"Training F1-Score: {train_f1:.4f}")
        print(f"Testing  F1-Score: {test_f1:.4f}")
        print("-" * 50)
        print(f"Test Accuracy:  {test_acc:.4f}")
        print(f"Test Precision: {test_prec:.4f}")
        print(f"Test Recall:    {test_rec:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, test_preds))
        
        return best_model
        
    def train_regression_model(self, X_train, X_test, y_train, y_test):
        logger.info("Starting Hyperparameter Tuning for Regression Model (Random Forest)...")
        
        rf_reg = RandomForestRegressor(random_state=42)
        
        # AGGRESSIVE REGULARIZATION: Restricting max_depth to prevent memorization
        param_dist = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'min_samples_split': [10, 50, 100]
        }
        
        random_search = RandomizedSearchCV(
            estimator=rf_reg,
            param_distributions=param_dist,
            n_iter=10,
            cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        best_model = random_search.best_estimator_
        logger.info(f"Best Hyperparameters Found (Regression): {random_search.best_params_}")
        
        # Evaluate on Training Set
        train_preds = best_model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_preds)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        train_r2 = r2_score(y_train, train_preds)
        
        # Evaluate on Testing Set
        test_preds = best_model.predict(X_test)
        test_mae = mean_absolute_error(y_test, test_preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_r2 = r2_score(y_test, test_preds)
        
        print("\n" + "="*50)
        print("TUNED REGRESSION MODEL RESULTS (Clean Features)")
        print("="*50)
        print(f"Training MAE:  {train_mae:.4f} stars (RMSE: {train_rmse:.4f}, R2: {train_r2:.4f})")
        print(f"Testing  MAE:  {test_mae:.4f} stars (RMSE: {test_rmse:.4f}, R2: {test_r2:.4f})")
        print("="*50)
        
        return best_model
        
    def save_model(self, model, filename: str):
        filepath = self.models_dir / filename
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    df = trainer.load_data()
    splits = trainer.prepare_data_splits(df)
    
    # Train and tune Classification Model
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = splits['classification']
    best_cls_model = trainer.train_classification_model(X_train_cls, X_test_cls, y_train_cls, y_test_cls)
    
    # Train and tune Regression Model
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = splits['regression']
    best_reg_model = trainer.train_regression_model(X_train_reg, X_test_reg, y_train_reg, y_test_reg)
    
    # Step 4.4: Model Serialization
    logger.info("Saving tuned models to disk...")
    trainer.save_model(best_cls_model, "visit_mode_classifier.pkl")
    trainer.save_model(best_reg_model, "rating_regressor.pkl")
    logger.info("Predictive Modeling Phase Complete!")
