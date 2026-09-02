# Tourism Experience Analytics: Implementation Plan

This document outlines the end-to-end plan for building the Tourism Experience Analytics Classification, Prediction, and Recommendation System. 

> [!TIP]
> **Architecture & Best Practices:** The project will be built using a highly modular and flexible structure. We will rely on configuration files (`config.yaml`), Object-Oriented/Functional design (e.g., `DataProcessor`, `ModelTrainer` classes), and keep our data pipelines modular so that new features or datasets can be added later without rewriting the core logic.

## User Review Required

> [!IMPORTANT]  
> Please review this updated granular plan. We will not proceed until you give the explicit instruction to start. When you are ready, you can say "Start Phase 1, Step 1.1", and we will execute this plan one micro-step at a time.

---

## Proposed Project Structure

```text
d:\Projects\Tourism Experience Analytics Classification_Prediction_and Recommendation System\
│
├── datasets/                   # Raw .xlsx files (already exists)
├── processed_data/             # Cleaned and merged datasets (feature store)
├── notebooks/                  # Jupyter notebooks for Data Understanding and EDA
├── src/                        # Python modular source code
│   ├── config.yaml             # Configurations (paths, model params)
│   ├── data_ingestion.py       # Scripts for loading and merging relational data
│   ├── preprocessing.py        # Cleaning and feature engineering logic
│   ├── train_models.py         # Scripts to train and evaluate ML models
│   └── recommenders.py         # Recommendation system logic
├── models/                     # Saved/pickled ML models
└── app/                        # Streamlit application
    ├── main.py                 # Streamlit UI entry point
    └── components/             # Reusable UI components (forms, charts)
```

---

## Phase 1: Data Consolidation & Pipeline Setup

**Goal:** Create modular scripts to read and merge the 9 relational tables into a unified analytical dataset (Feature Store) for Machine Learning.

*   **Step 1.1: Project Initialization.** Create the defined folder structure (`src/`, `notebooks/`, `processed_data/`, `models/`, `app/`) and an initial `config.yaml` to hold dataset paths.
*   **Step 1.2: Data Loader Module.** Write a robust `data_ingestion.py` script that loads all 9 `.xlsx` files into memory, utilizing error handling and logging.
*   **Step 1.3: Relational Merging Logic.** Implement the join logic (e.g., merging `User` with `City`, `City` with `Country`, `Transaction` with `User` & `Item`) to denormalize the data into a single wide dataset.
*   **Step 1.4: Pipeline Execution.** Run the ingestion pipeline and save the resulting flattened dataset to `processed_data/raw_merged.csv`.

---

## Phase 2: Data Understanding, Cleaning & Feature Engineering

**Goal:** Deeply profile the merged data, handle anomalies, and engineer features suitable for ML algorithms.

*   **Step 2.1: Data Understanding & Profiling.** Create a Jupyter notebook (`notebooks/01_Data_Understanding.ipynb`). Run profiling to understand data types, missing value distributions, summary statistics, and cardinalities of categorical features. *This will inform our cleaning strategy.*
*   **Step 2.2: Data Cleaning Module.** In `src/preprocessing.py`, implement methods to handle the missing values, drop duplicates, and correct string inconsistencies identified in Step 2.1.
*   **Step 2.3: Feature Engineering.** Add logic to extract temporal features (from Year/Month) and aggregate user/item interaction statistics (e.g., user's average rating).
*   **Step 2.4: Encoding & Scaling.** Implement transformers to encode categorical variables (One-Hot/Target Encoding) and scale numerical variables. 
*   **Step 2.5: Finalizing Feature Store.** Execute the preprocessing pipeline on `raw_merged.csv` and export the final `modeling_ready_data.csv`.

---

## Phase 3: Exploratory Data Analysis (EDA)

**Goal:** Uncover business insights, trends, and hotspots through visual analysis.

*   **Step 3.1: Univariate Analysis.** In `notebooks/02_EDA.ipynb`, plot distributions of the target variables (`Rating` and `VisitMode`).
*   **Step 3.2: Bivariate/Multivariate Analysis.** Analyze relationships (e.g., how `VisitMode` changes by `Continent` or which `AttractionType` is most popular in specific `Regions`).
*   **Step 3.3: Business Insight Generation.** Document actionable findings that address the project's use cases (identifying hotspots and customer segments).

---

## Phase 4: Predictive Modeling (Regression & Classification)

**Goal:** Train, evaluate, and save modular machine learning models for predictions.

*   **Step 4.1: Model Setup & Splitting.** In `src/train_models.py`, write logic to split the `modeling_ready_data.csv` into training and testing sets.
*   **Step 4.2: Classification Model (Visit Mode).** Train a classifier (e.g., XGBoost/Random Forest). Perform hyperparameter tuning and evaluate using Accuracy, Precision, Recall, and F1-Score.
*   **Step 4.3: Regression Model (Rating).** Train a regressor to predict Attraction Ratings. Evaluate using RMSE, MAE, and $R^2$.
*   **Step 4.4: Model Serialization.** Save the best models and their preprocessing pipelines as `.pkl` objects in the `models/` directory for production use.

---

## Phase 5: Recommendation System Engine

**Goal:** Build the logic for personalized attraction suggestions.

*   **Step 5.1: Collaborative Filtering Base.** Implement a User-Item interaction matrix in `src/recommenders.py` and build collaborative filtering logic (e.g., using Cosine Similarity or Matrix Factorization).
*   **Step 5.2: Content-Based Filtering.** Build logic to recommend attractions based on item metadata (Type, Location).
*   **Step 5.3: Hybrid Recommender.** Combine both scores into a single robust recommendation function.
*   **Step 5.4: Engine Testing.** Manually test the recommender module with sample User IDs to ensure the suggestions are logical.

---

## Phase 6: Streamlit Application Deployment

**Goal:** Develop an interactive and modular frontend application.

*   **Step 6.1: UI Skeleton & Architecture.** Set up `app/main.py` and `app/components/` with a professional, aesthetically pleasing layout (sidebars, headers, thematic colors).
*   **Step 6.2: Prediction Integration.** Integrate the pickled Classification model so users can input demographics and receive a `Visit Mode` prediction.
*   **Step 6.3: Recommendation Integration.** Hook up the recommendation engine to display personalized attraction cards based on user inputs.
*   **Step 6.4: Interactive Dashboards.** Add a section to display the top insights and dynamic charts from the EDA phase.
*   **Step 6.5: Final QA.** Perform end-to-end testing of the Streamlit application and finalize the documentation.
