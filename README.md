# Tourism Experience Analytics: Classification, Prediction & Recommendation System

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B?logo=streamlit&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/Neon-PostgreSQL-336791?logo=postgresql&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Scikit_Learn-F7931E?logo=scikit-learn&logoColor=white)

## 📌 Project Overview
This project is an end-to-end, full-stack Data Science and Machine Learning web application designed for the tourism industry. It analyzes tourist behaviors to predict their preferred travel style, predicts the exact ratings they would give to specific attractions, and provides personalized travel itineraries to solve the "Cold Start" recommendation problem.

## ✨ Key Features
- **Machine Learning Classification:** Predicts a tourist's "Visit Mode" (e.g., Family, Business, Solo) based on their demographics using a highly-tuned Random Forest Classifier (achieving 34% accuracy on noisy 5-way human behavioral data).
- **Hybrid Recommendation Engine:** A dynamic system that mathematically ranks attractions. For new users, it uses Predictive ML to guess their preferences. For returning users, it uses Collaborative Filtering (Cosine Similarity) based on past interactions.
- **Custom SQL Authentication:** A complete, low-level PostgreSQL driver built using `psycopg2`. Features fully automated table generation, secure user sign-ups, and military-grade `bcrypt` password hashing.
- **B2B Analytics Dashboard:** An interactive admin portal that visualizes exploratory data analysis (EDA) insights and macro-level ML demographic predictions for business stakeholders.
- **B2C Personal Travel Guide:** A beautifully designed frontend where users can interact with the ML models to receive real-time, personalized attraction recommendations.

## 🛠️ Technology Stack
*   **Frontend / Web Framework:** Streamlit
*   **Database:** Neon PostgreSQL (Cloud)
*   **Machine Learning:** Scikit-Learn, Pandas, NumPy
*   **Security:** Bcrypt, Python-Dotenv
*   **Database Driver:** Psycopg2-Binary

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Tourism-Experience-Analytics.git
cd Tourism-Experience-Analytics
```

### 2. Set Up the Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure the Environment
Create a `.env` file in the root directory and add your Neon PostgreSQL Direct Connection string (ensure you are using the IPv4 Session Pooler if your local network does not support IPv6):
```env
DATABASE_URL="postgresql://user:password@aws-0-ap-south-1.pooler.supabase.com:6543/postgres"
```

### 5. Run the Application
```bash
streamlit run src/app.py
```

## 📁 Repository Structure
```text
├── assets/                  # Images and EDA plots used in documentation
├── models/                  # Serialized .pkl Machine Learning models
├── processed_data/          # Cleaned feature stores and numerical matrices
├── src/                     # Core application logic
│   ├── views/               # Streamlit frontend UI portals
│   ├── app.py               # Main Streamlit application router
│   ├── data_ingestion.py    # Raw data processing and merging logic
│   ├── db.py                # PostgreSQL driver and bcrypt authentication
│   ├── preprocessing.py     # Feature engineering and scaling
│   └── recommenders.py      # Hybrid ML prediction and similarity engine
├── config.yaml              # Global path and parameter configuration
├── project_documentation.md # Detailed academic log of methodology and results
└── requirements.txt         # Project dependencies
```

## 🌐 Live Deployment
This project is fully optimized for **Streamlit Community Cloud**. It utilizes `@st.cache_resource` for connection pooling and is capable of automatically refreshing stale database connections to accommodate cloud constraints.
