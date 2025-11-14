# EstateLens — House Price Prediction & Analytics


**Tagline:** Local property valuation, visual maps, and similar-property recommendations.


## Overview


EstateLens is a Streamlit-based application that helps users estimate house prices and explore property listings visually. It provides:


- Price prediction based on an ML pipeline
- Similar-property recommendations using cosine similarity
- Interactive Folium map with clustered property markers
- Filters and quick statistics for exploratory analysis


## Key Features


- Predict property prices in PKR (displays min/avg/max estimates)
- Convert sizes from sqft to Marla/Kanal for local readability
- Recommend similar societies/properties using a precomputed cosine similarity matrix
- Interactive map with clustered markers and popups
- Sidebar navigation and model metadata display


## Requirements


- `df.pkl` — DataFrame containing properties and the features used for prediction
- `pipeline.pkl` — Trained ML pipeline to predict log-transformed price (model expects the same feature schema)
- `cosine_sim.pkl` — Cosine similarity matrix for recommendations (optional but required for recommendations)
- Optional: `model_info.pkl`, `feature_scaler.pkl`, `tfidf_vectorizer.pkl` depending on how recommendations were generated
- `merged_output_containment.csv` — CSV used for map plotting (latitude/longitude required)


## Installation


1. Clone the repository:


```bash
git clone <your-repo-url>
cd <your-repo-folder>
