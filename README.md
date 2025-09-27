# California House Price Predictor: End-to-End ML Deployment (Internship Project)

##  Project Overview
This repository presents a **complete End-to-End Machine Learning project** to predict median house values in California. Developed during my **internship at Arch Technology**, this project demonstrates model training, evaluation, and deployment as a **production-ready web application**.  

Key highlights include a **high-performance XGBoost Regressor**, advanced feature engineering, and interactive visualizations.

---

##  Features & Highlights
- **Interactive Web App:** Built with **Streamlit** for real-time predictions.  
- **Robust ML Pipeline:** Preprocessing handles scaling, one-hot encoding, and imputation automatically.  
- **Advanced Features:** Includes `rooms_per_household`, `bedrooms_per_room`, and `population_per_household`.  
- **Interactive Visualizations:**  
  - **PyDeck** for geographic visualization  
  - **Plotly** for exploring feature distributions  
- **High Model Performance:** **R² Score of 0.8462**  

---

## 🛠️ Technology Stack

| Category | Tools/Libraries | Purpose |
|----------|----------------|---------|
| ML Framework | XGBoost, Scikit-learn | Model training & preprocessing |
| Deployment | Streamlit | Web application interface |
| Data Handling | Pandas, NumPy | Data cleaning & manipulation |
| Serialization | Dill, Joblib | Save/load trained models |
| Visualization | Plotly, PyDeck | Interactive charts & mapping |

---
##📁 Repository Structure
| File/Folder                        | Description                                                                               |
| ---------------------------------- | ----------------------------------------------------------------------------------------- |
| `app.py`                           | Main Streamlit app with input forms, prediction logic, and visualization.                 |
| `model_pipeline.ipynb`             | Notebook detailing data preprocessing, pipeline creation, model training, and evaluation. |
| `plots.ipynb`                      | Exploratory Data Analysis (EDA) using Matplotlib, Seaborn, and Plotly.                    |
| `california_house_model_dill.pkl`  | Serialized ML pipeline (transformers + XGBoost model).                                    |
| `California_House_Prices_Data.csv` | Raw dataset used for model training.                                                      |
| `requirements.txt`                 | List of all Python packages needed to run the project.                                    |



## 📈 Model Performance

| Metric | Value |
|--------|-------|
| **R² Score** | 0.8462 |
| **MAE** | $29,108 |
| **RMSE** | $45,699 |

##📌 Author
#Daniyal Rajput
Internship Project – Arch Technology

"Every time I solved one error, another appeared — but I kept going. Persistence paid off!" 💪
---

## 🚀 Getting Started

```bash
git clone https://github.com/daanni69/house-price-predictor-model/tree/main
cd california-house-price-prediction
pip install -r requirements.txt
streamlit run app.py
