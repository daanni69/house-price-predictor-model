# California House Price Prediction App
# Author: Daniya Rajput
# Built with: Streamlit + XGBoost + Plotly + PyDeck
# Description:
#   This app predicts house prices in California using a trained XGBoost model.
#   It allows users to input house details, apply feature engineering, and
#   visualize the prediction results with charts and maps.


# Import Required Libraries
import streamlit as st  # For building the web app
import numpy as np
import pandas as pd
import dill  as dill # Used for loading the trained ML model
import plotly.express as px  # For interactive bar plots
import pydeck as pdk  # For map visualization

# Feature Engineering Function
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add new features based on raw input values.
    These features improve the model's predictive performance.
    
    Args:
        df (pd.DataFrame): Input dataframe with house details.
    
    Returns:
        pd.DataFrame: Dataframe with additional engineered features.
    """
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']
    df['income_per_person'] = df['median_income'] / df['population']
    return df

# Load the Trained Model
# The trained XGBoost model has been serialized using `dill`.
# We load it here so the app can make predictions instantly.
with open("california_house_model_dill.pkl", "rb") as f:
    model = dill.load(f)


# Streamlit Page Config
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# App Title and Description
st.title("🏠 California House Price Predictor")
st.markdown(
    "### Predict California house prices using a trained **XGBoost Machine Learning model** 🔮"
)

# Sidebar: User Input Section
st.sidebar.header("📌 Enter House Details:")

# Collect input features from the user
longitude = st.sidebar.number_input("Longitude", value=-122.23)
latitude = st.sidebar.number_input("Latitude", value=37.88)
housing_median_age = st.sidebar.number_input("Housing Median Age", value=41)
total_rooms = st.sidebar.number_input("Total Rooms", value=880)
total_bedrooms = st.sidebar.number_input("Total Bedrooms", value=129)
population = st.sidebar.number_input("Population", value=322)
households = st.sidebar.number_input("Households", value=126)
median_income = st.sidebar.number_input("Median Income", value=8.3252)
ocean_proximity = st.sidebar.selectbox(
    "Ocean Proximity",
    ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND']
)

# Create Input DataFrame
# Convert user inputs into a DataFrame for prediction.
input_df = pd.DataFrame([{
    'longitude': longitude,
    'latitude': latitude,
    'housing_median_age': housing_median_age,
    'total_rooms': total_rooms,
    'total_bedrooms': total_bedrooms,
    'population': population,
    'households': households,
    'median_income': median_income,
    'ocean_proximity': ocean_proximity
}])

# Apply feature engineering to enhance predictive power
input_df = add_features(input_df)

# Prediction Section
if st.button("🔮 Predict Price"):
    # Make prediction using the trained model
    prediction = model.predict(input_df)[0]
    st.success(f"🏡 Predicted House Price: **${round(prediction, 2)}**")

    # Display the processed input summary
    with st.expander("📊 View Input Summary"):
        st.write(input_df)

    # Visualization: Bar Plot
    plot_df = pd.DataFrame({
        "Feature": ["Total Rooms", "Total Bedrooms", "Households", "Population"],
        "Value": [total_rooms, total_bedrooms, households, population]
    })

    fig = px.bar(
        plot_df,
        x="Feature",
        y="Value",
        color="Feature",
        title="House Features Comparison",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig, use_container_width=True)

    # Visualization: Map View
    st.subheader("🗺️ House Location on Map")
    map_df = pd.DataFrame([{"lat": latitude, "lon": longitude}])

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/streets-v11",
        initial_view_state=pdk.ViewState(
            latitude=latitude,
            longitude=longitude,
            zoom=10,
            pitch=45,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position='[lon, lat]',
                get_color='[0, 128, 255, 200]',  # Blue dot for house location
                get_radius=800,
            ),
        ],
    ))


# Model Information Section
st.markdown("---")
st.subheader("🤖 Model Information")

with st.expander("🔎 About this Model"):
    st.markdown("""
    - **Algorithm Used**: XGBoost Regressor (XGBRegressor)  
    - **Trained On**: California Housing Dataset  
    - **Features Considered**:  
        - Location: Longitude, Latitude  
        - Housing Median Age  
        - Population & Households  
        - Median Income  
        - Total Rooms & Total Bedrooms  
        - Ocean Proximity  
    - **Feature Engineering Applied**:  
        - Rooms per Household  
        - Bedrooms per Room  
        - Population per Household  
        - Income per Person  
    """)

with st.expander("📈 Model Performance Metrics"):
    # Display performance metrics from evaluation
    col1, col2, col3 = st.columns(3)
    col1.metric("Test R² Score", "0.8462")
    col2.metric("MAE", "29,108.42")
    col3.metric("RMSE", "44,887.87")

    st.markdown("✅ The model explains **~85% variance** in house prices using XGBoost.")


# Footer / Credits
st.markdown("---")
st.caption("⚡ Built with Streamlit | XGBoost ML Project | Created by **Dani**")
