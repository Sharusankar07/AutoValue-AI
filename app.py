
import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="AutoValue AI | Car Valuation",
    page_icon="🚗",
    layout="wide"
)

# -------------------------------
# LOAD TRAINED MODEL
# -------------------------------

MODEL_PATH = "model/car_price_model.pkl"
INFO_PATH = "model/model_info.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(INFO_PATH):
    st.error("Trained model not found. Please run train_model.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)
model_info = joblib.load(INFO_PATH)

# -------------------------------
# CUSTOM CSS
# -------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #0B1120;
    color: #F8FAFC;
}

[data-testid="stSidebar"] {
    background-color: #111C30;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #38BDF8;
    text-align: center;
}

.subtitle {
    color: #94A3B8;
    text-align: center;
    font-size: 17px;
}

.result-card {
    background: linear-gradient(135deg, #132B48, #164E75);
    padding: 30px;
    border-radius: 18px;
    border: 1px solid #38BDF8;
    text-align: center;
}

.price {
    font-size: 42px;
    font-weight: bold;
    color: #38BDF8;
}

.small-text {
    color: #CBD5E1;
    font-size: 14px;
}

div.stButton > button {
    background-color: #0284C7;
    color: white;
    border-radius: 10px;
    border: none;
    height: 48px;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #0369A1;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------

st.markdown(
    '<div class="main-title">🚗 AutoValue AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Smart AI-Powered Used Car Price Prediction</div>',
    unsafe_allow_html=True
)

st.write("")

st.markdown("---")

# -------------------------------
# INTRODUCTION
# -------------------------------

st.markdown("### Welcome to AutoValue AI")
st.write(
    "Discover the estimated resale value of your car using "
    "Machine Learning. Enter your vehicle details below "
    "and let our Random Forest model estimate its market value."
)

st.write("")

# -------------------------------
# CAR INPUT SECTION
# -------------------------------

st.markdown("### 🚘 Enter Your Car Details")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox(
        "Car Brand",
        options=model_info["brands"]
    )

    year = st.selectbox(
        "Manufacturing Year",
        options=list(
            range(
                model_info["max_year"],
                model_info["min_year"] - 1,
                -1
            )
        )
    )

    km_driven = st.number_input(
        "Kilometres Driven (KM)",
        min_value=0,
        max_value=1000000,
        value=30000,
        step=1000
    )

with col2:
    fuel = st.selectbox(
        "Fuel Type",
        options=model_info["fuel_types"]
    )

    transmission = st.selectbox(
        "Transmission",
        options=model_info["transmissions"]
    )

    st.info(
        "💡 Tip: Enter the correct kilometre reading "
        "for a more relevant estimate."
    )

st.write("")

# -------------------------------
# PREDICTION BUTTON
# -------------------------------

if st.button("✨ PREDICT RESALE PRICE"):

    input_data = pd.DataFrame({
        "brand": [brand],
        "year": [year],
        "km_driven": [km_driven],
        "fuel": [fuel],
        "transmission": [transmission]
    })

    prediction = model.predict(input_data)[0]

    # Avoid displaying negative price
    prediction = max(0, prediction)

    # ---------------------------
    # CONDITION INDICATOR
    # ---------------------------

    current_year = datetime.now().year
    car_age = current_year - year

    if car_age <= 5 and km_driven <= 50000:
        condition = "Relatively New"
    elif car_age <= 10 and km_driven <= 100000:
        condition = "Moderate Usage"
    else:
        condition = "Higher Age / Usage"

    # ---------------------------
    # RESULT DISPLAY
    # ---------------------------

    st.markdown("---")

    st.markdown(
        '<div class="result-card">'
        '<h3>Estimated Resale Value</h3>'
        f'<div class="price">₹ {prediction:,.0f}</div>'
        '<p class="small-text">'
        'AI-generated price estimate using Random Forest Regression'
        '</p></div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown("### 📋 Your Car Summary")

    r1, r2, r3 = st.columns(3)

    r1.metric("Brand", brand)
    r2.metric("Year", year)
    r3.metric("Kilometres", f"{km_driven:,} KM")

    r4, r5, r6 = st.columns(3)

    r4.metric("Fuel Type", fuel)
    r5.metric("Transmission", transmission)
    r6.metric("Usage Category", condition)

    st.success(
        "Your car valuation has been generated successfully!"
    )

    st.caption(
        "Note: This is an estimated value based on historical "
        "dataset patterns. Actual resale prices may vary "
        "depending on vehicle condition, location, demand "
        "and other factors."
    )

# -------------------------------
# MODEL INFORMATION
# -------------------------------

st.markdown("---")

st.markdown("### 🤖 About Our AI Model")

m1, m2, m3 = st.columns(3)

m1.metric(
    "Algorithm",
    "Random Forest"
)

m2.metric(
    "R² Score",
    f"{model_info['r2'] * 100:.2f}%"
)

m3.metric(
    "Mean Absolute Error",
    f"₹ {model_info['mae']:,.0f}"
)

st.markdown("---")

# -------------------------------
# FOOTER
# -------------------------------

st.markdown(
    "<div style='text-align:center;color:#94A3B8;'>"
    "🚗 AutoValue AI | Smart Car Valuation Platform<br>"
    "Powered by Machine Learning & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
