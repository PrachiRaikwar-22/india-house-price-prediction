import os
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="India House Price Predictor",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# THEME SELECTOR - TOP RIGHT
# ============================================================

left, right = st.columns([8, 2])

with right:

    theme = st.selectbox(
        "Theme",
        [
            "🖥️ Default",
            "☀️ Light",
            "🌙 Dark"
        ],
        label_visibility="collapsed"
    )


# ============================================================
# THEME CSS
# ============================================================

if theme == "☀️ Light":

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #ffffff;
            color: #111111;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


elif theme == "🌙 Dark":

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }

        [data-testid="stHeader"] {
            background-color: #0e1117;
        }

        [data-testid="stToolbar"] {
            background-color: #0e1117;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MODEL FILE
# ============================================================

MODEL_FILE = "India_house_price_model.joblib"


# ============================================================
# CHECK MODEL FILE
# ============================================================

if not os.path.exists(MODEL_FILE):

    st.error(
        f"""
        ❌ Model file not found.

        Expected file:

        `{MODEL_FILE}`

        Please run `model.py` first.
        """
    )

    st.stop()


# ============================================================
# LOAD MODEL PACKAGE
# ============================================================

@st.cache_resource
def load_model_package():

    return joblib.load(
        MODEL_FILE
    )


package = load_model_package()


# ============================================================
# VERIFY THAT THIS IS THE NEW MODEL PACKAGE
# ============================================================

if not isinstance(package, dict):

    st.error(
        """
        ❌ Wrong model file detected.

        The loaded `.joblib` file contains only an
        XGBRegressor instead of the complete model package.

        Please run `model.py` again to create:

        `India_house_price_model.joblib`
        """
    )

    st.stop()


# ============================================================
# VERIFY REQUIRED KEYS
# ============================================================

required_keys = [
    "model",
    "scaler",
    "feature_columns",
    "cities",
    "locality_tiers",
    "furnishing_options"
]

missing_keys = [
    key
    for key in required_keys
    if key not in package
]

if missing_keys:

    st.error(
        "❌ Model package is incomplete."
    )

    st.write(
        "Missing:",
        missing_keys
    )

    st.stop()


# ============================================================
# EXTRACT MODEL COMPONENTS
# ============================================================

model = package["model"]

scaler = package["scaler"]

feature_columns = package[
    "feature_columns"
]

cities = package["cities"]

locality_tiers = package[
    "locality_tiers"
]

furnishing_options = package[
    "furnishing_options"
]


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    return pd.read_csv(
        "house_price_dataset_india_12k.csv",
        sep=None,
        engine="python"
    )


df = load_dataset()


# ============================================================
# TITLE
# ============================================================

st.title(
    "🏠 India House Price Predictor"
)

st.write(
    "Predict the estimated market price of a property "
    "using location, size, amenities and other property factors."
)

st.divider()


# ============================================================
# LOCATION
# ============================================================

st.header("📍 Location")

col1, col2 = st.columns(2)

with col1:

    city = st.selectbox(
        "City",
        cities
    )

with col2:

    locality_tier = st.selectbox(
        "Locality Tier",
        locality_tiers
    )


# ============================================================
# PROPERTY DETAILS
# ============================================================

st.header("🏡 Property Details")

col1, col2, col3, col4 = st.columns(4)

with col1:

    bhk = st.number_input(
        "BHK",
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )

with col2:

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )

with col3:

    super_area = st.number_input(
        "Super Area (sq ft)",
        min_value=100.0,
        max_value=100000.0,
        value=1000.0,
        step=50.0
    )

with col4:

    carpet_area = st.number_input(
        "Carpet Area (sq ft)",
        min_value=50.0,
        max_value=100000.0,
        value=800.0,
        step=50.0
    )


# ============================================================
# FLOOR INFORMATION
# ============================================================

st.header("🏢 Floor Information")

col1, col2, col3 = st.columns(3)

with col1:

    floor_no = st.number_input(
        "Floor Number",
        min_value=0,
        max_value=200,
        value=2,
        step=1
    )

with col2:

    total_floors = st.number_input(
        "Total Floors",
        min_value=1,
        max_value=200,
        value=5,
        step=1
    )

with col3:

    property_age = st.number_input(
        "Property Age (years)",
        min_value=0,
        max_value=200,
        value=5,
        step=1
    )


# ============================================================
# AMENITIES
# ============================================================

st.header("🅿️ Amenities")

col1, col2, col3, col4 = st.columns(4)

with col1:

    parking = st.number_input(
        "Parking",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

with col2:

    furnishing = st.selectbox(
        "Furnishing",
        furnishing_options
    )

with col3:

    lift = st.selectbox(
        "Lift",
        [0, 1],
        format_func=lambda x:
        "Yes" if x == 1 else "No"
    )

with col4:

    gated_society = st.selectbox(
        "Gated Society",
        [0, 1],
        format_func=lambda x:
        "Yes" if x == 1 else "No"
    )


# ============================================================
# LOCATION FACTORS
# ============================================================

st.header("📏 Location Factors")

col1, col2, col3 = st.columns(3)

with col1:

    distance_metro = st.number_input(
        "Distance to Metro (km)",
        min_value=0.0,
        max_value=100.0,
        value=2.0,
        step=0.1
    )

with col2:

    distance_city_center = st.number_input(
        "Distance to City Center (km)",
        min_value=0.0,
        max_value=200.0,
        value=5.0,
        step=0.1
    )

with col3:

    nearby_school = st.number_input(
        "Nearby School (km)",
        min_value=0.0,
        max_value=100.0,
        value=1.0,
        step=0.1
    )


col1, col2 = st.columns(2)

with col1:

    nearby_hospital = st.number_input(
        "Nearby Hospital (km)",
        min_value=0.0,
        max_value=100.0,
        value=2.0,
        step=0.1
    )

with col2:

    crime_rate = st.number_input(
        "Crime Rate Index",
        min_value=0.0,
        max_value=1000.0,
        value=50.0,
        step=1.0
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict House Price",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ====================================================
        # CREATE RAW INPUT DATAFRAME
        # ====================================================

        raw_input = pd.DataFrame({

            "Locality_Tier": [
                locality_tier
            ],

            "City": [
                city
            ],

            "BHK": [
                bhk
            ],

            "Bathrooms": [
                bathrooms
            ],

            "Super_Area_sqft": [
                super_area
            ],

            "Carpet_Area_sqft": [
                carpet_area
            ],

            "Floor_No": [
                floor_no
            ],

            "Total_Floors": [
                total_floors
            ],

            "Property_Age_years": [
                property_age
            ],

            "Parking": [
                parking
            ],

            "Furnishing": [
                furnishing
            ],

            "Lift": [
                lift
            ],

            "Gated_Society": [
                gated_society
            ],

            "Distance_to_Metro_km": [
                distance_metro
            ],

            "Distance_to_CityCenter_km": [
                distance_city_center
            ],

            "Nearby_School_km": [
                nearby_school
            ],

            "Nearby_Hospital_km": [
                nearby_hospital
            ],

            "Crime_Rate_Index": [
                crime_rate
            ]
        })


        # ====================================================
        # ONE-HOT ENCODING
        # ====================================================

        input_df = pd.get_dummies(
            raw_input,
            columns=[
                "Locality_Tier",
                "City",
                "Furnishing"
            ],
            dtype=int
        )


        # ====================================================
        # REMOVE AREA COLUMNS TEMPORARILY
        # ====================================================

        area_columns = [
            "Super_Area_sqft",
            "Carpet_Area_sqft"
        ]

        area_df = input_df[
            area_columns
        ].copy()

        input_df = input_df.drop(
            columns=area_columns
        )


        # ====================================================
        # MATCH ALL NON-AREA MODEL FEATURES
        # ====================================================

        non_area_features = [
            column
            for column in feature_columns
            if column not in area_columns
        ]

        input_df = input_df.reindex(
            columns=non_area_features,
            fill_value=0
        )


        # ====================================================
        # SCALE AREA
        # ====================================================

        scaled_area = scaler.transform(
            area_df
        )


        # ====================================================
        # ADD SCALED AREA
        # ====================================================

        input_df["Super_Area_sqft"] = float(
            scaled_area[0][0]
        )

        input_df["Carpet_Area_sqft"] = float(
            scaled_area[0][1]
        )


        # ====================================================
        # EXACT SAME FEATURE ORDER AS TRAINING
        # ====================================================

        input_df = input_df[
            feature_columns
        ]


        # ====================================================
        # FORCE FLOAT
        # ====================================================

        input_df = input_df.astype(
            float
        )


        # ====================================================
        # FINAL CHECK
        # ====================================================

        if list(input_df.columns) != list(
            feature_columns
        ):

            st.error(
                "❌ Feature order mismatch."
            )

            st.write(
                "Expected:"
            )

            st.write(
                feature_columns
            )

            st.write(
                "Received:"
            )

            st.write(
                input_df.columns.tolist()
            )

            st.stop()


        # ====================================================
        # PREDICT
        # ====================================================

        prediction = model.predict(
            input_df
        )


        predicted_price = float(
            prediction[0]
        )


        # ====================================================
        # PREVENT NEGATIVE PRICE
        # ====================================================

        predicted_price = max(
            0,
            predicted_price
        )


        # ====================================================
        # CONVERT SQ FT TO SQUARE METRE
        # ====================================================

        area_m2 = (
            super_area / 10.7639
        )


        # ====================================================
        # CALCULATE ₹ PER SQUARE METRE
        # ====================================================

        price_per_m2 = (
            predicted_price / area_m2
        )


        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        st.success(
            "✅ Prediction completed successfully!"
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "🏠 Predicted Market Price",
                f"₹{predicted_price:,.0f}"
            )

        with col2:

            st.metric(
                "📐 Price per Square Metre",
                f"₹{price_per_m2:,.0f}"
            )


        # ====================================================
        # PROPERTY SUMMARY
        # ====================================================

        st.subheader(
            "📋 Property Summary"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.write(
                f"**City:** {city}"
            )

            st.write(
                f"**Locality Tier:** "
                f"{locality_tier}"
            )

            st.write(
                f"**BHK:** {bhk}"
            )

            st.write(
                f"**Bathrooms:** {bathrooms}"
            )

        with col2:

            st.write(
                f"**Super Area:** "
                f"{super_area:,.0f} sq ft"
            )

            st.write(
                f"**Carpet Area:** "
                f"{carpet_area:,.0f} sq ft"
            )

            st.write(
                f"**Area:** "
                f"{area_m2:,.2f} m²"
            )

            st.write(
                f"**Floor:** "
                f"{floor_no}/{total_floors}"
            )

        with col3:

            st.write(
                f"**Furnishing:** "
                f"{furnishing}"
            )

            st.write(
                f"**Parking:** "
                f"{parking}"
            )

            st.write(
                f"**Property Age:** "
                f"{property_age} years"
            )

            st.write(
                f"**Price/m²:** "
                f"₹{price_per_m2:,.0f}"
            )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(e)