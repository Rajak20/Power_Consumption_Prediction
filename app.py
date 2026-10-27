import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from datetime import datetime

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Power Consumption Prediction",
    page_icon="⚡",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "improved_model_zone1.pkl"

FEATURES = [
    "Temperature",
    "Humidity",
    "WindSpeed",
    "GeneralDiffuseFlows",
    "DiffuseFlows",
    "Hour",
    "Month",
    "DayOfWeek",
    "IsWeekend"
]


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚡ Power Predictor")
    st.write("Machine Learning-Based Power Consumption Prediction")
    st.divider()

    st.subheader("About")
    st.write(
        "Enter weather conditions and time information "
        "to estimate power consumption using your trained model."
    )

    st.info(
        "DayOfWeek uses 0 for Monday and 6 for Sunday. "
        "IsWeekend uses 1 for weekends and 0 for weekdays."
    )


# ---------------- MAIN TITLE ----------------
st.title("⚡ Power Consumption Prediction")
st.write(
    "Enter the nine input features below and click the button "
    "to predict power consumption."
)

st.divider()


# ---------------- INPUT FORM ----------------
st.subheader("🌤️ Environmental Conditions")

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-20.0,
            max_value=60.0,
            value=25.0,
            step=0.5
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=1.0
        )

        wind_speed = st.number_input(
            "Wind Speed (m/s)",
            min_value=0.0,
            max_value=100.0,
            value=1.0,
            step=0.5
        )

    with col2:
        general_diffuse = st.number_input(
            "General Diffuse Flows",
            min_value=0.0,
            value=200.0,
            step=10.0
        )

        diffuse = st.number_input(
            "Diffuse Flows",
            min_value=0.0,
            value=100.0,
            step=10.0
        )

        hour = st.selectbox(
            "Hour of Day",
            options=list(range(24)),
            index=datetime.now().hour
        )

    with col3:
        month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            index=datetime.now().month - 1
        )

        day_of_week = st.selectbox(
            "Day of Week",
            options=list(range(7)),
            index=datetime.now().weekday(),
            format_func=lambda x: [
                "Monday (0)",
                "Tuesday (1)",
                "Wednesday (2)",
                "Thursday (3)",
                "Friday (4)",
                "Saturday (5)",
                "Sunday (6)"
            ][x]
        )

        is_weekend = st.selectbox(
            "Is Weekend?",
            options=[0, 1],
            format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
            index=1 if day_of_week >= 5 else 0
        )

    submitted = st.form_submit_button(
        "⚡ Predict Power Consumption",
        type="primary",
        use_container_width=True
    )


# ---------------- PREDICTION ----------------
if submitted:

    input_data = pd.DataFrame(
        [[
            temperature,
            humidity,
            wind_speed,
            general_diffuse,
            diffuse,
            hour,
            month,
            day_of_week,
            is_weekend
        ]],
        columns=FEATURES
    )

    try:
        model = load_model()

        expected_features = getattr(
            model, "feature_names_in_", None
        )

        if expected_features is not None:
            if set(expected_features) != set(FEATURES):
                st.error(
                    "The model's expected feature names do not match "
                    "the nine inputs in this application."
                )
                st.stop()

            input_data = input_data[list(expected_features)]

        prediction = model.predict(input_data)
        result = float(np.asarray(prediction).reshape(-1)[0])

        st.divider()
        st.subheader("📊 Prediction Result")

        if np.isfinite(result):
            st.success("Prediction completed successfully!")

            st.metric(
                label="Predicted Power Consumption",
                value=f"{result:,.2f}"
            )

            st.caption(
                "The output unit depends on the target variable "
                "used when training the model."
            )

            with st.expander("View Input Details"):
                st.dataframe(
                    input_data,
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.error("The model returned an invalid prediction.")

    except FileNotFoundError:
        st.error(
            "Model file not found. Place improved_model_zone1.pkl "
            "in the same folder as app.py."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

        st.info(
            "Check the model's expected input features and "
            "the installed Python library versions."
        )


# ---------------- FOOTER ----------------
st.divider()

st.caption(
    "Power Consumption Prediction | Machine Learning | Streamlit"
)
