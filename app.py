
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Prediction System")
st.write(
    "A Machine Learning project using Logistic Regression."
)

@st.cache_resource
def train_model():
    data = pd.read_csv("diabetes.csv")

    features = [
        "Pregnancies", "Glucose", "BloodPressure",
        "SkinThickness", "Insulin", "BMI",
        "DiabetesPedigreeFunction", "Age"
    ]

    X = data[features]
    y = data["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y
    )

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, accuracy

try:
    model, accuracy = train_model()

    st.metric("Model Test Accuracy", f"{accuracy:.1%}")
    st.subheader("Enter Patient Details")

    with st.form("prediction_form"):
        pregnancies = st.number_input(
            "Pregnancies", 0, 20, 0
        )
        glucose = st.number_input(
            "Glucose Level", 0, 300, 120
        )
        blood_pressure = st.number_input(
            "Blood Pressure", 0, 200, 70
        )
        skin_thickness = st.number_input(
            "Skin Thickness", 0, 100, 20
        )
        insulin = st.number_input(
            "Insulin Level", 0, 900, 79
        )
        bmi = st.number_input(
            "BMI", 0.0, 80.0, 25.0
        )
        pedigree = st.number_input(
            "Diabetes Pedigree Function",
            0.0, 3.0, 0.5
        )
        age = st.number_input(
            "Age", 1, 120, 25
        )

        submitted = st.form_submit_button(
            "Predict Diabetes Risk"
        )

    if submitted:
        patient = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": age
        }])

        prediction = model.predict(patient)[0]
        probability = model.predict_proba(patient)[0][1]

        if prediction == 1:
            st.warning(
                f"Model result: Positive label "
                f"(estimated probability: {probability:.1%})"
            )
        else:
            st.success(
                f"Model result: Negative label "
                f"(estimated probability: {probability:.1%})"
            )

        st.caption(
            "Educational demonstration only. "
            "This result is not a medical diagnosis."
        )

except FileNotFoundError:
    st.error(
        "diabetes.csv not found. Keep it in the same "
        "folder as app.py."
    )
except Exception as error:
    st.error(f"Could not run the application: {error}")
