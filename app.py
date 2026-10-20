import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1220, #13263b);
    color: #f1f5f9;
}
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}
.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(110deg, #12334a, #164e63);
    border: 1px solid #286078;
    margin-bottom: 24px;
}
.hero h1 {
    color: white;
    margin-bottom: 8px;
}
.hero p {
    color: #cbd5e1;
    font-size: 16px;
}
div[data-testid="stMetric"] {
    background: #17283b;
    border: 1px solid #30465e;
    padding: 18px;
    border-radius: 14px;
}
div[data-testid="stForm"] {
    background: #142337;
    padding: 22px;
    border: 1px solid #30465e;
    border-radius: 16px;
}
.stButton button, .stFormSubmitButton button {
    background: #0e9f8a;
    color: white;
    border-radius: 9px;
    border: none;
    min-height: 45px;
    font-weight: 600;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🩺 Diabetes Prediction System</h1>
    <p>Machine learning dashboard for educational risk estimation.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

@st.cache_resource
def train_model():
    data = load_data()

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

    return model, accuracy, features

try:
    data = load_data()
    model, accuracy, features = train_model()

    st.subheader("📊 Model Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset Records", f"{len(data):,}")
    col2.metric("Input Features", len(features))
    col3.metric("Test Accuracy", f"{accuracy:.1%}")

    st.caption(
        "Accuracy is measured on a held-out test split. "
        "It does not establish clinical reliability."
    )

    st.divider()

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("👤 Patient Information")
        st.write("Enter the values below to view the model output.")

        with st.form("prediction_form"):
            c1, c2 = st.columns(2)

            with c1:
                pregnancies = st.number_input(
                    "Pregnancies", 0, 20, 0
                )
                glucose = st.number_input(
                    "Glucose", 0, 300, 120
                )
                blood_pressure = st.number_input(
                    "Blood Pressure", 0, 200, 70
                )
                skin_thickness = st.number_input(
                    "Skin Thickness", 0, 100, 20
                )

            with c2:
                insulin = st.number_input(
                    "Insulin", 0, 900, 79
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
                "🔍 Estimate Model Output",
                use_container_width=True
            )

    with right:
        st.subheader("📈 Dataset Insights")

        outcome_counts = data["Outcome"].value_counts().reindex(
            [0, 1], fill_value=0
        )

        chart_data = pd.DataFrame({
            "Outcome": ["Negative label", "Positive label"],
            "Records": [
                int(outcome_counts.loc[0]),
                int(outcome_counts.loc[1])
            ]
        }).set_index("Outcome")

        st.bar_chart(chart_data)
        st.write("**Feature summary**")
        st.dataframe(
            data[features].describe().round(2),
            use_container_width=True
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

        st.divider()
        st.subheader("🧾 Model Output")

        result_col, probability_col = st.columns(2)

        with result_col:
            if prediction == 1:
                st.warning("Positive model label")
            else:
                st.success("Negative model label")

        with probability_col:
            st.metric(
                "Estimated model probability",
                f"{probability:.1%}"
            )

        st.progress(float(probability))

        st.caption(
            "Educational demonstration only. This is not a medical "
            "diagnosis or a clinically validated risk assessment. "
            "Consult a qualified healthcare professional for medical advice."
        )

    st.divider()
    st.caption(
        "Diabetes Prediction System | Python • Streamlit • Scikit-learn"
    )

except FileNotFoundError:
    st.error(
        "diabetes.csv not found. Keep it in the same folder as app.py."
    )
except Exception as error:
    st.error(f"Could not run the application: {error}")
