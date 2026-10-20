from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from pathlib import Path

import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #07111f 0%, #10263d 55%, #091522 100%);
        color: #f1f5f9;
    }
    [data-testid="stSidebar"] {
        background: #0b1726;
        border-right: 1px solid #263b52;
    }
    [data-testid="stMetric"] {
        background: rgba(26, 49, 72, 0.78);
        border: 1px solid #35536d;
        padding: 18px;
        border-radius: 14px;
    }
    .hero {
        background: linear-gradient(110deg, #123d59, #176b87);
        padding: 28px;
        border-radius: 18px;
        border: 1px solid #3d829b;
        margin-bottom: 22px;
    }
    .hero h1 {
        color: white;
        margin-bottom: 8px;
    }
    .hero p {
        color: #dcecf5;
        font-size: 16px;
        margin-bottom: 0;
    }
    .section-note {
        color: #a9c1d4;
        font-size: 14px;
    }
    div.stButton > button,
    div.stFormSubmitButton > button {
        background: #1686a8;
        color: white;
        border: none;
        border-radius: 9px;
        font-weight: 600;
        min-height: 44px;
    }
    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background: #20a1c6;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
def create_prediction_pdf(patient, predicted_label, probability):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    pdf.setTitle("Diabetes Prediction Report")

    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(50, height - 55, "Diabetes Prediction Report")

    pdf.setFont("Helvetica", 10)
    pdf.drawString(
        50, height - 80,
        f"Generated: {datetime.now().strftime('%d-%m-%Y %H:%M')}"
    )

    pdf.line(50, height - 95, width - 50, height - 95)

    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, height - 125, "Patient Input Summary")

    y_position = height - 150
    pdf.setFont("Helvetica", 11)

    for column, value in patient.iloc[0].items():
        pdf.drawString(55, y_position, f"{column}: {value}")
        y_position -= 22

    y_position -= 10
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y_position, "Model Output")
    y_position -= 25

    pdf.setFont("Helvetica", 11)
    label = "Positive" if predicted_label == 1 else "Negative"
    pdf.drawString(55, y_position, f"Model label: {label}")
    y_position -= 22
    pdf.drawString(
        55, y_position,
        f"Estimated probability for label 1: {probability:.1%}"
    )

    y_position -= 45
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(50, y_position, "Important Disclaimer")

    y_position -= 22
    pdf.setFont("Helvetica", 9)
    pdf.drawString(
        50, y_position,
        "Educational ML output only; not a medical diagnosis."
    )
    y_position -= 15
    pdf.drawString(
        50, y_position,
        "Consult a qualified healthcare professional for health concerns."
    )

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()

# ---------------- LOAD DATA ----------------
DATA_PATH = Path(__file__).resolve().parent / "diabetes.csv"

REQUIRED_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

@st.cache_data
def load_dataset(file_path):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()

    missing_columns = [
        col for col in REQUIRED_COLUMNS if col not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            "Dataset lo required columns levu: "
            + ", ".join(missing_columns)
        )

    data = data[REQUIRED_COLUMNS].copy()

    for col in REQUIRED_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=REQUIRED_COLUMNS)
    data = data[data["Outcome"].isin([0, 1])]
    data["Outcome"] = data["Outcome"].astype(int)

    if data.empty or data["Outcome"].nunique() < 2:
        raise ValueError(
            "Valid dataset lo Outcome values 0 and 1 rendu undali."
        )

    return data


try:
    df = load_dataset(DATA_PATH)
except FileNotFoundError:
    st.error(
        "diabetes.csv file dorakaledu. app.py pakkana "
        "diabetes.csv file repository lo undho check cheyyi."
    )
    st.stop()
except Exception as error:
    st.error(f"Dataset load cheyyadam lo problem: {error}")
    st.stop()

FEATURES = REQUIRED_COLUMNS[:-1]
X = df[FEATURES]
y = df["Outcome"]

# ---------------- TRAIN MODEL ----------------
@st.cache_resource
def train_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return model, X_test, y_test, predictions


try:
    model, X_test, y_test, y_pred = train_model(X, y)
except Exception as error:
    st.error(f"Model train cheyyadam lo problem: {error}")
    st.stop()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# ---------------- HEADER ----------------
st.markdown(
    """
    <div class="hero">
        <h1>🩺 Diabetes Prediction System</h1>
        <p>Machine Learning Dashboard for Educational Risk Estimation</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Educational demonstration only. This model is not a medical "
    "diagnosis tool and is not clinically validated."
)

# ---------------- NAVIGATION ----------------
with st.sidebar:
    st.title("🩺 Diabetes AI")
    st.caption("Machine Learning Dashboard")
    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard Overview",
            "Patient Prediction",
            "Model Evaluation",
            "Dataset Explorer",
        ],
    )

    st.divider()
    st.caption("Model: Logistic Regression")
    st.caption("Dataset: diabetes.csv")

# =========================================================
# DASHBOARD OVERVIEW
# =========================================================
if page == "Dashboard Overview":
    st.subheader("📊 Model Overview")

    total_records = len(df)
    positive_count = int((df["Outcome"] == 1).sum())
    negative_count = int((df["Outcome"] == 0).sum())
    positive_rate = positive_count / total_records * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Records", f"{total_records:,}")
    c2.metric("Input Features", len(FEATURES))
    c3.metric("Test Accuracy", f"{accuracy:.1%}")
    c4.metric("Outcome 1 in Dataset", f"{positive_rate:.1f}%")

    st.caption(
        "Accuracy is measured on a held-out test split. "
        "It does not establish clinical reliability."
    )

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Outcome Distribution")
        outcome_counts = (
            df["Outcome"]
            .value_counts()
            .reindex([0, 1], fill_value=0)
        )
        outcome_chart = pd.DataFrame(
            {
                "Outcome": ["0 — Negative label", "1 — Positive label"],
                "Records": outcome_counts.values,
            }
        ).set_index("Outcome")
        st.bar_chart(outcome_chart)
        st.caption(
            "These are dataset labels, not diagnoses of the app user."
        )

    with right:
        st.subheader("Average Glucose by Dataset Label")
        glucose_summary = (
            df.groupby("Outcome")["Glucose"]
            .mean()
            .reindex([0, 1])
        )
        glucose_chart = pd.DataFrame(
            {
                "Dataset label": ["0 — Negative", "1 — Positive"],
                "Average glucose": glucose_summary.values,
            }
        ).set_index("Dataset label")
        st.bar_chart(glucose_chart)

    st.divider()
    st.subheader("📈 Feature Snapshot")

    feature_a, feature_b = st.columns(2)

    with feature_a:
        st.markdown("**Average BMI by dataset label**")
        bmi_summary = df.groupby("Outcome")["BMI"].mean().reindex([0, 1])
        bmi_chart = pd.DataFrame(
            {
                "Dataset label": ["0 — Negative", "1 — Positive"],
                "Average BMI": bmi_summary.values,
            }
        ).set_index("Dataset label")
        st.bar_chart(bmi_chart)

    with feature_b:
        st.markdown("**Average age by dataset label**")
        age_summary = df.groupby("Outcome")["Age"].mean().reindex([0, 1])
        age_chart = pd.DataFrame(
            {
                "Dataset label": ["0 — Negative", "1 — Positive"],
                "Average age": age_summary.values,
            }
        ).set_index("Dataset label")
        st.bar_chart(age_chart)

    st.info(
        "Charts summarize patterns in this dataset. They do not show "
        "that a feature causes diabetes or predict an individual's health."
    )

# =========================================================
# PATIENT PREDICTION
# =========================================================
elif page == "Patient Prediction":
    st.subheader("👤 Patient Information")
    st.write("Enter sample values to see the model's output.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input(
                "Pregnancies", min_value=0, max_value=20,
                value=0, step=1
            )
            glucose = st.number_input(
                "Glucose", min_value=0, max_value=300,
                value=120, step=1
            )
            blood_pressure = st.number_input(
                "Blood Pressure", min_value=0, max_value=200,
                value=70, step=1
            )
            skin_thickness = st.number_input(
                "Skin Thickness", min_value=0, max_value=100,
                value=20, step=1
            )

        with col2:
            insulin = st.number_input(
                "Insulin", min_value=0, max_value=900,
                value=79, step=1
            )
            bmi = st.number_input(
                "BMI", min_value=0.0, max_value=80.0,
                value=25.0, step=0.1
            )
            pedigree = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0, max_value=3.0,
                value=0.5, step=0.01, format="%.2f"
            )
            age = st.number_input(
                "Age", min_value=1, max_value=120,
                value=30, step=1
            )

        submitted = st.form_submit_button(
            "Predict Sample", use_container_width=True
        )

    if submitted:
        patient = pd.DataFrame(
            [
                {
                    "Pregnancies": pregnancies,
                    "Glucose": glucose,
                    "BloodPressure": blood_pressure,
                    "SkinThickness": skin_thickness,
                    "Insulin": insulin,
                    "BMI": bmi,
                    "DiabetesPedigreeFunction": pedigree,
                    "Age": age,
                }
            ],
            columns=FEATURES,
        )

        predicted_label = int(model.predict(patient)[0])
        probability = float(model.predict_proba(patient)[0][1])

        st.divider()
        st.subheader("Model Output")

        result_col, probability_col = st.columns(2)

        with result_col:
            if predicted_label == 1:
                st.warning("Model label: Positive")
            else:
                st.success("Model label: Negative")

        with probability_col:
            st.metric(
                "Estimated model probability for label 1",
                f"{probability:.1%}",
            )

        st.progress(probability)

        st.caption(
            "This probability is the model's estimate for dataset label 1. "
            "It is not a personal medical risk percentage or diagnosis."
        )

        st.dataframe(patient, use_container_width=True)
                pdf_bytes = create_prediction_pdf(
            patient, predicted_label, probability
        )

        st.download_button(
            label="📄 Download Prediction Report (PDF)",
            data=pdf_bytes,
            file_name="diabetes_prediction_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        st.info(
            "For real health concerns, consult a qualified healthcare "
            "professional. Do not use this demo to make treatment decisions."
        )

# =========================================================
# MODEL EVALUATION
# =========================================================
elif page == "Model Evaluation":
    st.subheader("🧪 Model Evaluation")
    st.write(
        "Metrics below are calculated using the held-out test split."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy:.1%}")
    m2.metric("Precision", f"{precision:.1%}")
    m3.metric("Recall", f"{recall:.1%}")
    m4.metric("F1-score", f"{f1:.1%}")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Confusion Matrix")
        matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
        matrix_df = pd.DataFrame(
            matrix,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        )
        st.dataframe(matrix_df, use_container_width=True)

    with right:
        st.subheader("Classification Report")
        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Label 0", "Label 1"],
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(
            report_df.round(3),
            use_container_width=True,
        )

    st.divider()
    st.subheader("Model Coefficients")
    st.caption(
        "Coefficient direction and size are model associations after "
        "feature scaling, not proof of cause or medical importance."
    )

    classifier = model.named_steps["classifier"]
    coefficients = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Coefficient": classifier.coef_[0],
        }
    )
    coefficients["Absolute value"] = coefficients["Coefficient"].abs()
    coefficients = coefficients.sort_values(
        "Absolute value", ascending=False
    )

    st.bar_chart(
        coefficients.set_index("Feature")[["Coefficient"]]
    )
    st.dataframe(
        coefficients.drop(columns=["Absolute value"]).round(4),
        use_container_width=True,
    )

# =========================================================
# DATASET EXPLORER
# =========================================================
elif page == "Dataset Explorer":
    st.subheader("🗂️ Dataset Explorer")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric(
        "Missing cells after cleaning",
        int(df.isna().sum().sum()),
    )

    st.divider()
    st.subheader("Preview Dataset")

    rows_to_show = st.selectbox(
        "Number of rows to display",
        [10, 25, 50, 100],
        index=1,
    )
    st.dataframe(
        df.head(rows_to_show),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Summary Statistics")
        st.dataframe(
            df.describe().transpose().round(2),
            use_container_width=True,
        )

    with right:
        st.subheader("Missing Values by Column")
        missing_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Missing values": [
                    int(df[col].isna().sum()) for col in df.columns
                ],
            }
        )
        st.dataframe(
            missing_df,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("Download Cleaned Dataset")
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_data,
        file_name="diabetes_dataset_cleaned.csv",
        mime="text/csv",
    )

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "Diabetes Prediction System | Python • Streamlit • Scikit-learn"
)
st.caption(
    "For educational use only — not a clinically validated diagnostic tool."
)
