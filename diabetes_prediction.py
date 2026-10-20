import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("diabetes.csv")

# Input features
features = [
	"Pregnancies",
	"Glucose",
	"BloodPressure",
	"SkinThickness",
	"Insulin",
	"BMI",
	"DiabetesPedigreeFunction",
	"Age",
]

# Separate inputs and output
X = df[features]
y = df["Outcome"]

# Split dataset into training and testing data
Xtr, Xte, ytr, yte = train_test_split(
	X, y, test_size=0.2, random_state=42, stratify=y
)

# Create Machine Learning model
model = make_pipeline(
	SimpleImputer(strategy="median"),
	StandardScaler(),
	LogisticRegression(max_iter=1000),
)

# Train model
model.fit(Xtr, ytr)

# Evaluate model
predictions = model.predict(Xte)

print("Classification Report:")
print(classification_report(yte, predictions))

print("Confusion Matrix:")
print(confusion_matrix(yte, predictions))

# Get user inputs
print("\nEnter sample values for prediction:")
values = [float(input(f"{col}: ")) for col in features]

# Predict dataset label
prediction = model.predict(pd.DataFrame([values], columns=features))[0]

if prediction == 1:
	print("Predicted dataset label: Positive")
else:
	print("Predicted dataset label: Negative")

print("Note: This is an educational model, not a medical diagnosis.")
