import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/loan_approval_dataset.csv")

# Drop ID
df = df.drop(columns=["loan_id"])

# Target
y = df["loan_status"].map({"Approved": 1, "Rejected": 0})
X = df.drop(columns=["loan_status"])

# OPTIONAL (recommended): drop leakage columns if you did that in your project
leakage_cols = [
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]
X = X.drop(columns=[c for c in leakage_cols if c in X.columns])

cat_cols = ["education", "self_employed"]
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ]
)

model = RandomForestClassifier(n_estimators=200, random_state=42)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe.fit(X_train, y_train)

joblib.dump(pipe, "model/loan_pipeline.pkl")
print("Saved: model/loan_pipeline.pkl")
