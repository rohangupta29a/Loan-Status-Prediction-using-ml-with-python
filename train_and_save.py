import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("loan_approval_dataset.csv")

df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip().str.lower()

df = df.drop(columns=["loan_id"], errors="ignore")

y = df["loan_status"].map({"approved": 1, "rejected": 0})

mask = y.notna()
df = df[mask]
y = y[mask]

X = df.drop(columns=["loan_status"])

leakage_cols = [
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]
X = X.drop(columns=leakage_cols, errors="ignore")

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_cols),
    ]
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=8,
    min_samples_split=20,
    min_samples_leaf=10,
)

pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

print(classification_report(y_test, pred))

joblib.dump(pipe, "model/loan_pipeline.pkl")
print("Saved: model/loan_pipeline.pkl")