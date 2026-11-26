
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
print("Loading dataset...")

df = pd.read_csv("dataset/CAR DETAILS FROM CAR DEKHO.csv")

# 2. Extract car brand from car name
def extract_brand(name):
    name = str(name).strip()

    if name.lower().startswith("land rover"):
        return "Land Rover"

    return name.split()[0]

df["brand"] = df["name"].apply(extract_brand)

# 3. Select required columns
features = [
    "brand",
    "year",
    "km_driven",
    "fuel",
    "transmission"
]

target = "selling_price"

X = df[features]
y = df[target]

# 4. Identify categorical and numerical columns
categorical_features = [
    "brand",
    "fuel",
    "transmission"
]

numerical_features = [
    "year",
    "km_driven"
]

# 5. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

# 6. Create Random Forest model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    min_samples_leaf=2,
    n_jobs=-1
)

# 7. Create complete ML pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# 8. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Random Forest model...")

# 9. Train model
pipeline.fit(X_train, y_train)

# 10. Make predictions
predictions = pipeline.predict(X_test)

# 11. Evaluate model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n========== MODEL PERFORMANCE ==========")
print(f"Mean Absolute Error: Rs. {mae:,.2f}")
print(f"Root Mean Squared Error: Rs. {rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")
print(f"Accuracy indicator: {r2 * 100:.2f}%")
print("========================================")

# 12. Save trained model
os.makedirs("model", exist_ok=True)

joblib.dump(pipeline, "model/car_price_model.pkl")

# 13. Save model information
model_info = {
    "brands": sorted(df["brand"].unique().tolist()),
    "fuel_types": sorted(df["fuel"].unique().tolist()),
    "transmissions": sorted(df["transmission"].unique().tolist()),
    "min_year": int(df["year"].min()),
    "max_year": int(df["year"].max()),
    "min_km": int(df["km_driven"].min()),
    "max_km": int(df["km_driven"].max()),
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2)
}

joblib.dump(model_info, "model/model_info.pkl")

print("\nSUCCESS!")
print("Model saved successfully.")
print("Location: model/car_price_model.pkl")
