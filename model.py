import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(
    "house_price_dataset_india_12k.csv",
    sep=None,
    engine="python"
)

print("Dataset loaded successfully.")
print("Original shape:", df.shape)


# ============================================================
# 2. KEEP ORIGINAL DATASET
# ============================================================

# We keep this before preprocessing so that the frontend
# can later know the original City, Locality_Tier and
# Furnishing values.

df_original = df.copy()


# ============================================================
# 3. SAVE CATEGORY OPTIONS
# ============================================================

cities = sorted(
    df_original["City"]
    .dropna()
    .unique()
    .tolist()
)

locality_tiers = sorted(
    df_original["Locality_Tier"]
    .dropna()
    .unique()
    .tolist()
)

furnishing_options = sorted(
    df_original["Furnishing"]
    .dropna()
    .unique()
    .tolist()
)


print("\nCities:")
print(cities)

print("\nLocality Tiers:")
print(locality_tiers)

print("\nFurnishing Options:")
print(furnishing_options)


# ============================================================
# 4. DROP UNNECESSARY COLUMNS
# ============================================================

df = df.drop(
    columns=[
        "House_ID",
        "Price_per_sqft_INR"
    ]
)


# ============================================================
# 5. ONE-HOT ENCODING
# ============================================================

df = pd.get_dummies(
    df,
    columns=[
        "Locality_Tier",
        "City",
        "Furnishing"
    ],
    dtype=int
)


print("\nShape after one-hot encoding:")
print(df.shape)


# ============================================================
# 6. STANDARD SCALING
# ============================================================

ss = StandardScaler()


X_scaled = pd.DataFrame(
    ss.fit_transform(
        df[
            [
                "Super_Area_sqft",
                "Carpet_Area_sqft"
            ]
        ]
    ),
    columns=[
        "Super_Area_sqft",
        "Carpet_Area_sqft"
    ],
    index=df.index
)


# ============================================================
# 7. REPLACE ORIGINAL AREA COLUMNS
# ============================================================

df = pd.concat(
    [
        df.drop(
            columns=[
                "Super_Area_sqft",
                "Carpet_Area_sqft"
            ]
        ),
        X_scaled
    ],
    axis=1
)


# ============================================================
# 8. CREATE X AND y
# ============================================================

X = df.drop(
    columns=[
        "Market_Price_INR"
    ]
)

y = df[
    "Market_Price_INR"
]


# ============================================================
# 9. SAVE EXACT FEATURE ORDER
# ============================================================

feature_columns = X.columns.tolist()


print("\nNumber of features:", len(feature_columns))

print("\nExact feature order:")

for i, column in enumerate(feature_columns):
    print(i, column)


# ============================================================
# 10. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ============================================================
# 11. CREATE XGBOOST MODEL
# ============================================================

model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    random_state=42
)


# ============================================================
# 12. TRAIN MODEL
# ============================================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ============================================================
# 13. TEST MODEL
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 14. MODEL EVALUATION
# ============================================================

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print(
    "MSE:",
    mse
)

print(
    "RMSE:",
    rmse
)

print(
    "MAE:",
    mae
)

print(
    "R2 Score:",
    r2
)


# ============================================================
# 15. CREATE MODEL PACKAGE
# ============================================================

model_package = {

    # Trained XGBoost model
    "model": model,

    # StandardScaler used during training
    "scaler": ss,

    # EXACT order of model features
    "feature_columns": feature_columns,

    # Original category values
    "cities": cities,

    "locality_tiers": locality_tiers,

    "furnishing_options": furnishing_options
}


# ============================================================
# 16. SAVE EVERYTHING IN ONE FILE
# ============================================================

model_filename = (
    "India_house_price_model.joblib"
)

joblib.dump(
    model_package,
    model_filename
)


print("\n========================================")
print("MODEL SAVED SUCCESSFULLY")
print("========================================")

print(
    f"Saved as: {model_filename}"
)

print(
    "Model + scaler + feature columns + "
    "categories saved together."
)


# ============================================================
# 17. VERIFY SAVED MODEL
# ============================================================

loaded_package = joblib.load(
    model_filename
)


print("\n========================================")
print("VERIFICATION")
print("========================================")

print(
    "Saved object type:",
    type(loaded_package)
)

print(
    "Model type:",
    type(loaded_package["model"])
)

print(
    "Scaler type:",
    type(loaded_package["scaler"])
)

print(
    "Number of features:",
    len(
        loaded_package["feature_columns"]
    )
)

print(
    "Cities:",
    loaded_package["cities"]
)

print(
    "Locality tiers:",
    loaded_package["locality_tiers"]
)

print(
    "Furnishing options:",
    loaded_package["furnishing_options"]
)


# ============================================================
# 18. FINAL FEATURE CHECK
# ============================================================

print("\n========================================")
print("FINAL FEATURES")
print("========================================")

for i, feature in enumerate(
    loaded_package["feature_columns"]
):

    print(
        f"{i + 1}. {feature}"
    )


print("\n========================================")
print("DONE")
print("========================================")