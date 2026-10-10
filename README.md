# India House Price Prediction Using Machine Learning

## Project Overview

This project uses Machine Learning to predict house market prices in India based on property characteristics such as super area, carpet area, city, locality tier, and furnishing status.

An XGBoost Regressor is trained on the dataset to estimate house prices. The project uses Pandas and Scikit-learn for data preprocessing, feature engineering, model training, and evaluation.

## Objectives

- Predict house market prices using property-related features.
- Process categorical data using One-Hot Encoding.
- Standardize area-related numerical features using StandardScaler.
- Train a regression model using XGBoost.
- Evaluate model performance using MAE, MSE, RMSE, and R² score.
- Save the trained model and preprocessing information using Joblib for future predictions.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- XGBoost
- Joblib

## Machine Learning Workflow

1. Load and inspect the house price dataset.
2. Extract available cities, locality tiers, and furnishing categories.
3. Remove unnecessary columns.
4. Apply One-Hot Encoding to categorical features.
5. Standardize super area and carpet area.
6. Separate input features and the target variable.
7. Split the dataset into training and testing sets.
8. Train an XGBoost regression model.
9. Evaluate predictions using regression metrics.
10. Save the trained model, scaler, feature order, and category options in a Joblib file.

## Dataset

Dataset file: `house_price_dataset_india_12k.csv`

Target variable: `Market_Price_INR`

Input features include:
- Super_Area_sqft
- Carpet_Area_sqft
- City
- Locality_Tier
- Furnishing

The exact available categories and remaining features depend on the dataset.

## Model Evaluation

The model is evaluated using:

- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **R² Score:** Measures how well the model explains variation in house prices.

Actual evaluation results should be added after running the training script.

## Saved Model

The trained model package is saved as:

`India_house_price_model.joblib`

It contains:
- Trained XGBoost Regressor
- StandardScaler
- Exact feature-column order
- Available city options
- Locality-tier options
- Furnishing options

## How to Run

1. Clone or download this repository.
2. Install the required Python libraries.
3. Place the dataset in the appropriate project directory.
4. Run the model training script.
5. Check the evaluation metrics and generated model file.

Install dependencies:

```bash
pip install pandas scikit-learn xgboost joblib
```

Run the training script:

```bash
python train_model.py
```

Replace `train_model.py` with the actual name of your Python file if it is different.

## Future Improvements

- Build an interactive web application using Streamlit.
- Add visualizations for house prices across cities.
- Improve model performance through hyperparameter tuning.
- Compare XGBoost with other regression algorithms.
- Test the model on unseen data from reliable sources.

## Disclaimer

Predictions are estimates based on the training dataset and are not guaranteed to represent current market prices in every Indian city.