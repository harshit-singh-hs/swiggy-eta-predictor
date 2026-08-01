import pandas as pd
import numpy as np
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import VotingRegressor
import xgboost as xgb
import lightgbm as lgb
import mlflow

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def clean_data(df):
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df.replace('NaN', np.nan, inplace=True)
    df.dropna(inplace=True)
    
    df['Time_taken(min)'] = df['Time_taken(min)'].str.replace('(min) ', '', regex=False).astype(float)
    df['Weatherconditions'] = df['Weatherconditions'].str.replace('conditions ', '', regex=False)
    
    df['Delivery_person_Age'] = df['Delivery_person_Age'].astype(float)
    df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].astype(float)
    df['multiple_deliveries'] = df['multiple_deliveries'].astype(float)
    
    df['Distance_KM'] = haversine(
        df['Restaurant_latitude'], df['Restaurant_longitude'],
        df['Delivery_location_latitude'], df['Delivery_location_longitude']
    )
    return df

def train_and_evaluate():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "real_train.csv")
    if not os.path.exists(data_path):
        print("Real data not found.")
        return

    df = clean_data(pd.read_csv(data_path))
    target = 'Time_taken(min)'
    numeric_features = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Distance_KM', 'Vehicle_condition', 'multiple_deliveries']
    categorical_features = ['Weatherconditions', 'Road_traffic_density', 'Type_of_vehicle', 'Festival', 'City']
    
    X = df[numeric_features + categorical_features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    # Fit preprocessor and get feature names
    preprocessor.fit(X_train)
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(cat_feature_names)
    
    # Transform data and convert back to DataFrame for SHAP compatibility
    X_train_processed = pd.DataFrame(preprocessor.transform(X_train), columns=feature_names)
    X_test_processed = pd.DataFrame(preprocessor.transform(X_test), columns=feature_names)
    
    # MLflow tracking
    mlflow.set_experiment("Swiggy_ETA_Ensemble")
    with mlflow.start_run():
        print("Training XGBoost + LightGBM Ensemble...")
        
        xgb_model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
        lgb_model = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
        
        ensemble = VotingRegressor(estimators=[('xgb', xgb_model), ('lgb', lgb_model)])
        ensemble.fit(X_train_processed, y_train)
        
        y_pred = ensemble.predict(X_test_processed)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_params({"xgb_n_estimators": 150, "lgb_n_estimators": 150, "learning_rate": 0.05})
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        
        print(f"Metrics - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}")
        
        # Save metrics to a file for the Streamlit dashboard
        metrics_dict = {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "R2": round(r2, 3)}
        with open(os.path.join(base_dir, "mlflow_metrics.json"), "w") as f:
            json.dump(metrics_dict, f)
        
        # Save models and preprocessor
        output_dir = base_dir
        joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))
        joblib.dump(ensemble, os.path.join(output_dir, "ensemble_model.joblib"))
        
        # Save isolated XGBoost model for SHAP (VotingRegressor isn't directly supported by TreeExplainer)
        xgb_model.fit(X_train_processed, y_train)
        joblib.dump(xgb_model, os.path.join(output_dir, "shap_explainer_model.joblib"))
        
        print("Models, preprocessor, and metrics saved successfully.")

if __name__ == "__main__":
    train_and_evaluate()
