import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import shap
import folium
import streamlit.components.v1 as components
from folium.plugins import HeatMap

# --- Page Configuration ---
st.set_page_config(page_title="Swiggy ETA Predictor (Pro)", page_icon="🍔", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #fc8019; font-family: 'Arial', sans-serif; }
    .stButton>button { background-color: #fc8019; color: white; font-weight: bold; width: 100%; border-radius: 8px; }
    .stButton>button:hover { background-color: #e06c11; color: white; }
    .metric-box { background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }
    </style>
""", unsafe_allow_html=True)

# --- Load Models & Assets ---
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(os.path.abspath(__file__))
    try:
        preprocessor = joblib.load(os.path.join(base_path, "preprocessor.joblib"))
        model = joblib.load(os.path.join(base_path, "ensemble_model.joblib"))
        shap_model = joblib.load(os.path.join(base_path, "shap_explainer_model.joblib"))
        raw_data = pd.read_csv(os.path.join(base_path, "real_train.csv"))
        
        with open(os.path.join(base_path, "mlflow_metrics.json"), "r") as f:
            metrics = json.load(f)
            
        return preprocessor, model, shap_model, metrics, raw_data
    except Exception as e:
        return None, None, None, None, None

preprocessor, model, shap_model, metrics, raw_data = load_assets()

# --- App Layout ---
st.title("🍔 Swiggy ETA Predictor (MLOps Edition)")
st.markdown("Advanced Machine Learning project featuring **Explainable AI (SHAP)**, **Geospatial Mapping**, and **Ensemble Modeling**.")

if model is None:
    st.error("Assets not found! Please run `train_model.py` to generate the models and metrics.")
else:
    tab1, tab2, tab3 = st.tabs(["🚀 Predict & Explain (SHAP)", "🗺️ Delay Hotspot Map", "📊 MLflow Metrics"])

    # --- TAB 1: PREDICTION & EXPLAINABILITY ---
    with tab1:
        col_input, col_result = st.columns([1, 1])
        
        with col_input:
            st.subheader("Order Details")
            with st.form("prediction_form"):
                distance = st.slider("Distance (KM)", 0.5, 25.0, 5.0, 0.1)
                dp_age = st.slider("Delivery Partner Age", 18, 50, 25)
                dp_rating = st.slider("Delivery Partner Rating", 1.0, 5.0, 4.5, 0.1)
                weather = st.selectbox("Weather Condition", ['Sunny', 'Stormy', 'Sandstorms', 'Windy', 'Fog', 'Cloudy'])
                traffic = st.selectbox("Traffic Density", ['Low', 'Medium', 'High', 'Jam'])
                vehicle_type = st.selectbox("Type of Vehicle", ['motorcycle', 'scooter', 'electric_scooter', 'bicycle'])
                vehicle_condition = st.selectbox("Vehicle Condition (0-2)", [0, 1, 2], index=2)
                multiple_deliveries = st.selectbox("Multiple Deliveries (Batched)", [0, 1, 2, 3], index=0)
                festival = st.selectbox("Festival Day?", ['No', 'Yes'])
                city_type = st.selectbox("City Type", ['Urban', 'Metropolitian', 'Semi-Urban'])
                
                submit_button = st.form_submit_button(label="Predict ETA & Explain")

        with col_result:
            if submit_button:
                # 1. Prepare Input
                input_df = pd.DataFrame({
                    'Delivery_person_Age': [float(dp_age)],
                    'Delivery_person_Ratings': [float(dp_rating)],
                    'Distance_KM': [distance],
                    'Vehicle_condition': [vehicle_condition],
                    'multiple_deliveries': [float(multiple_deliveries)],
                    'Weatherconditions': [weather],
                    'Road_traffic_density': [traffic],
                    'Type_of_vehicle': [vehicle_type],
                    'Festival': [festival],
                    'City': [city_type]
                })
                
                # 2. Predict using Ensemble
                prediction = model.predict(preprocessor.transform(input_df))[0]
                
                st.markdown(f"""
                    <div class="metric-box" style="margin-bottom: 20px;">
                        <h3>Estimated Delivery Time</h3>
                        <h1 style="color: #fc8019; font-size: 48px;">{int(round(prediction))} Minutes ⏱️</h1>
                    </div>
                """, unsafe_allow_html=True)

                # 3. Explain using SHAP (on XGBoost base model for speed)
                st.subheader("Why this ETA? (SHAP Explanation)")
                
                # Get feature names from preprocessor
                num_features = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Distance_KM', 'Vehicle_condition', 'multiple_deliveries']
                cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
                all_features = num_features + list(cat_features)
                
                # Transform data
                processed_input = pd.DataFrame(preprocessor.transform(input_df), columns=all_features)
                
                # Generate SHAP values
                explainer = shap.TreeExplainer(shap_model)
                shap_values = explainer(processed_input)
                
                # Plot
                fig, ax = plt.subplots(figsize=(8, 4))
                shap.plots.waterfall(shap_values[0], show=False)
                plt.tight_layout()
                st.pyplot(fig)
                
                base_val = explainer.expected_value
                if isinstance(base_val, np.ndarray): base_val = base_val[0]
                diff = prediction - base_val
                direction = "added" if diff > 0 else "reduced"
                
                st.info(f"⬆️ **ETA Explanation:** Under normal conditions, an average delivery takes **{int(base_val)} minutes**. Based on the order details you provided, the AI has {direction} **{abs(int(diff))} minutes** to the final estimate. Red bars show factors that increased the time, and blue bars decreased it.")

    # --- TAB 2: GEOSPATIAL MAP ---
    with tab2:
        st.subheader("High-Delay Zones (Geospatial Analysis)")
        st.markdown("This heatmap highlights areas where deliveries historically took **more than 40 minutes**. It helps logistics teams identify bottleneck zones.")
        
        # Clean data slightly for map (taking a sample for performance)
        map_data = raw_data[['Delivery_location_latitude', 'Delivery_location_longitude', 'Time_taken(min)']].dropna()
        map_data['Time_taken(min)'] = map_data['Time_taken(min)'].astype(str).str.replace('(min) ', '', regex=False).astype(float)
        
        delayed_orders = map_data[
            (map_data['Time_taken(min)'] > 40) & 
            (map_data['Delivery_location_latitude'] > 0) & 
            (map_data['Delivery_location_longitude'] > 0)
        ].head(5000)
        
        if not delayed_orders.empty:
            # Center map around average coordinates
            m = folium.Map(location=[delayed_orders['Delivery_location_latitude'].mean(), 
                                     delayed_orders['Delivery_location_longitude'].mean()], 
                           zoom_start=11, tiles="CartoDB dark_matter")
            
            # Add Heatmap
            heat_data = [[row['Delivery_location_latitude'], row['Delivery_location_longitude']] for index, row in delayed_orders.iterrows()]
            HeatMap(heat_data, radius=15).add_to(m)
            
            # Display in Streamlit bypassing st_folium bug
            components.html(m._repr_html_(), height=500)
        else:
            st.warning("Could not generate map data.")

    # --- TAB 3: MODEL PERFORMANCE ---
    with tab3:
        st.subheader("MLflow Training Metrics")
        st.markdown("We used **MLflow** locally to track our experiments. Our final model is a **Voting Ensemble** of XGBoost and LightGBM.")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("RMSE (Root Mean Squared Error)", f"{metrics.get('RMSE')} mins")
        col_m2.metric("MAE (Mean Absolute Error)", f"{metrics.get('MAE')} mins")
        col_m3.metric("R² Score", f"{metrics.get('R2')}")
        
        st.markdown("### Architecture")
        st.code("""
        Pipeline(
            preprocessor = ColumnTransformer(StandardScaler, OneHotEncoder),
            model = VotingRegressor([
                ('xgb', XGBRegressor(n_estimators=150, learning_rate=0.05)),
                ('lgb', LGBMRegressor(n_estimators=150, learning_rate=0.05))
            ])
        )
        """, language="python")
