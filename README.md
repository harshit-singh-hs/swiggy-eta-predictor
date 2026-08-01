# 🍔 Swiggy ETA Predictor

Welcome to the **Swiggy ETA Predictor**, a senior-level Data Science and Machine Learning project designed to predict the Estimated Time of Arrival (ETA) for food deliveries using a real-world logistics dataset. 

This project goes beyond simple predictions by implementing **Explainable AI (SHAP)**, **Geospatial Hotspot Mapping (Folium)**, and **Enterprise Model Tracking (MLflow)**.

---

## 📊 About the Dataset
This model was trained on a **real-world dataset containing over 45,000 food delivery records** across India. The data features high-cardinality geographic coordinates, weather conditions, traffic density, and delivery partner demographics.
* **Target Variable:** `Time_taken(min)`
* **Key Features:** `Delivery_person_Age`, `Delivery_person_Ratings`, `Distance_KM` (calculated via Haversine formula from GPS coordinates), `Weather`, `Traffic_Density`, `Vehicle_condition`.

---

## 🚀 Features & Dashboard

The dashboard is built entirely in **Streamlit** and features a 3-tab layout:

### 1. Predict & Explain (SHAP)
Input live order details and instantly get the predicted ETA. More importantly, this tab generates an interactive **SHAP Waterfall Plot** that explains the "why" behind the AI's decision. 
* **Red Bars:** Factors that *increased* the ETA (e.g., Heavy Traffic, Rain).
* **Blue Bars:** Factors that *decreased* the ETA (e.g., Short Distance, High Driver Rating).

![Predict & Explain SHAP Plot](assets/predict.png)  
*(Note the red and blue SHAP bars showing exactly how the model calculates the final minutes!)*

### 2. Geospatial Delay Hotspots Map
An interactive map (built with **Folium**) that visualizes the exact GPS coordinates of historical delivery bottlenecks. It filters out normal deliveries and highlights zones where orders took **more than 40 minutes**.

![Folium Heatmap](assets/heatmap.png)

### 3. MLOps Metrics
Proves the model's robustness by displaying live metrics (`RMSE`, `MAE`, `R2`) generated locally via **MLflow** during the training pipeline.

![MLflow Metrics](assets/metrics.png)

---

## 🛠️ Architecture & Tech Stack
* **Data Processing:** `pandas`, `numpy`, `scikit-learn` (Pipelines, ColumnTransformers)
* **Ensemble Modeling:** `VotingRegressor` combining `XGBoost` and `LightGBM`.
* **Explainable AI (XAI):** `SHAP` (TreeExplainer)
* **Geospatial Mapping:** `Folium`, `CartoDB`
* **MLOps / Tracking:** `MLflow`
* **Frontend:** `Streamlit`

---

## 💻 How to Run Locally

1. **Clone this repository:**
   ```bash
   git clone https://github.com/harshit-singh-hs/swiggy-eta-predictor.git
   cd swiggy-eta-predictor
   ```
2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the Dashboard:**
   ```bash
   python -m streamlit run streamlit_app.py
   ```
