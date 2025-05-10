from flask import Flask, render_template, request
import joblib
import json
import numpy as np
import os
import pandas as pd

app = Flask(__name__)

# --- Load model, kmeans, feature values ---
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'car_price_model.pkl')
KMEANS_PATH = os.path.join(BASE_DIR, 'model', 'car_price_kmeans.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'model', 'feature_values.json')

model = joblib.load(MODEL_PATH)
kmeans = joblib.load(KMEANS_PATH)

with open(FEATURES_PATH, 'r') as f:
    feature_values = json.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # --- Lấy dữ liệu từ form ---
        company = request.form['company']
        model_input = request.form['model']
        fuel_type = request.form['fuel_type']
        seller_type = request.form['seller_type']
        transmission = request.form['transmission']
        owner = request.form['owner']
        car_age = int(request.form['car_age'])
        km_driven = int(request.form['km_driven'])

        # --- Tạo các feature bổ sung ---
        km_driven_log = np.log1p(km_driven)
        avg_km_per_year = km_driven / car_age
        km_carage_interact = km_driven_log * car_age
        company_fuel_owner = f"{company}_{fuel_type}_{owner}"

        # --- Tạo cluster ---
        cluster_input = pd.DataFrame([{
            'company': company,
            'model': model_input,
            'fuel_type': fuel_type,
            'transmission': transmission,
            'car_age': car_age,
            'km_driven_log': km_driven_log
        }])
        cluster_encoded = pd.get_dummies(cluster_input)
        cluster_encoded = cluster_encoded.reindex(columns=kmeans.feature_names_in_, fill_value=0)
        model_cluster = str(kmeans.predict(cluster_encoded)[0])

        # --- Tạo DataFrame đầu vào ---
        input_df = pd.DataFrame([{
            'company': company,
            'model': model_input,
            'fuel_type': fuel_type,
            'seller_type': seller_type,
            'transmission': transmission,
            'owner': owner,
            'car_age': car_age,
            'km_driven_log': km_driven_log,
            'avg_km_per_year': avg_km_per_year,
            'km_carage_interact': km_carage_interact,
            'model_cluster': model_cluster,
            'company_fuel_owner': company_fuel_owner
        }])

        # --- Dự đoán ---
        predicted_price = model.predict(input_df)[0]

        return render_template('index.html',
                               prediction=int(predicted_price),
                               feature_values=feature_values)
    
    return render_template('index.html', feature_values=feature_values)

if __name__ == '__main__':
    app.run(debug=True)
