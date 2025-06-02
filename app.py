from flask import Flask, render_template, request, jsonify
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
FEATURES_PATH = os.path.join(BASE_DIR, 'model/config/', 'feature_values.json')

model = joblib.load(MODEL_PATH)
kmeans = joblib.load(KMEANS_PATH)

with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
    feature_values = json.load(f)

# --- Ánh xạ ngược từ tiếng Việt sang tiếng Anh ---
reverse_mapping = {
    "Số sàn": "Manual",
    "Số tự động": "Automatic",
    "Dầu Diesel": "Diesel",
    "Xăng": "Petrol",
    "Khí hóa lỏng (LPG)": "LPG",
    "Khí thiên nhiên (CNG)": "CNG",
    "Cá nhân": "Individual",
    "Đại lý": "Dealer",
    "Đại lý uy tín": "Trustmark Dealer",
    "Chủ đầu tiên": "First Owner",
    "Chủ thứ hai": "Second Owner",
    "Nhiều chủ": "Multiple Owners"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # --- Nhận dữ liệu ---
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            data = request.get_json()
        else:
            data = request.form

        # --- Tiền xử lý dữ liệu ---
        company = data['company']
        model_input = data['model']
        fuel_type = reverse_mapping.get(data['fuel_type'], data['fuel_type'])
        seller_type = reverse_mapping.get(data['seller_type'], data['seller_type'])
        transmission = reverse_mapping.get(data['transmission'], data['transmission'])
        owner = reverse_mapping.get(data['owner'], data['owner'])
        car_age = int(data['car_age'])
        km_driven = int(data['km_driven'])

        # --- Feature bổ sung ---
        km_driven_log = np.log1p(km_driven)
        avg_km_per_year = km_driven / car_age if car_age != 0 else 0
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
        # Đảm bảo cluster_encoded có đủ các cột như lúc train KMeans
        cluster_encoded = cluster_encoded.reindex(columns=kmeans.feature_names_in_, fill_value=0)
        model_cluster = int(kmeans.predict(cluster_encoded)[0])

        # --- DataFrame đầu vào ---
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

        # --- Ép kiểu các cột phân loại về string ---
        categorical_cols = ['company', 'model', 'fuel_type', 'seller_type',
                            'transmission', 'owner', 'model_cluster', 'company_fuel_owner']
        for col in categorical_cols:
            input_df[col] = input_df[col].astype(str)

        # --- Kiểm tra thiếu cột ---
        missing_cols = set(model.feature_names_in_) - set(input_df.columns)
        if missing_cols:
            return jsonify({'error': f'Không đủ cột cho mô hình: {missing_cols}'}), 400

        # --- Dự đoán ---
        try:
            input_data = input_df[model.feature_names_in_]
            print("Input DataFrame:", input_data)
            print("Data Types:\n", input_data.dtypes)
            predicted_price = model.predict(input_data)[0]
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Lỗi khi dự đoán: {str(e)}'}), 500

        # --- Trả kết quả ---
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'prediction': int(predicted_price)})
        else:
            return render_template('index.html',
                                   prediction=int(predicted_price),
                                   feature_values=feature_values)

    return render_template('index.html', feature_values=feature_values)

if __name__ == '__main__':
    app.run(debug=True)
