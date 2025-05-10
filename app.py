from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model & kmeans
model = joblib.load('model/car_price_model.pkl')
kmeans = joblib.load('model/car_price_kmeans.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        company = request.form['company']
        model_car = request.form['model']
        fuel_type = request.form['fuel_type']
        seller_type = request.form['seller_type']
        transmission = request.form['transmission']
        owner = request.form['owner']
        car_age = int(request.form['car_age'])
        km_driven = int(request.form['km_driven'])
        avg_km_per_year = km_driven / car_age if car_age != 0 else 0
        km_driven_log = np.log1p(km_driven)
        km_carage_interact = km_driven_log * car_age
        company_fuel_owner = f"{company}_{fuel_type}_{owner}"

        sample = pd.DataFrame([{
            'company': company,
            'model': model_car,
            'fuel_type': fuel_type,
            'seller_type': seller_type,
            'transmission': transmission,
            'owner': owner,
            'car_age': car_age,
            'avg_km_per_year': avg_km_per_year,
            'km_driven_log': km_driven_log,
            'company_fuel_owner': company_fuel_owner,
            'km_carage_interact': km_carage_interact
        }])

        # Tạo model_cluster
        cluster_data = pd.DataFrame([{
            'company': company,
            'model': model_car,
            'fuel_type': fuel_type,
            'transmission': transmission,
            'car_age': car_age,
            'km_driven_log': km_driven_log
        }])
        cluster_encoded = pd.get_dummies(cluster_data, columns=['company','model','fuel_type','transmission'])
        sample['model_cluster'] = kmeans.predict(cluster_encoded.reindex(columns=kmeans.feature_names_in_, fill_value=0)).astype(str)

        pred_price = model.predict(sample)[0]
        prediction = f"{pred_price:,.0f} VND"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
