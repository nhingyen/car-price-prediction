import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
import joblib
import optuna
import os

# --- Load data ---
car = pd.read_csv('D:\MachineLearning\project-final\dataset-kaggle\Car details v3.csv')

# --- Preprocessing ---
exchange_rate = 305
car[['company','model']] = car['name'].str.split(pat=' ', n=1, expand=True)
car['model'] = car['model'].astype(str).str.split().str[:3].str.join(' ')
car = car[['company','model','year','km_driven','fuel','seller_type','transmission','owner','selling_price']]
car.rename(columns={'fuel':'fuel_type','selling_price':'price'}, inplace=True)
car.dropna(inplace=True)
car.drop_duplicates(inplace=True)
car = car[car['owner']!='Test Drive Car']
car['owner'] = car['owner'].replace({
    'Third Owner':'Multiple Owners',
    'Fourth & Above Owner':'Multiple Owners'
})

car['car_age'] = 2025 - car['year']
car.drop(columns='year', inplace=True)
car['avg_km_per_year'] = car['km_driven'] / car['car_age']
car['avg_km_per_year'] = car['avg_km_per_year'].replace([np.inf, -np.inf], 0)
car['avg_km_per_year'] = car['avg_km_per_year'].fillna(0)
car['km_driven_log'] = np.log1p(car['km_driven'])
car['price'] = car['price'] * exchange_rate
car = car[(car['price']<2e8) & (car['km_driven']<500000) & (car['car_age']<=20)]

# Clustering
cluster_data = car[['company','model','fuel_type','transmission','car_age','km_driven_log']]
cluster_encoded = pd.get_dummies(cluster_data, columns=['company','model','fuel_type','transmission'])
kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
car['model_cluster'] = kmeans.fit_predict(cluster_encoded).astype(str)

# Additional features
car['company_fuel_owner'] = car['company'] + '_' + car['fuel_type'] + '_' + car['owner']
car['km_carage_interact'] = car['km_driven_log'] * car['car_age']

X = car.drop(columns=['price','km_driven'])
y = car['price']

numeric_features = ['car_age','km_driven_log','avg_km_per_year','km_carage_interact']
categorical_features = [
    'company','model','fuel_type','seller_type',
    'transmission','owner','model_cluster','company_fuel_owner'
]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
    }
    model = Pipeline([
        ('pre', preprocessor),
        ('xgb', XGBRegressor(**params, random_state=42, verbosity=0))
    ])
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=40)

# Final model
best_model = Pipeline([
    ('pre', preprocessor),
    ('xgb', XGBRegressor(**study.best_params, random_state=42))
])
best_model.fit(X_train, y_train)

# Save model & kmeans
base_dir = os.path.dirname(__file__)  # Thư mục chứa train_model.py
model_path = os.path.join(base_dir, 'model', 'car_price_model.pkl')
kmeans_path = os.path.join(base_dir, 'model', 'car_price_kmeans.pkl')

joblib.dump(best_model, model_path)
joblib.dump(kmeans, kmeans_path)
print(f"✅ Model saved to {model_path}")
