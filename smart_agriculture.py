
"""
smart_agriculture.py
Simulated IoT data generation and simple yield prediction using Random Forest.
Run:
    python smart_agriculture.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def generate_synthetic_data(n=1000, random_state=42):
    np.random.seed(random_state)
    # features: soil_moisture, temp, humidity, light, rainfall, fertilizer
    soil = np.random.uniform(0.1, 0.5, size=n)
    temp = np.random.uniform(15, 35, size=n)
    hum = np.random.uniform(30, 90, size=n)
    light = np.random.uniform(100, 1000, size=n)
    rain = np.random.uniform(0, 200, size=n)
    fert = np.random.uniform(0, 5, size=n)
    # simplistic yield function + noise
    yield_kg = (soil*50) + ( (30 - np.abs(temp-25)) * 0.8 ) + (light/500) + (fert*10) + np.random.normal(0,5,size=n)
    df = pd.DataFrame({
        'soil':soil, 'temp':temp, 'hum':hum, 'light':light, 'rain':rain, 'fert':fert, 'yield':yield_kg
    })
    return df

def train_model(df):
    X = df[['soil','temp','hum','light','rain','fert']]
    y = df['yield']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)
    model = RandomForestRegressor(n_estimators=100, random_state=1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Test MSE: {mse:.3f}")
    return model

if __name__ == "__main__":
    df = generate_synthetic_data(1000)
    model = train_model(df)
    # Save a simple model using joblib if available
    try:
        import joblib
        joblib.dump(model, 'yield_model.joblib')
        print("Saved yield_model.joblib")
    except Exception as e:
        print("joblib not available; skipping model save.", e)
