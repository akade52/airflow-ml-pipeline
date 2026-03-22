import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PowerTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os

DATA_URL = 'https://raw.githubusercontent.com/dayekb/Basic_ML_Alg/main/cars_moldova_no_dup.csv'
DATA_PATH = '/tmp/cars.csv'
CLEAR_DATA_PATH = '/tmp/df_clear.csv'
MODEL_PATH = '/tmp/lr_cars.pkl'

def download_data():
    df = pd.read_csv(DATA_URL, delimiter=',')
    df.to_csv(DATA_PATH, index=False)
    print(f"Загружено {len(df)} строк")
    return len(df)

def clear_data():
    df = pd.read_csv(DATA_PATH)
    cat_cols = ['Make', 'Model', 'Style', 'Fuel_type', 'Transmission']
    
    df = df.drop(df[(df.Year < 2021) & (df.Distance < 1100)].index)
    df = df.drop(df[df.Distance > 1e6].index)
    df = df.drop(df[df["Engine_capacity(cm3)"] < 200].index)
    df = df.drop(df[df["Engine_capacity(cm3)"] > 5000].index)
    df = df.drop(df[df["Price(euro)"] < 101].index)
    df = df.drop(df[df["Price(euro)"] > 1e5].index)
    df = df.drop(df[df.Year < 1971].index)
    
    ord_enc = OrdinalEncoder()
    ord_enc.fit(df[cat_cols])
    df[cat_cols] = ord_enc.transform(df[cat_cols])
    df = df.reset_index(drop=True)
    df.to_csv(CLEAR_DATA_PATH, index=False)
    print(f"Очищено {len(df)} строк")
    return len(df)

def train_model():
    df = pd.read_csv(CLEAR_DATA_PATH)
    X = df.drop(columns=['Price(euro)'])
    y = df['Price(euro)']
    
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scaled = scaler.fit_transform(X)
    y_scaled = power_trans.fit_transform(y.values.reshape(-1, 1))
    
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)
    
    lr = SGDRegressor(random_state=42, max_iter=1000, alpha=0.01)
    lr.fit(X_train, y_train.ravel())
    
    y_pred_scaled = lr.predict(X_val)
    y_pred = power_trans.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_true = power_trans.inverse_transform(y_val)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    joblib.dump(lr, MODEL_PATH)
    
    print(f"Модель обучена!")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    return rmse

def validate_model():
    if os.path.exists(MODEL_PATH):
        print(f"Модель сохранена: {MODEL_PATH}")
        return True
    else:
        raise Exception("Модель не найдена!")

default_args = {
    'owner': 'student',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    dag_id="ml_pipeline_cars",
    default_args=default_args,
    start_date=datetime(2025, 3, 1),
    schedule=timedelta(minutes=5),
    catchup=False,
    max_active_runs=1,
)

download_task = PythonOperator(
    task_id="download_data",
    python_callable=download_data,
    dag=dag,
)

clear_task = PythonOperator(
    task_id="clear_data",
    python_callable=clear_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag,
)

validate_task = PythonOperator(
    task_id="validate_model",
    python_callable=validate_model,
    dag=dag,
)

download_task >> clear_task >> train_task >> validate_task
