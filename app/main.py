"""
FastAPI app برای پیش‌بینی قیمت خانه + سرو کردن فرم HTML
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import joblib
import numpy as np

# ------------------------
# تنظیم مسیرهای پروژه
# ------------------------

# BASE_DIR = پوشه اصلی پروژه (جایی که پوشه‌های app، model، data قرار دارند)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# مسیر کامل فایل‌های مدل
MODEL_PATH = os.path.join(BASE_DIR, "model", "linear_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

# مسیر فایل HTML (فرض می‌کنیم index.html در روت پروژه است)
HTML_PATH = os.path.join(BASE_DIR, "index.html")

# ------------------------
# ساخت اپ FastAPI
# ------------------------

app = FastAPI()

# (الان چون قرار است HTML را هم از همین سرور سرو کنیم، حتی CORS هم ضروری نیست
# ولی برای آینده نگهش می‌داریم)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # برای توسعه، همه Originها اجازه دارند
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# لود مدل و اسکیلر
# ------------------------

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ------------------------
# سرو کردن فرم HTML روی /
# ------------------------

@app.get("/", response_class=HTMLResponse)
def read_root():
    """
    این تابع فایل index.html را می‌خواند و همان را به مرورگر برمی‌گرداند.
    یعنی وقتی بروی http://127.0.0.1:8000/ فرم را می‌بینی.
    """
    with open(HTML_PATH, encoding="utf-8") as f:
        return f.read()

# ------------------------
# endpoint پیش‌بینی قیمت
# ------------------------

@app.post("/predict")
def predict_price(data: dict):
    """
    دریافت داده‌ها از فرم (از طریق fetch در JavaScript)،
    نرمال‌سازی و پیش‌بینی قیمت.
    """

    features = np.array([[
        data["Elevator"],
        data["Floor"],
        data["Area"],
        data["Parking"],
        data["Room"],
        data["Warehouse"],
        data["Age"],
        data["Address_Encoded"],
        data["HasParkingElevatorWarehouse"]
    ]])

    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]

    return {"predicted_price": round(prediction)}
