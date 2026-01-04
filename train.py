# ============================================================
# هدف برنامه:
# ساخت یک مدل رگرسیون خطی برای پیش‌بینی قیمت خانه با استفاده
# از داده‌های خام، انجام پیش‌پردازش، حذف داده‌های پرت،
# نرمال‌سازی، آموزش مدل و ذخیره آن برای استفاده در API.
# ============================================================


# -----------------------------
# Import libraries
# -----------------------------

import pandas as pd              # کار با داده‌ها و ساخت DataFrame
import numpy as np               # محاسبات عددی و آرایه‌ها
import seaborn as sns            # رسم نمودارهای آماری (در این کد استفاده نشده)
from openpyxl import load_workbook
from sklearn.preprocessing import LabelEncoder, StandardScaler,MinMaxScaler   # تبدیل داده‌های متنی و نرمال‌سازی
from sklearn.model_selection import train_test_split             # تقسیم داده‌ها به train/test
from sklearn.linear_model import LinearRegression                # مدل رگرسیون خطی
from sklearn.metrics import mean_absolute_error, mean_squared_error  # معیارهای ارزیابی مدل
from joblib import dump         # ذخیره مدل و اسکیلر روی دیسک


# -----------------------------
# Load Data
# -----------------------------

df = pd.read_csv('data/HouseNew.csv', usecols=[
    'Elevator', 'Floor', 'Area', 'Parking', 'Room', 'Price',
    'Warehouse', 'YearOfConstruction', 'Address'
])
# خواندن فایل CSV و انتخاب فقط ستون‌های مورد نیاز


# -----------------------------
# Preprocessing
# -----------------------------

df['Address'] = df['Address'].fillna('نامشخص')
# پر کردن مقدارهای خالی ستون Address با مقدار "نامشخص"

mode_floor = df['Floor'].mode()[0]
# پیدا کردن پرتکرارترین مقدار ستون Floor

df['Floor'] = df['Floor'].fillna(mode_floor)
# جایگزینی مقدارهای خالی Floor با مقدار پرتکرار

df['Age'] = 1404 - df['YearOfConstruction']
# محاسبه سن ساختمان = سال جاری (۱۴۰۴) - سال ساخت

le = LabelEncoder()
df['Address_Encoded'] = le.fit_transform(df['Address'])
# تبدیل آدرس‌ها به عدد برای استفاده در مدل

df[['Address','Address_Encoded']].to_excel("output.xlsx", sheet_name="MySheet", index=False)
# ذخیره جدول آدرس‌ها و کدگذاری‌شان در یک فایل اکسل برای بررسی

unique_list = df[['Address', 'Address_Encoded']].drop_duplicates()
# حذف مقادیر تکراری و تهیه لیست یونیک از مقادیر آدرس و کد آدرس
with pd.ExcelWriter('output.xlsx',engine='openpyxl',mode='a',if_sheet_exists='replace') as writer:
    unique_list.to_excel(writer, sheet_name="Address_RemovedDuplicate", index=False)
# ذخیره جدول آدرس‌ها و کدگذاری‌شان که یونیک شده در یک فایل اکسل و در سیت آدرس ریموود داپلیکیت برای بررسی

print(unique_list)
# چاپ لیست یکتا از آدرس‌ها و کدگذاری‌شان

print(df['Address'].nunique())
# چاپ تعداد آدرس‌های یکتا

df['Elevator'] = df['Elevator'].astype('int')
# تبدیل ستون Elevator به عدد صحیح

df['Parking'] = df['Parking'].map({True: 1, False: 0})
# تبدیل True/False به 1/0 برای Parking

df['Warehouse'] = df['Warehouse'].map({True: 1, False: 0})
# تبدیل True/False به 1/0 برای Warehouse

df['HasParkingElevatorWarehouse'] = df['Elevator'] + df['Parking'] + df['Warehouse']
# ساخت یک ویژگی جدید: مجموع امکانات (آسانسور + پارکینگ + انباری)


# -----------------------------
# Remove Outliers (IQR Method)
# -----------------------------

Q1 = df['Price'].quantile(0.25)   # محاسبه چارک اول قیمت
Q3 = df['Price'].quantile(0.75)   # محاسبه چارک سوم قیمت
IQR = Q3 - Q1                     # فاصله بین چارکی

lower_bound = Q1 - 1.5 * IQR      # حد پایین مجاز
upper_bound = Q3 + 1.5 * IQR      # حد بالای مجاز

df2 = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)]
# حذف داده‌های پرت (outlier) بر اساس قیمت با روش IQR


# -----------------------------
# Features & Labels
# -----------------------------

X = df2[['Elevator', 'Floor', 'Area', 'Parking', 'Room', 'Warehouse',
         'Age', 'Address_Encoded', 'HasParkingElevatorWarehouse']]

# X = df2[['Elevator', 'Floor', 'Area', 'Parking', 'Room', 'Warehouse',
#          'Age', 'HasParkingElevatorWarehouse']]
# انتخاب ویژگی‌های ورودی مدل

y = df2['Price']
# انتخاب خروجی مدل (قیمت)


# -----------------------------
# Scaling
# -----------------------------

# scaler = StandardScaler()
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# نرمال‌سازی ویژگی‌ها برای بهبود عملکرد مدل

dump(scaler, 'model/scaler.pkl')
# ذخیره اسکیلر برای استفاده در API (برای ورودی‌های جدید)


# -----------------------------
# Train/Test Split
# -----------------------------

x_train, x_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)
# تقسیم داده‌ها: ۷۰٪ آموزش، ۳۰٪ تست


# -----------------------------
# Train Model
# -----------------------------

lr = LinearRegression()
# ساخت مدل رگرسیون خطی

lr.fit(x_train, y_train)
# آموزش مدل با داده‌های train

dump(lr, 'model/linear_model.pkl')
# ذخیره مدل آموزش‌دیده برای استفاده در API


# -----------------------------
# Evaluation
# -----------------------------

y_pred = lr.predict(x_test)
# پیش‌بینی قیمت‌ها روی داده‌های تست

MAE = mean_absolute_error(y_test, y_pred)
# محاسبه میانگین خطای مطلق

MSE = mean_squared_error(y_test, y_pred)
# محاسبه میانگین مربعات خطا

RMSE = np.sqrt(MSE)
# محاسبه ریشه میانگین مربعات خطا (در مقیاس واقعی قیمت)

print("MAE:", MAE)
print("MSE:", MSE)
print("RMSE:", RMSE)
# چاپ نتایج ارزیابی مدل
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)
print(r2) #r