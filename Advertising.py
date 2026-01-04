import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
df = pd.read_csv('Advertising.csv',usecols=['TV', 'radio', 'newspaper', 'sales'])
Q1 = df['sales'].quantile(0.25)
Q3 = df['sales'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['sales']>= lower_bound) & (df['sales'] <= upper_bound)]
X = df[['TV', 'radio', 'newspaper']]
y = df['sales']
ss = MinMaxScaler()
x_Scaled = ss.fit_transform(X)
x_scaled_df = pd.DataFrame(x_Scaled, columns=['TV', 'radio', 'newspaper'])
new_df = pd.concat([x_scaled_df, y], axis=1)
x_train,x_test,y_train,y_test = train_test_split(x_scaled_df,y,test_size=0.3,random_state=42)
le = LinearRegression()
le.fit(x_train,y_train)
y_pridict = le.predict(x_test)
# print(df2['sales'].describe())

# print(df.columns)
# print(df.shape)
# print(df.info())
# sns.boxplot(data=df,y= 'sales')
# sns.pairplot(new_df)
# plt.show()
# sns.histplot(new_df['sales'])
# plt.show()
# print(df['sales'].skew())
y_residual = y_test-y_pridict
# sns.scatterplot(x = y_test , y = y_residual)
# plt.axhline(y = 0, color = 'b' , linestyle= '--')
# plt.axhline(y = 1.96, color = 'r' , linestyle = '--')
# plt.axhline(y = -1.96, color = 'r' , linestyle = '--')
# plt.show()
mae = mean_absolute_error(y_true=y_test,y_pred=y_pridict)
mse = mean_squared_error(y_true=y_test,y_pred=y_pridict)
rmse = np.sqrt(mse)
r2 = f"{r2_score(y_true=y_test,y_pred=y_pridict):.2%}"
print(mae)
print(mse)
print(rmse)
print(r2)
finalModel = LinearRegression()
finalModel.fit(X= X.values,y = y)
print(finalModel.coef_)

y_hat = finalModel.predict(X)
r2total = r2_score(df['sales'] , y_hat)
print(f'R2 = {r2total:.1%}')
fig , axes = plt.subplots(nrows=1,ncols=3 , figsize=(15,5))
sns.scatterplot(x= X['TV']  , y = df['sales'] , ax = axes[0] ,color = 'r'  )
sns.scatterplot(x= X['TV']  , y = y_hat , ax = axes[0] ,color = 'b'  )

sns.scatterplot(x= X['radio']  , y = df['sales'] , ax = axes[1] ,color = 'r'  )
sns.scatterplot(x= X['radio']  , y = y_hat , ax = axes[1] ,color = 'b'  )

sns.scatterplot(x= X['newspaper']  , y = df['sales'] , ax = axes[2] ,color = 'r'  )
sns.scatterplot(x= X['newspaper']  , y = y_hat , ax = axes[2] ,color = 'b'  )
plt.tight_layout()
plt.show()
new_sample = [[1001,0,0]]
print(finalModel.predict(new_sample))
from joblib import dump , load
# dump(finalModel,'finalmodel.pkl')
loaded_model = load('finalmodel.pkl')
print(loaded_model.predict(new_sample))
from tkinter import *
import ttkbootstrap as tb
from tkinter import messagebox
import numpy as np
from joblib import load

# -----------------------------
# لود مدل
# -----------------------------
loaded_model = load("finalmodel.pkl")   # مسیر مدل را درست بگذار

# -----------------------------
# ساخت پنجره اصلی
# -----------------------------
window = tb.Window(themename="cosmo")
window.title("Advertising Prediction Form")
window.geometry("400x350")
window.resizable(False, False)
# icon = PhotoImage(file="myicon.png")
# window.iconphoto(True, icon)

title_label = tb.Label(
    window,
    text="فرم ورود داده‌های تبلیغات",
    font=("IRANSans", 16, "bold"),
    bootstyle="primary"
)
title_label.pack(pady=15)

form_frame = tb.Frame(window)
form_frame.pack(pady=10)

# -----------------------------
# ورودی TV
# -----------------------------
label_tv = tb.Label(form_frame, text="TV:", font=("IRANSans", 12))
label_tv.grid(row=0, column=0, padx=10, pady=10, sticky="w")

entry_tv = tb.Entry(form_frame, width=25, bootstyle="info")
entry_tv.grid(row=0, column=1, padx=10, pady=10)

# -----------------------------
# ورودی Radio
# -----------------------------
label_radio = tb.Label(form_frame, text="Radio:", font=("IRANSans", 12))
label_radio.grid(row=1, column=0, padx=10, pady=10, sticky="w")

entry_radio = tb.Entry(form_frame, width=25, bootstyle="info")
entry_radio.grid(row=1, column=1, padx=10, pady=10)

# -----------------------------
# ورودی Newspaper
# -----------------------------
label_news = tb.Label(form_frame, text="Newspaper:", font=("IRANSans", 12))
label_news.grid(row=2, column=0, padx=10, pady=10, sticky="w")

entry_news = tb.Entry(form_frame, width=25, bootstyle="info")
entry_news.grid(row=2, column=1, padx=10, pady=10)

# -----------------------------
# تابع Predict
# -----------------------------
def on_predict():
    try:
        # گرفتن ورودی‌ها
        tv = float(entry_tv.get())
        radio = float(entry_radio.get())
        newspaper = float(entry_news.get())

        # ساخت آرایه 2بعدی برای مدل
        new_sample = np.array([[tv, radio, newspaper]])

        # پیش‌بینی
        prediction = loaded_model.predict(new_sample)[0]

        # نمایش نتیجه
        messagebox.showinfo("Prediction Result", f"Predicted Sales: {prediction:.2f}")

    except ValueError:
        messagebox.showerror("Error", "لطفاً فقط عدد وارد کنید!")

# -----------------------------
# دکمه Predict
# -----------------------------
predict_btn = tb.Button(
    window,
    text="Predict",
    bootstyle="success",
    width=20,
    command=on_predict
)
predict_btn.pack(pady=20)

window.mainloop()
