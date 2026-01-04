import pandas as pd

# فایل خودت
df = pd.read_csv("Address_Encoded.txt", sep=r"\s{2,}", engine="python")

# فقط ستون‌های لازم
df = df[["Address", "Address_Encoded"]]

# مرتب‌سازی بر اساس نام محله (آدرس)
df = df.sort_values("Address")

items = []
for _, row in df.iterrows():
    name = str(row["Address"]).strip()
    code = int(row["Address_Encoded"])
    items.append({"name": name, "code": code})

# خروجی به شکل جاوااسکریپت
print("const neighborhoods = [")
for item in items:
    print(f'    {{"name": "{item["name"]}", "code": {item["code"]}}},')
print("];")
