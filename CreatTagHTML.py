options = []

with open("Address_Encoded.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# حذف خط اول (هدر)
lines = lines[1:]

for line in lines:
    line = line.strip()
    if not line:
        continue

    # جدا کردن عدد از آخر خط
    parts = line.rsplit(" ", 1)

    if len(parts) != 2:
        print("خط مشکل‌دار:", line)
        continue

    address = parts[0].strip()
    code = parts[1].strip()

    option_tag = f'<option value="{code}">{address}</option>'
    options.append(option_tag)

# ذخیره خروجی
with open("address_options.txt", "w", encoding="utf-8") as f:
    for opt in options:
        f.write(opt + "\n")

print("✔ فایل address_options.txt ساخته شد.")
