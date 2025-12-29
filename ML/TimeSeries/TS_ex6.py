import pandas as pd

# ۱. فرض کن این داده‌های اصلی ماست (۵ روز اول سال)
# داده‌های ما شامل همه روزهاست
dates = pd.date_range("2024-01-01", periods=5)
df_main = pd.DataFrame({'traffic': [100, 120, 110, 130, 90]}, index=dates)

print("--- ۱. داده‌های اصلی (همه روزها) ---")
print(df_main)
print("\n")

# ۲. این لیست تعطیلات است (فقط روزهای خاص را داریم)
# مثلاً اول ژانویه (سال نو) و چهارم ژانویه (تولد)
holidays_data = {
    'date': ['2024-01-01', '2024-01-04'],
    'event': ['New Year', 'Birthday']
}
df_holidays = pd.DataFrame(holidays_data)
df_holidays['date'] = pd.to_datetime(df_holidays['date'])
df_holidays = df_holidays.set_index('date')

# ۳. تبدیل نام تعطیلات به اعداد ۰ و ۱ (One-Hot Encoding)
# این همان کاری است که get_dummies انجام می‌دهد
encoded_holidays = pd.get_dummies(df_holidays)

print("--- ۲. تعطیلات تبدیل شده به ۰ و ۱ ---")
print(encoded_holidays)
print("\n")

# ۴. چسباندن به داده‌های اصلی
# نکته مهم: روزهایی که تعطیل نیستند NaN می‌شوند، که با 0.0 پرشان می‌کنیم
final_df = df_main.join(encoded_holidays).fillna(0.0)

print("--- ۳. جدول نهایی (ترکیب شده) ---")
print(final_df)