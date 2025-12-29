import numpy as np
import pandas as pd

def fourier_features(index, freq, order):
    time = np.arange(len(index), dtype=np.float32)
    k = 2 * np.pi * (1 / freq) * time
    features = {}
    for i in range(1, order + 1):
        features.update({
            f"sin_{freq}_{i}": np.sin(i * k),
            f"cos_{freq}_{i}": np.cos(i * k),
        })
    return pd.DataFrame(features, index=index)

# ۲. ساخت داده نمایشی برای یک هفته (۷ روز)
days = pd.Index(["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"], name="Day")

# ۳. اجرای تابع
# freq=7: طول دوره یک هفته است
# order=1: فقط یک دایره (هارمونیک اول) بساز
result = fourier_features(days, freq=7, order=1)

print(result)