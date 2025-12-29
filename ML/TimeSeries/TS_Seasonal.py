import numpy as np
import pandas as pd

def fourier_features(index, freq, order):
    time = np.arange(len(index),dtype=np.float32)
    k = 2 * np.pi*(1/freq)*time
    features = {}
    for i in range(1,order+1):
        features.update({
            f"sin_{freq}_{i}": np.sin(k*i),
            f"cos_{freq}_{i}": np.cos(k*i)
        })
    return  pd.DataFrame(features,index=index)
