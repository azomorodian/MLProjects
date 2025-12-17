import matplotlib
matplotlib.use('tkAgg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
autos = pd.read_csv('../input/fe-course-data/autos.csv')
print(autos.columns)
pd.set_option('display.max_columns', None)
A =autos.groupby("make")['price'].transform('mean')
Data = pd.DataFrame()
Data['make_encoded'] = A
autos = autos.join(Data)
print(autos)

