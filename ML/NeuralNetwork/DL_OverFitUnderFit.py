import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.rc("figure", autolayout=True)
plt.rc("axes", titlesize=18, labelsize='large', titleweight='bold', labelweight='bold', titlepad=10)
plt.rc("animation", html='html5')
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector,make_column_transformer
from sklearn.model_selection import GroupShuffleSplit
from IPython import display

from tensorflow import keras
from tensorflow.keras import layers , callbacks
spotify = pd.read_csv('../input/dl-course-data/spotify.csv')
X = spotify.copy().dropna()
y = X.pop('track_popularity')
artists = X['track_artist']
print(X.head())
print(y.head())
print(artists.head())
print(list(X.columns))
print(X.shape)
print(spotify.shape)
pd.set_option('display.max_columns', None)
print(spotify.loc[0,spotify.dtypes == 'object'])
print(spotify.dtypes)
feature_num = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms']
features_cat = ['playlist_genre']
preprocessor = make_column_transformer((StandardScaler(),feature_num),(OneHotEncoder(),features_cat))
def group_split(X, y, group, train_size=0.75):
    splitter = GroupShuffleSplit(train_size=train_size)
    train, test = next(splitter.split(X, y, groups=group))
    return (X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test])

X_train, X_valid, y_train, y_valid = group_split(X,y,artists)
print(" {} {} ".format(X_train.shape,X.shape))
print(X_train.head(50))
X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
y_train = y_train / 100
y_valid = y_valid / 100
input_shape = [X_train.shape[1]]
print("Input shape: {}".format(input_shape))
early_stopping =callbacks.EarlyStopping(
    patience= 20,
    restore_best_weights= True,
    min_delta= 0.001,
    verbose= 0
)
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='adam',loss='mae')
history = model.fit(
    X_train,y_train,
    validation_data=(X_valid, y_valid),
    epochs=50,
    batch_size=512,
    verbose=0,
    callbacks=[early_stopping]
)
history_df = pd.DataFrame(history.history)
history_df.loc[0:,['loss','val_loss']].plot()
print("Minimum loss: {:0.4f}".format(history_df['loss'].min()))
plt.show()

