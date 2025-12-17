import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
plt.rc('figure',autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
accidents = pd.read_csv('../input/fe-course-data/accidents.csv')
autos = pd.read_csv('../input/fe-course-data/autos.csv')
concrete = pd.read_csv('../input/fe-course-data/concrete.csv')
customers = pd.read_csv('../input/fe-course-data/customer.csv')
#print full columns
pd.set_option('display.max_columns',None)
autos['stroke_ratio'] = autos.stroke/autos.bore
print(autos.head(1))
for c in autos.columns:
    print('{}:{}'.format(c,autos[c][1]))
print(autos[["stroke", "bore", "stroke_ratio"]].head())
print(accidents.columns)
accidents["LogWindSpeed"] = accidents.WindSpeed.apply(np.log1p)
fig, axs = plt.subplots(1,2,figsize=(8,4))
sns.kdeplot(accidents.WindSpeed, fill= True, ax=axs[0])
sns.kdeplot(accidents.LogWindSpeed, fill= True, ax=axs[1])
plt.show()
roadway_features  = ['Amenity','Bump','Crossing','GiveWay','Junction', 'NoExit', 'Railway', 'Roundabout', 'Station', 'Stop','TrafficCalming', 'TrafficSignal']
accidents["roadway_features"] = accidents[roadway_features].sum(axis = 1)
print(accidents[accidents.roadway_features.max(axis = 0) == accidents.roadway_features])

components = [ "Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
               "Superplasticizer", "CoarseAggregate", "FineAggregate"]
concrete["Components"] = concrete[components].gt(0).sum(axis=1)

print(concrete[components + ["Components"]].head(10))

customers[["Type", "Level"]] = (  # Create two new features
    customers["Policy"]           # from the Policy feature
    .str                         # through the string accessor
    .split(" ", expand=True)     # by splitting on " "                        # and expanding the result into separate columns
)

print(customers[["Policy", "Type", "Level"]].head(10))

autos["make_and_style"] = autos["make"] + "_" + autos["body_style"]
autos[["make", "body_style", "make_and_style"]].head()

print(customers.head())
customers["AverageIncome"] = (
    customers.groupby("State")  # for each state
    ["Income"]                 # select the income
    .transform("mean")         # and compute its mean
)

print(customers[["State", "Income", "AverageIncome"]].head(10))

customers["StateFreq"] = (
    customers.groupby("State")
    ["State"]
    .transform("count")
    / customers.State.count()
)
print(customers.State.count())
print(customers[["State", "StateFreq"]].head(10))

# Create splits
df_train = customers.sample(frac=0.5)
df_valid = customers.drop(df_train.index)

# Create the average claim amount by coverage type, on the training set
df_train["AverageClaim"] = df_train.groupby("Coverage")["ClaimAmount"].transform("mean")

# Merge the values into the validation set
df_valid = df_valid.merge(
    df_train[["Coverage", "AverageClaim"]].drop_duplicates(),
    on="Coverage",
    how="left",
)

df_valid[["Coverage", "AverageClaim"]].head(10)