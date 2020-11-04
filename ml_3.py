from sklearn.datasets import fetch_california_housing

cali = fetch_california_housing() #bunch object

#print(cali.DESCR)

#print(cali.data.shape)

#print(cali.target.shape)

#print(cali.feature_names)

import pandas as pd
pd.set_option("precision",4)
pd.set_option("max_columns", 9)
pd.set_option("display.width", None)

cali_df = pd.DataFrame(cali.data, columns=cali.feature_names)

cali_df["MedHouseValue"] = pd.Series(cali.target)

#print(cali_df.head())

sample_df = cali_df.sample(frac=.1, random_state=17)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=2)
sns.set_style("whitegrid")

for feature in cali.feature_names:
    plt.figure(figsize=(8,4.5))
    sns.scatterplot(
        data=sample_df,
        x=feature,
        y="MedHouseValue",
        hue="MedHouseValue",
        legend=False,
    )
#plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
'''
split
LinearRegression
fit
compare the split to the regression
'''

x_train, x_test, y_train, y_test = train_test_split(
    cali.data, cali.target, random_state=11
    )

lr = LinearRegression()

lr.fit(X=x_train,y=y_train)

predicted = lr.predict(x_test)

expected = y_test

print(f"predicted:{predicted[::5]} expected: {expected[::5]}")

df = pd.DataFrame()

df["Expected"] = pd.Series(expected)
df["Predicted"] = pd.Series(predicted)

import matplotlib.pyplot as plt2

figure = plt.figure(figsize=(9,9))

axes = sns.scatterplot(
    data=df,
    x="Expected",
    y="Predicted",
    hue="Predicted",
    palette="cool",
    legend=False
)

start = min(expected.min(), predicted.min())
end = max(expected.max(), predicted.max())

print(start)
print(end)

axes.set_xlim(start,end)
axes.set_ylim(start,end)

line = plt2.plot([start,end], [start,end],"k--")

plt2.show()