from sklearn.datasets import fetch_california_housing

cali = fetch_california_housing() #bunch object

print(cali.DESCR)

print(cali.data.shape)

print(cali.target.shape)

print(cali.feature_names)