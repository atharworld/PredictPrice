import pandas
from sklearn import linear_model as lm

df = pandas.read_csv('/storage/emulated/0/Download/fuel1.csv')

X = df[['Month']]
y = df['Price']

reg = lm.LinearRegression()
reg.fit(X,y)

predPrice = reg.predict([[0]])
print('Price in July: ',predPrice)