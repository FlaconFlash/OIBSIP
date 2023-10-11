import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv('G:\\New folder\\Advertising.csv')
print(data.head())
x= data[['TV', 'Radio', 'Newspaper']]  #features
y = data['Sales']  #target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Square Error: ", mse)
print("Mean Absolute Error: ", mae)
print("R-Squared (R2) Score: ", r2)

