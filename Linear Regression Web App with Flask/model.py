import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# Importing Data
df = pd.read_csv('speakers_prices.csv')

# Cleaning data
df.replace('$','', regex=True, inplace=True)
df.replace(',','', regex=True, inplace=True)

# Defining x variables
x = df.select_dtypes('number')
# Dropping first column, which will be the Y variable
x.drop(columns=x.columns[0], axis=1, inplace=True)

# Defining y variable as Col A
y = df.select_dtypes('number')
# Dropping all columns other than first column, which is the Y variable
y.drop(columns=y.columns[1:], axis=1, inplace=True)

# Creating linear regression model
model = LinearRegression()
model.fit(x, y)

# Saving model
pickle.dump(model, open('model.pkl', 'wb'))