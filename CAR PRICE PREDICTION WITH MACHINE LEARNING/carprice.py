import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load your car price prediction dataset from a CSV file
# Replace 'your_dataset.csv' with the actual file path to your dataset
data = pd.read_csv('C:\\Users\\hp\\OneDrive\\Desktop\\PYTHON\\car data.csv')

# Display the first few rows of the dataset to inspect the data
print(data.head())

# Data Preprocessing (customize this section based on your dataset)
# For example, you may need to encode categorical variables, handle missing values, etc.

# Split the dataset into features (X) and target variable (y)
X = data[['Year', 'Present_Price', 'Driven_kms']]
y = data['Selling_Price']

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared (R2) Score:", r2)
