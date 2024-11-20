
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
file_path = '/content/sample_data/Practice2_Chapter2.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Dataset preview:")
print(data.head())
from google.colab import drive
drive.mount('/content/drive')

# Check for missing values
print("\nMissing values:\n", data.isnull().sum())

# Assuming 'Sales' is the target variable and others are features
X = data.drop('Sales', axis=1)
y = data['Sales']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining set size:", X_train.shape)
print("Test set size:", X_test.shape)
# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Get model coefficients
print("\nModel coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)

# Predict sales for new data (example input)
new_data = pd.DataFrame({'TV': [200], 'Radio': [150], 'Newspaper': [100]})
predicted_sales = model.predict(new_data)

print("\nPredicted Sales for new data:", predicted_sales)
