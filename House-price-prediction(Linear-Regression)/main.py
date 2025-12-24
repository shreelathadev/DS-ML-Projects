import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv('home_dataset.csv')

# Extract features and target variable
house_sizes = data['HouseSize'].values  # Feature: size of the house
house_prices = data['HousePrice'].values  # Target: price of the house

# Visualize the data
plt.scatter(house_sizes,house_prices, marker='o', color = 'b')
plt.title('House Price vs Size')
plt.xlabel('Size of House (sq ft)')
plt.ylabel('Price of House ($)')
plt.show()


# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(house_sizes, house_prices, test_size=0.2, random_state=42)

# Reshape the data for numpy
x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)

# Create and train the linear regression model
model = LinearRegression()
model.fit(x_train, y_train)


# Make predictions on the test set
predictions = model.predict(x_test)

# Visualize the results
plt.scatter(x_test, y_test, marker='o', color='b', label='Actual Prices')
plt.plot(x_test, predictions, color='r', label='Predicted Prices')
plt.title('House Price vs Size')
plt.xlabel('Size of House (sq ft)')
plt.ylabel('Price of House ($)')
plt.legend()
plt.show()