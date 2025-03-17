from sklearn.datasets import fetch_california_housing
import pandas as pd

#Part 1
#Loading the Housing Dataset 
housing = fetch_california_housing()

#Pandas DataFrames for Features and Series fo rthe target variable
X = pd.DataFrame(housing.data, columns = housing.feature_names)
Y = pd.Series(housing.target, name='med_house_value')

#First 5 Rows of Feature Variable
print(X.head())

#First 5 Rows of Target Variable
print(Y.head())

#Printing the Feature Names & Missing Values Per Column
print(X.dtypes)
print(X.isnull().sum())
#Because the sum of the is null for all the features is 0, that means that there are no missing values in this data set

#Summary Statistics of Features
print(X.describe(percentiles=None, include=None, exclude=None))

#Part 2
#Splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Training the linear regression model on unscaled data
lin_reg_raw = LinearRegression()
lin_reg_raw.fit(X_train_raw, y_train)

print("Model Coefficients (Unscaled):")
print(pd.Series(lin_reg_raw.coef_,
                index=X.columns))
print("\nModel Intercept (Unscaled):")
print(pd.Series(lin_reg_raw.intercept_))

# Make predictions on the test set
y_pred_raw = lin_reg_raw.predict(X_test_raw)

# Model Performance Metrics
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score

mse_raw = mean_squared_error(y_test, y_pred_raw)
rmse_raw = root_mean_squared_error(y_test, y_pred_raw)
r2_raw = r2_score(y_test, y_pred_raw)

print("Unscaled Data Model:")
print(f"Mean Squared Error: {mse_raw:.2f}")
print(f"Root Mean Squared Error: {rmse_raw:.2f}")
print(f"R² Score: {r2_raw:.2f}")

# MSE = .56
# RMSE = .75
# R^2 Score = .58

#Part 3: Scaled Data 
from sklearn.preprocessing import StandardScaler

# Initialize the scaler and apply it to the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split the scaled data
X_train_scaled, X_test_scaled, _, _ = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model on scaled data
lin_reg_scaled = LinearRegression()
lin_reg_scaled.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_scaled = lin_reg_scaled.predict(X_test_scaled)

# Evaluate model performance
mse_scaled = mean_squared_error(y_test, y_pred_scaled)
r2_scaled = r2_score(y_test, y_pred_scaled)
rmse_scaled = root_mean_squared_error(y_test, y_pred_raw)

print("\nScaled Data Model:")
print(f"Mean Squared Error: {mse_scaled:.2f}")
print(f"Root Mean Squared Error: {rmse_scaled:.2f}")
print(f"R² Score: {r2_scaled:.2f}")
print("Model Coefficients (Scaled):")
print(pd.Series(lin_reg_scaled.coef_, index=X.columns))
#Part 4: Simplified Model 

#Features being used, median household income, average rooms, average bedrooms

#Selecting three features
Simple_X = X[['MedInc', 'AveRooms', 'AveBedrms']]

#Splitting the dataset into training and test sets
X_train_raw_S, X_test_raw_S, y_train_S, y_test_S = train_test_split(Simple_X, Y, test_size=0.2, random_state=42)

#Training the linear regression model on unscaled data
lin_reg_raw_S = LinearRegression()
lin_reg_raw_S.fit(X_train_raw_S, y_train_S)

print("Model Coefficients (Unscaled):")
print(pd.Series(lin_reg_raw_S.coef_,
                index=Simple_X.columns))
print("\nModel Intercept (Unscaled):")
print(pd.Series(lin_reg_raw_S.intercept_))

# Make predictions on the test set
y_pred_raw_S = lin_reg_raw_S.predict(X_test_raw_S)

# Model Performance Metrics
mse_raw_S = mean_squared_error(y_test_S, y_pred_raw_S)
rmse_raw_S = root_mean_squared_error(y_test_S, y_pred_raw_S)
r2_raw_S = r2_score(y_test_S, y_pred_raw_S)

print("Unscaled Data Model:")
print(f"Mean Squared Error: {mse_raw_S:.2f}")
print(f"Root Mean Squared Error: {rmse_raw_S:.2f}")
print(f"R² Score: {r2_raw_S:.2f}")

# MSE = .68
# RMSE = .82
# R^2 Score = .48











