import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
# data = pd.read_csv(r"C:\vscode\coding\VSCODE\Upgrad\webml\housePricepred\Ames_Housing_Subset(in).csv")

st.title("House Price Prediction App")
st.write('Choose a machine learning model to predict house prices based on the Ames Housing Dataset.')

@st.cache_data
def load_data():
    data = pd.read_csv(r"C:\vscode\coding\VSCODE\Upgrad\webml\housePricepred\Ames_Housing_Subset(in).csv")
    return data

data = load_data()

st.subheader("Dataset Preview")
st.dataframe(data)

data = data.select_dtypes(include=[np.number])
X=data.drop('SalePrice', axis=1)
y=data['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_name = st.selectbox("Select a model", 
                          ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"])

if model_name == "Linear Regression":
    model = LinearRegression()
elif model_name == "Decision Tree Regressor":
    model = DecisionTreeRegressor()
else:
    model = RandomForestRegressor()    
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.subheader("Model Performance")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Input features for prediction
st.subheader("Predict House Price")
input_features = {}

for column in X.columns:
    input_features[column] = st.number_input(f"Enter {column}", 
                                             value=float(X[column].median())
                                             )

input_df = pd.DataFrame([input_features])

if st.button("Predict"):
    price = model.predict(input_df)[0]
    st.success(f"Predicted House Price: ${price:.3f}")