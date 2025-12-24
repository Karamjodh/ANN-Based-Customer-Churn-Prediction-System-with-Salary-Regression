# Load all the libraries required
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('Models/Salary_Regressor/model.h5',compile=False)
model2 = tf.keras.models.load_model('Models/Salary_Regressor_rf/model.h5',compile=False)

# Load all the encoder as scalers
with open("Pickles/Salary_Regressor/StandardScaler.pkl","rb") as f:
    scaler = pickle.load(f)
with open("Pickles/Salary_Regressor/Geo_encoder.pkl","rb") as f:
    Geo_encoder = pickle.load(f)
with open("Pickles/Salary_Regressor/label_encoder_gender.pkl","rb") as f:
    label_encoder= pickle.load(f)

# Streamlit App
st.title("Customer Salary Prediction")

# Input Fields
geography = st.selectbox('Geography',Geo_encoder.categories_[0])
gender = st.selectbox('Gender',label_encoder.classes_)
age = st.slider('Age',18,92)
balance = st.number_input("Balance")
credit_score = st.number_input('Credit Score')
exited = st.selectbox("Exited",["Yes","No"])
tenure = st.slider("Tenure",0,10)
num_of_products = st.slider("Number Of Products",1,4)
has_cr_card = st.selectbox("Has Credit Card",["Yes","No"])
is_active_member = st.selectbox("Is Active Member",["Yes","No"])

# Taking it as input
input_data={
    'CreditScore':credit_score,
    'Geography':geography,
    'Gender':gender,
    'Age':age,
    'Tenure':tenure,
    'Balance':balance,
    'NumOfProducts':num_of_products,
    'HasCrCard':1 if has_cr_card == "Yes" else 0,
    'IsActiveMember':1 if is_active_member == "Yes" else 0,
    'Exited':1 if exited == "Yes" else 0
}

# Encoding
Geo_encoded = Geo_encoder.transform([[geography]]).toarray()
Geo_encoded_df = pd.DataFrame(Geo_encoded,columns = Geo_encoder.get_feature_names_out(["Geography"]))
label_encoded = label_encoder.transform([gender])

# Creating a single input dataframe for prediction
input_data['Gender'] = label_encoded[0]
input = pd.DataFrame([input_data])
input.drop(['Geography'],axis=1,inplace = True)
input = pd.concat([input,Geo_encoded_df],axis=1)

input_df = input.astype(float)

# Scaling the input
input = scaler.transform(input)

# Prediciton
prediction = (model.predict(input)[0][0]+model2.predict(input)[0][0])/2
prediction

# Streamlit Output
st.write(f"Predicted Estimated Salary: {prediction:.2f}")
