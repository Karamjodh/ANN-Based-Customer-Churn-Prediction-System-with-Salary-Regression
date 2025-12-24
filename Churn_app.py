# Load all the libraries required
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('Models/Churn_Classifier/model.h5')

# Load all the encoder as scalers
with open("Pickles/Churn_Classifier/StandardScaler.pkl","rb") as f:
    scaler = pickle.load(f)
with open("Pickles/Churn_Classifier/Geo_encoder.pkl","rb") as f:
    Geo_encoder = pickle.load(f)
with open("Pickles/Churn_Classifier/label_encoder_gender.pkl","rb") as f:
    label_encoder= pickle.load(f)

# Streamlit App
st.title("Customer Churn Prediction")

# Input Fields
geography = st.selectbox('Geography',Geo_encoder.categories_[0])
gender = st.selectbox('Gender',label_encoder.classes_)
age = st.slider('Age',18,92)
balance = st.number_input("Balance")
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input("Estimated Salary")
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
    'EstimatedSalary':estimated_salary
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
prediction = model.predict(input)[0][0]
prediction

# Streamlit Output
st.write(f"Churn Probability: {prediction:.2f}")
if prediction > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is unlikely to churn.")