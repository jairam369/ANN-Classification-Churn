## streamlit app
import streamlit as st
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


## Load the trained model, scalar pickle, onehot
model = tf.keras.models.load_model('./model.h5')

## Load the encoder and scaler
with open('./label_encoder_gender.pkl', 'rb') as file:
  label_encoder_gender = pickle.load(file)

with open('./onehot_encoder_geo.pkl', 'rb') as file:
  label_encoder_geo = pickle.load(file)

with open('./scaler.pkl', 'rb') as file:
  scaler = pickle.load(file)



st.title("Customer Churn prediction")



## User input
geography=st.selectbox('Geography', label_encoder_geo.categories_[0])
gender=st.selectbox('Gender', label_encoder_gender.classes_)
age=st.number_input('Age', 18, 92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
has_cr_card=st.selectbox('Has Credit Card', [0, 1])
is_active_member=st.selectbox('Is Active Member', [0, 1])
tenure=st.number_input('Tenure', min_value=0, max_value=10, value=5)
num_of_products=st.number_input('Number of Products', min_value=1, max_value=4)

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    # 'Geography': geography,
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

## Scale the input data
input_data_scaled = scaler.transform(input_data)

## Predict Churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Prediction Probability: {prediction_proba:.2f}')
if prediction_proba > 0.5:
  st.write('The customer is likely to churn')
else:
  st.write('The customer is not likely to churn')
