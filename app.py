#gender 1-female 0-male
#churn 1-yes 0-no
#techsupport 1-yes 0-no
#scaler is exported as scaler.pkl
#model is exported as best_model.pkl
#order of x  = age, gender,tenure,monthlycharges,techsupport

import joblib
import numpy as np
import streamlit as st

scaler=joblib.load("scaler.pkl")
model=joblib.load("best_model.pkl")

st.title("Customer Churn prediction")
st.divider()
st.write("Customer Churn prediction using a trained machine learning model.")
st.divider()
age=st.number_input("Enter the age of the customer",min_value=18,max_value=100,value=30)
gender=st.selectbox("Select the gender",["Male","Female"])
tenure=st.number_input("Enter the tenure of the customer",min_value=0,max_value=100,value=10)
monthlycharges=st.number_input("Enter the monthly charges of the customer",min_value=0,max_value=150,value=50)
techsupport=st.selectbox("Select if the customer has tech support",["Yes","No"])
st.divider()
predictbutton = st.button("Predict churn")
if predictbutton:
    gender_selected=1 if gender=="Female" else 0
    techsupport_selected=1 if techsupport=="Yes" else 0
    X=[age,gender_selected,tenure,monthlycharges,techsupport_selected]
    X1=np.array(X)
    X_array=scaler.transform([X1])
    prediction=model.predict(X_array)[0]
    predicted_label="Yes" if prediction==1 else "No"
    st.write(f"The model predicts that the customer will churn: {predicted_label}")
else:
    st.write("Please fill in the details and press the 'Predict churn' button to get the prediction.")



    
