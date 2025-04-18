import streamlit as st
import pandas as pd
import pickle

st.title("Machine Learning App - Loan")
st.markdown("Will you receive a loan or not?")

@st.cache_data
def load_data():
    df = pd.read_csv("Dataset_A_loan.csv")
    return df

df = load_data()

with st.expander("Data", expanded=True):
    st.markdown("This is the Loan Data")
    st.dataframe(df)

@st.cache_resource
def load_artifacts():
    with open("xgb_loan_model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("UTS_label_encoder.pkl", "rb") as file:
        label_encoders = pickle.load(file)
        
    with open("UTS_robust_scaler.pkl", "rb") as file:
        robust_scaler = pickle.load(file)
        
    return model, label_encoders, robust_scaler

cat_col = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']
num_col = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 
            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']


def preprocess_user_input(user_input, robust_scaler, label_encoders):

    df = pd.DataFrame([user_input])
    
    for col in cat_col:
        df[col] = label_encoders[col].transform(df[col])

    df[num_col] = robust_scaler.transform(df[num_col])
        
    return df

model, label_encoders, robust_scaler = load_artifacts()

st.markdown("Input Data below:")
col1, col2 = st.columns(2)

with col1:
    person_age = st.slider("Age", 10, 100, 30)
    person_gender = st.selectbox("Gender", ["male", "female"])
    person_education = st.selectbox("Education", ["Bachelor", "Associate", "High School", "Master", "Doctorate"])
    person_income = st.number_input("Income", min_value=0, max_value=5000000, step=1000)
    person_emp_exp = st.number_input("Work experience (in year)", min_value=0, max_value=80, step=1)
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    loan_amnt = st.number_input("Loan amount", min_value=500, max_value=50000, step=500)

with col2:
    loan_intent = st.selectbox("Loan Intention", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
    loan_int_rate = st.number_input("Interest rate", min_value=5.0, max_value=20.0, step=0.1)
    loan_percent_income = st.number_input("Loan percentage of Income", min_value=0.00, max_value=1.00, step=0.01)
    cb_person_cred_hist_length = st.number_input("Credit duration (in years)", min_value=1, max_value=30, step=1)
    credit_score = st.number_input("Credit Score", min_value=100, max_value=1000, step=10)
    previous_loan_defaults_on_file = st.selectbox("Previous Loan (Accepted or Not)", ["Yes", "No"])

user_input = {
    "person_age": person_age,
    "person_gender": person_gender,
    "person_education": person_education,
    "person_income": person_income,
    "person_emp_exp": person_emp_exp,
    "person_home_ownership": person_home_ownership,
    "loan_amnt": loan_amnt,
    "loan_intent": loan_intent,
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    "credit_score": credit_score,
    "previous_loan_defaults_on_file": previous_loan_defaults_on_file
}

if st.button("Predict your chance to receive a loan"):
    x_user = preprocess_user_input(user_input, robust_scaler, label_encoders)
    proba = model.predict_proba(x_user)[0]
    prediction = model.predict(x_user)[0]

    st.subheader("Data input by user")
    st.dataframe(pd.DataFrame([user_input]))

    st.subheader("Receive Loan Probabilities")
    proba_df = pd.DataFrame([proba], columns=["Loan will not be granted (0)", "Loan will be granted (1)"])
    st.dataframe(proba_df.style.format("{:.4f}"))
    
    if prediction == 1:
        st.write("Loan will be granted.")  # Menggunakan st.write() untuk menampilkan hasil di Streamlit
    else:
        st.write("Loan will not be granted.")










    
