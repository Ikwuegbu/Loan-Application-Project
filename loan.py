import pandas as pd
import streamlit as st
import joblib
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('Loan_Data.csv')

st.markdown("<h1 style = 'color: #114232; text-align: center; font-size: 60px; font-family: Monospace'>LOAN PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #87A922; text-align: center; font-family: cursive '>Built by Chibuzor</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)

# Add an image
st.image('pngwing.com (1).png', caption = 'Built by Chibuzor')   

#Add Project proble statement
st.markdown("<h2 style = 'color: #FF9800; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)

st.markdown("Loan prediction involves using historical data on loan applicants to develop predictive models that can assess the creditworthiness of future loan applicants. This typically includes analyzing factors such as income, coapplicant income, and other relevant financial information to determine the likelihood of a borrower repaying a loan. By studying these factors and their impact on loan approval and repayment, financial institutions can make more informed decisions when evaluating loan applications. This helps reduce the risk of default and ensures that loans are granted to individuals who are more likely to repay them.</p>", unsafe_allow_html=True)

# Sidebar design (to put what you want on the side)
st.sidebar.image('pngwing.com (2).png')

# markdown is for space
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.header('Project Data')
st.dataframe(data, use_container_width= True)

app_income = st.sidebar.number_input('Applicant Income', data['ApplicantIncome'].min(), data['ApplicantIncome'].max())
loan_amt = st.sidebar.number_input('Loan Amount', data['LoanAmount'].min(), data['LoanAmount'].max())
coapp_income = st.sidebar.number_input('CoApplicant Income', data['CoapplicantIncome'].min(), data['CoapplicantIncome'].max())
dep = st.sidebar.selectbox('Dependents', data['Dependents'].unique())
prop_area = st.sidebar.selectbox('Property Area', data['Property_Area'].unique())
cred_hist = st.sidebar.number_input('Credit History', data['Credit_History'].min(), data['Credit_History'].max())
loan_amt_term = st.sidebar.number_input('Loan Amount Term', data['Loan_Amount_Term'].min(), data['Loan_Amount_Term'].max())


#users input
input_var = pd.DataFrame()
input_var['ApplicantIncome'] = [app_income]
input_var['LoanAmount'] = [loan_amt]
input_var['CoapplicantIncome'] = [coapp_income]
input_var['Dependents'] = [dep]
input_var['Property_Area'] = [prop_area]
input_var['Credit_History'] = [cred_hist]
input_var['Loan_Amount_Term'] = [loan_amt_term]

st.markdown("<br>", unsafe_allow_html= True)
st.divider()
st.subheader('Users Inputs')
st.dataframe(input_var, use_container_width = True)


app_income = joblib.load('ApplicantIncome_scaler.pkl')
coapp_income = joblib.load('CoapplicantIncome_scaler.pkl')
dep = joblib.load('Dependents_encoder.pkl')
prop_area = joblib.load('Property_Area_encoder.pkl')



# transform the users input with the imported scalers
input_var['ApplicantIncome'] = app_income.transform(input_var[['ApplicantIncome']])
input_var['CoapplicantIncome'] = coapp_income.transform(input_var[['CoapplicantIncome']])
input_var['Property_Area'] = prop_area.transform(input_var[['Property_Area']])
input_var['Dependents'] = dep.transform(input_var[['Dependents']])





model = joblib.load('loanpredictionmodel.pkl')
prediction = model.predict(input_var)


if st.button('Check Your Loan Approval Status'):
    if prediction[0] == 0:
        st.error(f"Unfortunately...Your Loan request has been denied")
        
    else:
        st.success(f"Congratulations... Your loan request has been approved. Pls come to the office to process your loan")
        st.balloons()

