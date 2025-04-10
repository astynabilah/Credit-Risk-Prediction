import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import RobustScaler

# --- Load and Predict Class ---
class PredictRiskyLoan:
    def __init__(self):
        self.model = self.load_pickle("model_xgboost_ring1.pkl")
        self.imputation_values = self.load_pickle("imputation_values.pkl")
        self.label_encoders = self.load_pickle("label_encoders.pkl")
        self.category_order = self.load_pickle("category_orders.pkl")
        self.top_emp_title = self.load_pickle("top_categories_emp_title.pkl")
        self.winsor_bounds = self.load_pickle("winsor_bounds.pkl")
        self.scaler = self.load_pickle("scaler.pkl")

    def load_pickle(self, filename):
        if os.path.exists(filename):
            with open(filename, "rb") as file:
                return pickle.load(file)
        else:
            st.warning(f"File {filename} not found!")
            return None

    def preprocess(self, df):
        for col, val in self.imputation_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)

        if 'emp_title' in df.columns:
            df['emp_title'] = df['emp_title'].apply(lambda x: x.title() if isinstance(x, str) else 'Other')
            df['emp_title'] = df['emp_title'].apply(lambda x: x if x in self.top_emp_title else 'Other')

        for col, order in self.category_order.items():
            if col in df.columns:
                df[col] = df[col].astype(pd.CategoricalDtype(categories=order, ordered=True))
                df[col] = df[col].cat.codes

        for col, le in self.label_encoders.items():
            if col in df.columns and col not in self.category_order:
                try:
                    df[col] = le.transform(df[col])
                except Exception:
                    df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        for col in df.columns:
            if col in self.winsor_bounds['lower'] and col in self.winsor_bounds['upper']:
                lower = self.winsor_bounds['lower'][col]
                upper = self.winsor_bounds['upper'][col]
                df[col] = df[col].clip(lower, upper)

        scalable_cols = [col for col in df.columns if col in self.scaler.feature_names_in_]
        indices = [np.where(self.scaler.feature_names_in_ == col)[0][0] for col in scalable_cols]

        sub_scaler = RobustScaler()
        sub_scaler.center_ = self.scaler.center_[indices]
        sub_scaler.scale_ = self.scaler.scale_[indices]
        sub_scaler.n_features_in_ = len(scalable_cols)
        sub_scaler.feature_names_in_ = np.array(scalable_cols)

        scaled_array = sub_scaler.transform(df[scalable_cols].values)
        df.loc[:, scalable_cols] = scaled_array
        df = df.apply(pd.to_numeric, errors='raise')
        return df

    def predict(self, form_input_dict):
        if self.model is None:
            return "Model not available."
        user_data = pd.DataFrame([form_input_dict])
        processed_data = self.preprocess(user_data)
        expected_cols = self.model.get_booster().feature_names
        processed_data = processed_data[expected_cols]
        prediction = self.model.predict(processed_data)[0]
        return "Risky Loan" if prediction == 0 else "Safe Loan"

# === Streamlit App ===
st.set_page_config(page_title="Loan Risk Prediction", layout="wide")
st.markdown("""
<h1 style='text-align: center;'>Loan Risk Prediction Form</h1>

<p style='text-align: center; font-size: 16px;'>
This form is part of a project-based internship for the Data Scientist at ID/X Partners.<br>
Developed by <strong>Asty Nabilah 'Izzaturrahmah</strong><br/>
<a href="https://github.com/astynabilah/Credit-Risk-Prediction" target="_blank">View on GitHub</a><br/>
<a href="https://www.linkedin.com/in/asty-nabilah-izzaturrahmah/" target="_blank">Connect with me on Linkedin</a>
</p>

<hr style='margin-top: 10px; margin-bottom: 30px;'>

<div style='font-size:15px;'>
<h2>Instructions:</h2>
<ul>
  <li>Please fill in all relevant fields before submitting to ensure accurate predictions.</li>
  <li>Do not press <i>Enter</i> until all required fields have been filled in. Premature submission may lead to inaccurate results.</li>
  <li>For categorical fields, use the dropdown or input options based on the provided examples.</li>
  <li>If a field is not applicable or the information is unavailable, you may leave it blank.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Styling
st.markdown("""
<style>
    .form-text {
        font-size: 0.85rem;
        color: #888;
        margin-top: -8px;
        margin-bottom: 15px;
        display: block;
    }
    input[type="text"] {
        padding: 8px 10px;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# Load Model
PR = PredictRiskyLoan()

# Descriptions
descriptions = {
    'emp_title': "Your current job title or role.",
    'emp_length': "How long you’ve been working (in years).",
    'annual_inc': "Your total annual income before tax.",
    'addr_state': "The state where you currently live.",
    'inq_last_6mths': "Number of times your credit report was checked in the past 6 months.",
    'earliest_cr_line_year': "The year you first opened a credit line.",
    'earliest_cr_line_month': "The month you first opened a credit line.",
    'open_acc': "Total number of active credit accounts you currently have.",
    'pub_rec': "Number of public financial records (e.g., bankruptcies).",
    'delinq_2yrs': "How many times you've been over 30 days late on a payment in the past 2 years.",
    'total_acc': "Total number of credit accounts you’ve had (open and closed).",
    'revol_util': "Percentage of revolving credit you’re currently using.",
    'revol_bal': "Your total outstanding balance on revolving credit (e.g., credit cards).",
    'tot_cur_bal': "Total current balance across all accounts.",
    'total_rev_hi_lim': "Combined maximum limit on all your revolving credit accounts.",
    'grade': "Overall credit grade assigned to your loan.",
    'sub_grade': "More detailed breakdown of your loan’s credit grade.",
    'term': "Repayment term length in months (e.g., 36 or 60 months).",
    'home_ownership': "Your current home ownership status (e.g., rent, own, mortgage).",
    'purpose': "Reason for applying for the loan.",
    'int_rate': "The interest rate you’ll pay on the loan.",
    'dti': "Your debt-to-income ratio (monthly debt divided by monthly income).",
    'installment': "Monthly payment amount you'll make for this loan.",
    'issue_d_year': "The year the loan was issued.",
    'issue_d_month': "The month the loan was issued.",
    'initial_list_status': "Initial offering type of the loan (e.g., whole or fractional).",
    'verification_status': "Whether your income was verified by the lender.",
    'last_pymnt_year': "The year you made your most recent payment.",
    'last_pymnt_month': "The month you made your most recent payment.",
    'last_pymnt_amnt': "The amount you paid during your last payment.",
    'recoveries': "Amount recovered after the loan was charged off.",
    'collection_recovery_fee': "Fee charged for recovering a defaulted loan.",
    'total_rec_prncp': "Total principal amount you've already paid back.",
    'total_rec_int': "Total interest amount you've already paid.",
    'total_rec_late_fee': "Total amount paid in late fees so far.",
    'last_credit_pull_year': "The year when your credit was last checked.",
    'last_credit_pull_month': "The month when your credit was last checked.",
    'out_prncp': "Remaining loan principal that you still owe."
}


# Section structure
sections = {
    "1. Borrower Information": [
        'emp_title', 'emp_length', 'annual_inc', 'addr_state'
    ],
    "2. Credit History": [
        'inq_last_6mths', 'earliest_cr_line_year', 'earliest_cr_line_month',
        'open_acc', 'pub_rec', 'delinq_2yrs', 'total_acc',
        'revol_util', 'revol_bal', 'tot_cur_bal', 'total_rev_hi_lim'
    ],
    "3. Loan Details": [
        'grade', 'sub_grade', 'term', 'home_ownership', 'purpose',
        'int_rate', 'dti', 'installment', 'issue_d_year', 'issue_d_month',
        'initial_list_status', 'verification_status'
    ],
    "4. Repayment": [
        'last_pymnt_year', 'last_pymnt_month', 'last_pymnt_amnt',
        'recoveries', 'collection_recovery_fee', 'total_rec_prncp',
        'total_rec_int', 'total_rec_late_fee', 'last_credit_pull_year',
        'last_credit_pull_month', 'out_prncp'
    ]
}

# Categorical options
categorical_options = {
    'term': ['36 months', '60 months'],
    'grade': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    'sub_grade': ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5',
                  'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5',
                  'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5',
                  'G1', 'G2', 'G3', 'G4', 'G5'],
    'emp_title': ['Teacher', 'Manager', 'Registered Nurse', 'RN', 'Supervisor', 'Sales',
                  'Project Manager', 'Owner', 'Office Manager', 'manager', 'Driver',
                  'General Manager', 'Director', 'teacher', 'Engineer', 'driver',
                  'Vice President', 'President', 'owner', 'Administrative Assistant',
                  'Operations Manager', 'Attorney', 'Accountant', 'supervisor',
                  'Police Officer', 'sales', 'Sales Manager', 'Account Manager',
                  'Store Manager', 'Executive Assistant', 'truck driver', 'US Army',
                  'Analyst', 'Technician', 'Nurse', 'Software Engineer', 'Truck Driver',
                  'Assistant Manager', 'Paralegal', 'Controller', 'Program Manager',
                  'Branch Manager', 'registered nurse', 'Consultant', 'Account Executive',
                  'Administrator', 'Bank of America', 'Business Analyst', 'Principal',
                  'Mechanic', 'Professor', 'Server', 'Executive Director', 'IT Manager',
                  'mechanic', 'Electrician', 'Registered nurse', 'CEO',
                  'Customer Service', 'Associate', 'AT&T', 'Foreman',
                  'Director of Operations', 'Secretary', 'Financial Analyst',
                  'Kaiser Permanente', 'Legal Assistant', 'District Manager', 'LPN',
                  'USAF', 'USPS', 'Superintendent', 'Pharmacist', 'Physician', 'UPS',
                  'Walmart', 'Financial Advisor', 'technician', 'Operator', 'nurse',
                  'Social Worker', 'Accounting Manager', 'Instructor', 'Clerk', 'Officer',
                  'MANAGER', 'Bookkeeper', 'machine operator', 'clerk', 'Machinist',
                  'Firefighter', 'Maintenance', 'CNA', 'Service Manager', 'Wells Fargo',
                  'Bartender', 'server', 'Truck driver', 'Legal Secretary',
                  'IT Specialist', 'Others'],
    'emp_length': ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years',
                   '6 years', '7 years', '8 years', '9 years', '10+ years'],
    'home_ownership': ['RENT', 'OWN', 'MORTGAGE', 'OTHER', 'NONE', 'ANY'],
    'verification_status': ['Verified', 'Source Verified', 'Not Verified'],
    'purpose': ['credit_card', 'car', 'small_business', 'other', 'wedding',
                'debt_consolidation', 'home_improvement', 'major_purchase',
                'medical', 'moving', 'vacation', 'house', 'renewable_energy',
                'educational'],
    'addr_state': ['AZ', 'GA', 'IL', 'CA', 'OR', 'NC', 'TX', 'VA', 'MO', 'CT',
                   'UT', 'FL', 'NY', 'PA', 'MN', 'NJ', 'KY', 'OH', 'SC', 'RI',
                   'LA', 'MA', 'WA', 'WI', 'AL', 'CO', 'KS', 'NV', 'AK', 'MD',
                   'WV', 'VT', 'MI', 'DC', 'SD', 'NH', 'AR', 'NM', 'MT', 'HI',
                   'WY', 'OK', 'DE', 'MS', 'TN', 'IA', 'NE', 'ID', 'IN', 'ME'],
    'initial_list_status': ['f', 'w']
}

dtype_map = {key: str for sec in sections.values() for key in sec}
for k in descriptions.keys():
    if k not in dtype_map:
        dtype_map[k] = float

form_input = {}

with st.form("prediction_form"):
    for section_name, columns in sections.items():
        st.subheader(section_name)
        st.markdown("---")
        for col in columns:
            label = col.replace('_', ' ').title()
            desc = descriptions.get(col, "No description available.")

            # Jika kolom termasuk kategori, gunakan selectbox (dropdown yang bisa diketik)
            if col in categorical_options:
                form_input[col] = st.selectbox(
                    f"{label}",
                    options=categorical_options[col],
                    key=col
                )
            # Jika float, gunakan text input
            elif dtype_map[col] == float:
                form_input[col] = st.text_input(f"{label}", "", key=col)
            # Sisanya tetap pakai input teks biasa (untuk string non-kategori)
            else:
                form_input[col] = st.text_input(f"{label}", "", key=col).strip().upper()

            st.markdown(f"<span class='form-text'>{desc}</span>", unsafe_allow_html=True)

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        processed_input = {}
        for col in dtype_map:
            val = form_input[col]
            if isinstance(val, str):
                val = val.strip()
            if val == "":
                processed_input[col] = PR.imputation_values.get(col, np.nan)
            else:
                try:
                    if dtype_map[col] == float:
                        processed_input[col] = float(val)
                    else:
                        processed_input[col] = val
                except Exception:
                    processed_input[col] = PR.imputation_values.get(col, np.nan)

        pred = PR.predict(processed_input)

        if pred == "Risky Loan":
            st.error(f"Prediction Result: **{pred}**")
        else:
            st.success(f"Prediction Result: **{pred}**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
