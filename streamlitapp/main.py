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
st.title("Loan Risk Prediction Form")
st.markdown("Silakan isi form di bawah ini untuk memprediksi risiko pinjaman.")

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
    'emp_title': "Job title provided by borrower.",
    'emp_length': "Employment length in years (0 to 10+).",
    'annual_inc': "Self-reported annual income.",
    'addr_state': "US state provided by borrower in application.",
    'inq_last_6mths': "Number of credit inquiries in past 6 months.",
    'earliest_cr_line_year': "Year of borrower's earliest credit line.",
    'earliest_cr_line_month': "Month of borrower's earliest credit line.",
    'open_acc': "Number of open credit lines.",
    'pub_rec': "Number of derogatory public records.",
    'delinq_2yrs': "Number of 30+ days past-due incidents in last 2 years.",
    'total_acc': "Total number of credit lines.",
    'revol_util': "Revolving line utilization rate (% of credit used).",
    'revol_bal': "Total credit revolving balance.",
    'tot_cur_bal': "Total current balance of all accounts.",
    'total_rev_hi_lim': "Total revolving high credit/limit.",
    'grade': "Loan grade assigned by LendingClub.",
    'sub_grade': "Loan subgrade assigned by LendingClub.",
    'term': "The number of monthly payments (e.g., 36 or 60).",
    'home_ownership': "Home ownership status: RENT, OWN, MORTGAGE, OTHER.",
    'purpose': "Category for the loan request (e.g., debt_consolidation).",
    'int_rate': "Interest rate of the loan.",
    'dti': "Debt-to-income ratio.",
    'installment': "Monthly payment owed by the borrower.",
    'issue_d_year': "Year when the loan was funded.",
    'issue_d_month': "Month the loan was issued.",
    'initial_list_status': "Initial listing status: Whole, Fractional.",
    'verification_status': "Indicates if income was verified by LC.",
    'last_pymnt_year': "Year of the last payment received.",
    'last_pymnt_month': "Month of the last payment received.",
    'last_pymnt_amnt': "Last total payment amount received.",
    'recoveries': "Indicates if a payment plan has been put in place for the loan.",
    'collection_recovery_fee': "Post charge off collection fee.",
    'total_rec_prncp': "Principal received to date.",
    'total_rec_int': "Interest received to date.",
    'total_rec_late_fee': "Late fees received to date.",
    'last_credit_pull_year': "Year of the most recent credit pull.",
    'last_credit_pull_month': "Month of the most recent credit pull.",
    'out_prncp': "Remaining outstanding principal for total amount funded."
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
            if dtype_map[col] == float:
                form_input[col] = st.text_input(f"{label}", "")
            else:
                form_input[col] = st.text_input(f"{label}", "").upper()
            desc = descriptions.get(col, "No description available.")
            st.markdown(f"<span class='form-text'>{desc}</span>", unsafe_allow_html=True)

    submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            processed_input = {}
            for col in dtype_map:
                val = form_input[col].strip()
                if val == "":
                    processed_input[col] = PR.imputation_values.get(col, np.nan)
                else:
                    try:
                        processed_input[col] = float(val) if dtype_map[col] == float else val.upper()
                    except Exception:
                        processed_input[col] = PR.imputation_values.get(col, np.nan)

            pred = PR.predict(processed_input)
            st.success(f"Prediction Result: **{pred}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
