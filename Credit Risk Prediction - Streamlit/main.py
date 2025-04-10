import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import RobustScaler

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
            df['emp_title'] = df['emp_title'].apply(
                lambda x: x.title() if isinstance(x, str) else 'Other')
            df['emp_title'] = df['emp_title'].apply(
                lambda x: x if x in self.top_emp_title else 'Other')

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
            return "Model not available. Please run training first."
        
        user_data = pd.DataFrame([form_input_dict])
        processed_data = self.preprocess(user_data)
        expected_cols = self.model.get_booster().feature_names
        processed_data = processed_data[expected_cols]
        prediction = self.model.predict(processed_data)[0]
        return "Risky Loan" if prediction == 0 else "Safe Loan"

# Streamlit App
st.set_page_config(page_title="Loan Risk Prediction", layout="wide")
st.title("Loan Risk Prediction Form")
st.write("Silakan isi form di bawah ini untuk memprediksi risiko pinjaman.")

PR = PredictRiskyLoan()

selected_columns = [
    'recoveries', 'total_rec_prncp', 'collection_recovery_fee', 'last_pymnt_month', 'last_pymnt_amnt',
    'last_pymnt_year', 'out_prncp', 'home_ownership', 'grade', 'initial_list_status',
    'verification_status', 'int_rate', 'sub_grade', 'emp_length', 'term', 'total_rec_int',
    'installment', 'last_credit_pull_year', 'last_credit_pull_month', 'total_rec_late_fee',
    'issue_d_year', 'inq_last_6mths', 'earliest_cr_line_month', 'issue_d_month', 'annual_inc',
    'addr_state', 'purpose', 'earliest_cr_line_year', 'tot_cur_bal', 'total_rev_hi_lim',
    'open_acc', 'pub_rec', 'emp_title', 'delinq_2yrs', 'revol_util', 'total_acc', 'revol_bal', 'dti'
]

dtype_map = {
    'recoveries': float, 'total_rec_prncp': float, 'collection_recovery_fee': float,
    'last_pymnt_month': float, 'last_pymnt_amnt': float, 'last_pymnt_year': float,
    'out_prncp': float, 'int_rate': float, 'total_rec_int': float,
    'installment': float, 'last_credit_pull_year': float, 'last_credit_pull_month': float,
    'total_rec_late_fee': float, 'issue_d_year': float, 'inq_last_6mths': float,
    'earliest_cr_line_month': float, 'issue_d_month': float, 'annual_inc': float,
    'tot_cur_bal': float, 'total_rev_hi_lim': float, 'open_acc': float,
    'pub_rec': float, 'delinq_2yrs': float, 'revol_util': float, 'total_acc': float,
    'revol_bal': float, 'dti': float,
    'home_ownership': str, 'grade': str, 'initial_list_status': str, 'verification_status': str,
    'sub_grade': str, 'emp_length': str, 'term': str, 'addr_state': str,
    'purpose': str, 'emp_title': str
}

form_input = {}

with st.form("prediction_form"):
    for col in selected_columns:
        expected_type = dtype_map[col]
        if expected_type == float:
            form_input[col] = st.text_input(f"{col}", "")
        else:
            form_input[col] = st.text_input(f"{col}", "").upper()

    submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            processed_input = {}
            for col in selected_columns:
                val = form_input[col].strip()
                expected_type = dtype_map[col]
                if val == "":
                    processed_input[col] = PR.imputation_values.get(col, np.nan)
                else:
                    try:
                        if expected_type == float:
                            processed_input[col] = float(val)
                        elif expected_type == int:
                            processed_input[col] = int(float(val))
                        else:
                            processed_input[col] = val.upper()
                    except Exception:
                        processed_input[col] = PR.imputation_values.get(col, np.nan)

            pred = PR.predict(processed_input)
            st.success(f"Prediction Result: **{pred}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
