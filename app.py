import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set full layout configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ==========================================
# 1. LOAD TRAINED ASSETS
# ==========================================
@st.cache_resource
def load_assets():
    """Loads the model, scaler, and exact training column order."""
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        columns = joblib.load('columns.pkl')
        return model, scaler, columns
    except FileNotFoundError as e:
        st.error(f"Asset loading error: {e}. Ensure 'best_model.pkl', 'scaler.pkl', and 'columns.pkl' are in the working directory.")
        st.stop()

model, scaler, columns = load_assets()

# ==========================================
# 2. USER INTERFACE LAYOUT
# ==========================================
st.title('❤️ Heart Disease Prediction App')
st.write('Provide the patient clinical metrics below to evaluate risk probability.')

# Use a form to batch inputs together so the page doesn't refresh on every single slider click
with st.form("patient_data_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider('Age', 18, 100, 50)
        resting_bp = st.slider('Resting Blood Pressure (RestingBP in mm Hg)', 80, 200, 120)
        cholesterol = st.slider('Cholesterol (in mg/dl)', 100, 600, 200)
        fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (FastingBS)', options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
        max_hr = st.slider('Maximum Heart Rate Achieved (MaxHR)', 60, 210, 150)
        oldpeak = st.slider('Oldpeak (ST depression induced by exercise)', 0.0, 7.0, 1.0, step=0.1)

    with col2:
        sex = st.selectbox('Sex', ['Male', 'Female'])
        chest_pain_type = st.selectbox('Chest Pain Type (ChestPainType)', ['ATA', 'NAP', 'ASY', 'TA'])
        resting_ecg = st.selectbox('Resting Electrocardiogram (RestingECG)', ['Normal', 'ST', 'LVH'])
        exercise_angina = st.selectbox('Exercise Induced Angina (ExerciseAngina)', ['No', 'Yes'])
        st_slope = st.selectbox('ST Slope (Peak exercise ST segment)', ['Up', 'Flat', 'Down'])
        
    submit_button = st.form_submit_button(label='Predict Heart Disease')

# ==========================================
# 3. DATA PROCESSING & PREDICTION
# ==========================================
if submit_button:
    # Build raw dictionary replicating the precise state before one-hot encoding in training
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex': 'M' if sex == 'Male' else 'F',
        'ChestPainType': chest_pain_type,
        'RestingECG': resting_ecg,
        'ExerciseAngina': 'Y' if exercise_angina == 'Yes' else 'N',
        'ST_Slope': st_slope
    }
    
    # Transform to DataFrame
    input_df = pd.DataFrame([raw_input])
    
    # Replicate categorical dummy creation dynamically
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    input_df = pd.get_dummies(input_df, columns=categorical_cols)
    
    # Align feature space perfectly with columns.pkl order
    # reindex drops unneeded features and dynamically injects 0 for missing structural categories
    input_df = input_df.reindex(columns=columns, fill_value=0)
    
    # Robust scaling setup 
    try:
        processed_data = scaler.transform(input_df)
    except Exception as e:
        st.error(f"Scaling Alignment Failed: {e}. Verify that your scaler was trained on the matched columns schema.")
        st.stop()
        
    # Execute model logic
    prediction = model.predict(processed_data)
    prediction_proba = model.predict_proba(processed_data)[0]

    # ==========================================
    # 4. RESULTS VISUALIZATION
    # ==========================================
    st.markdown("---")
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.subheader('Prediction Result:')
        if prediction[0] == 1:
            st.error('⚠️ The model predicts a HIGH risk of Heart Disease.')
        else:
            st.success('✅ The model predicts a LOW risk of Heart Disease.')
            
    with res_col2:
        st.subheader('Confidence Metric:')
        risk_percentage = prediction_proba[1] * 100
        st.metric(label="Calculated Disease Risk Probability", value=f"{risk_percentage:.2f}%")
        st.progress(int(risk_percentage))
        
    st.markdown("<br><small>**Disclaimer:** This application serves strictly as a data-driven prediction tool based on machine learning. It does not replace clinical evaluation or certified healthcare counsel.</small>", unsafe_allow_html=True)