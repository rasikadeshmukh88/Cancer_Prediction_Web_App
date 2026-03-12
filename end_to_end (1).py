import streamlit as st
import numpy as np
import joblib

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Brain Tumor AI Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("random_forest_classifier.joblib")

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>

/* Background */
.main {
    background: linear-gradient(135deg, #e3f2fd, #f9fbff);
}

/* Header */
.header {
    background: linear-gradient(90deg, #1565c0, #42a5f5);
    padding: 25px;
    border-radius: 15px;
    color: white;
    font-size: 38px;
    font-weight: bold;
    text-align: center;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.15);
}

/* Subheader */
.subheader {
    text-align: center;
    font-size: 18px;
    color: #e3f2fd;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.9);
    backdrop-filter: blur(10px);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.12);
    margin-bottom: 20px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #1e88e5, #42a5f5);
    color: white;
    border-radius: 12px;
    padding: 12px 25px;
    font-size: 18px;
    font-weight: bold;
    border: none;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #0d47a1, #1976d2);
}

/* Results */
.good {
    background: linear-gradient(90deg, #c8e6c9, #a5d6a7);
    padding: 18px;
    border-radius: 12px;
    font-size: 22px;
    color: #1b5e20;
    font-weight: bold;
}

.bad {
    background: linear-gradient(90deg, #ffcdd2, #ef9a9a);
    padding: 18px;
    border-radius: 12px;
    font-size: 22px;
    color: #b71c1c;
    font-weight: bold;
}

/* Sidebar */
.css-1d391kg {
    background: linear-gradient(180deg, #0d47a1, #1565c0);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #
st.markdown("""
<div class="header">
🧠 Brain Tumor Cancer Prediction Dashboard
<div class="subheader">AI-Driven Medical Decision Support System</div>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("🩺 Patient Details")

age = st.sidebar.slider("Age", 0, 120, 30)
gender = st.sidebar.radio("Gender", ["Female", "Male"])
tumor_size = st.sidebar.slider("Tumor Size (cm)", 0.0, 10.0, 2.5)

location = st.sidebar.selectbox(
    "Tumor Location",
    ["Frontal", "Occipital", "Parietal", "Temporal"]
)

histology = st.sidebar.selectbox(
    "Histology Type",
    ["Astrocytoma", "Glioblastoma", "Medulloblastoma", "Meningioma"]
)

stage = st.sidebar.selectbox(
    "Cancer Stage",
    ["Stage 0", "Stage 1", "Stage 2", "Stage 3", "Stage 4"]
)

st.sidebar.subheader("⚠️ Symptoms")
symptom_1 = st.sidebar.selectbox("Symptom 1", ["None", "Mild", "Moderate", "Severe"])
symptom_2 = st.sidebar.selectbox("Symptom 2", ["None", "Mild", "Moderate", "Severe"])
symptom_3 = st.sidebar.selectbox("Symptom 3", ["None", "Mild", "Moderate", "Severe"])

st.sidebar.subheader("💊 Treatment")
radiation = st.sidebar.radio("Radiation Therapy", ["No", "Yes"])
surgery = st.sidebar.radio("Surgery", ["No", "Yes"])
chemo = st.sidebar.radio("Chemotherapy", ["No", "Yes"])

st.sidebar.subheader("📊 Reports")
survival_rate = st.sidebar.slider("Survival Rate (%)", 0, 100, 75)
tumor_growth_rate = st.sidebar.slider("Tumor Growth Rate", 0.0, 5.0, 1.2)
family_history = st.sidebar.radio("Family History", ["No", "Yes"])
mri_result = st.sidebar.radio("MRI Result", ["Normal", "Abnormal"])
follow_up = st.sidebar.radio("Follow-up Required", ["No", "Yes"])

# ---------------- CONVERSIONS ---------------- #
gender_val = 0 if gender == "Female" else 1
location_dict = {"Frontal": 0, "Occipital": 1, "Parietal": 2, "Temporal": 3}
histology_dict = {"Astrocytoma": 0, "Glioblastoma": 1, "Medulloblastoma": 2, "Meningioma": 3}
stage_dict = {"Stage 0": 0, "Stage 1": 1, "Stage 2": 2, "Stage 3": 3, "Stage 4": 4}
symptom_dict = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
yes_no = {"No": 0, "Yes": 1}
mri_dict = {"Normal": 0, "Abnormal": 1}

# ---------------- MAIN DASHBOARD ---------------- #
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 AI Prediction Engine")

    if st.button("🚀 Analyze Patient Data"):
        with st.spinner("Analyzing medical data..."):
            input_data = np.array([[
                age,
                gender_val,
                tumor_size,
                location_dict[location],
                histology_dict[histology],
                stage_dict[stage],
                symptom_dict[symptom_1],
                symptom_dict[symptom_2],
                symptom_dict[symptom_3],
                yes_no[radiation],
                yes_no[surgery],
                yes_no[chemo],
                survival_rate,
                tumor_growth_rate,
                yes_no[family_history],
                mri_dict[mri_result],
                yes_no[follow_up]
            ]])

            prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.markdown('<div class="bad">🔴 Malignant Brain Tumor Detected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="good">🟢 Benign Brain Tumor Detected</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📌 System Highlights")
    st.write("""
    ✔ Machine Learning powered  
    ✔ Random Forest Algorithm  
    ✔ Real-time prediction  
    ✔ Clinical decision support  
    ✔ Academic & research use  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.caption("© 2026 Brain Tumor AI Dashboard | Designed with ❤️ using Streamlit")