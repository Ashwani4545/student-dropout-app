import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -------------------------------
# Dummy Dataset & Model Setup
# -------------------------------

# Example Dataset Columns
feature_columns = [
    'age', 'gender', 'parent_education', 'socio_economic_status',
    'attendance_rate', 'grades_avg', 'family_support',
    'distance_school_km', 'study_hours'
]

# Dummy Label Encoders for categorical data
gender_encoder = LabelEncoder()
gender_encoder.classes_ = np.array(['F', 'M'])

parent_edu_encoder = LabelEncoder()
parent_edu_encoder.classes_ = np.array(['High School', 'College', 'No Formal Education'])

ses_encoder = LabelEncoder()
ses_encoder.classes_ = np.array(['Low', 'Medium', 'High'])

family_support_encoder = LabelEncoder()
family_support_encoder.classes_ = np.array(['No', 'Yes'])

# Dummy StandardScaler (normally, you fit it on training data)
scaler = StandardScaler()
scaler.mean_ = np.zeros(len(feature_columns))
scaler.scale_ = np.ones(len(feature_columns))

# Dummy trained Decision Tree Model
model = DecisionTreeClassifier(random_state=42)
# For demonstration, fit the model on dummy data
X_dummy = np.random.rand(100, len(feature_columns))
y_dummy = np.random.randint(0, 2, 100)
model.fit(X_dummy, y_dummy)

# -------------------------------
# Streamlit Interface
# -------------------------------

st.title("🌳 Student Dropout Prediction System")

# User Input
age = st.slider('Age', 10, 20, 15)
gender = st.selectbox('Gender', ['M', 'F'])
parent_education = st.selectbox('Parent Education', ['High School', 'College', 'No Formal Education'])
socio_economic_status = st.selectbox('Socio-Economic Status', ['Low', 'Medium', 'High'])
attendance_rate = st.slider('Attendance Rate (%)', 0, 100, 80)
grades_avg = st.slider('Average Grades (0-100)', 0, 100, 70)
family_support = st.selectbox('Family Support', ['Yes', 'No'])
distance_school_km = st.slider('Distance from School (km)', 0, 20, 5)
study_hours = st.slider('Study Hours per Day', 0, 10, 2)

if st.button('Predict Dropout Risk'):
    # Preprocess Inputs
    input_data = pd.DataFrame([{
        'age': age,
        'gender': gender_encoder.transform([gender])[0],
        'parent_education': parent_edu_encoder.transform([parent_education])[0],
        'socio_economic_status': ses_encoder.transform([socio_economic_status])[0],
        'attendance_rate': attendance_rate / 100,  # Scale between 0–1
        'grades_avg': grades_avg / 100,
        'family_support': family_support_encoder.transform([family_support])[0],
        'distance_school_km': distance_school_km,
        'study_hours': study_hours
    }])

    # Apply scaling (for demonstration using identity scaler)
    input_data_scaled = scaler.transform(input_data)

    # Predict class and probability
    predicted_class = model.predict(input_data_scaled)[0]
    predicted_prob = model.predict_proba(input_data_scaled)[0][1]

    # Feature importance explanation
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False).head(3)

    # Show results
    st.success(f"🔔 Dropout Prediction: {'Yes' if predicted_class == 1 else 'No'}")
    st.info(f"📊 Dropout Probability: {predicted_prob * 100:.2f}%")

    st.subheader("Top 3 Features Contributing to Dropout Risk:")
    st.table(importance_df)

    st.write("---")
    st.write("⚡ *Note: Model is trained on dummy data. Replace with real data/model for production use.*")
