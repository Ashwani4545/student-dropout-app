# 🌳 AI System to Predict and Explain School Dropouts

## 🎯 Project Overview

This project is designed to build an **Artificial Intelligence (Machine Learning) system** that predicts whether a student is likely to drop out of school based on various factors such as demographics, academic performance, socio-economic status, and attendance.  
The system also explains the key reasons contributing to the dropout risk, helping school administrators take informed actions.

---

## ✅ Problem Statement

> Design a machine learning system that predicts student dropout risk and provides interpretable reasons behind the risk using structured data.

---

## ⚙️ System Inputs

The system uses the following input features:

| Feature               | Description                              |
| --------------------- | ---------------------------------------- |
| Age                   | Age of the student                       |
| Gender                | Male / Female                            |
| Parent’s Education    | Highest education level of parent        |
| Socio-Economic Status | Low / Medium / High                      |
| Attendance Rate       | Percentage of attendance                 |
| Academic Grades       | Average marks in previous academic years |
| Family Support        | Yes / No                                 |
| Distance from School  | Distance in kilometers                   |
| Study Hours           | Daily average study time in hours        |

**Target Variable**:

- Dropout Label (Yes / No)

---

## ✅ System Outputs

- **Dropout Prediction**:  
    • 0 → No Dropout  
    • 1 → Dropout

- **Dropout Probability** (e.g., 85% probability of dropout)

- **Top Reasons for Dropout**:  
    • Key factors (based on feature importance) contributing to the risk

---

## 📊 Approach Overview

1. Data Collection (Structured dataset in CSV format)
2. Data Preprocessing  
     • Handling missing values  
     • Encoding categorical variables  
     • Scaling numerical features
3. Model Selection  
     • Decision Tree Classifier  
     • Logistic Regression  
     • Random Forest Classifier
4. Model Training & Evaluation  
     • Accuracy, Precision, Recall, F1-Score
5. Feature Importance Extraction
6. Report / Dashboard Creation (using Streamlit)
7. Optional Deployment for real-time usage

---

## 🎯 Success Criteria

- Model accuracy ≥ 80%
- Clear explanation of top dropout reasons
- Simple, user-friendly report/dashboard
- Usable by school administrators for informed decision-making

---

## 🚀 Future Improvements

- Extend to support larger datasets
- Add more advanced features (psychological data, family income, etc.)
- Build a full web-based real-time prediction dashboard
- Support multiple languages for regional usage

---

## 📚 Tools & Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn (for visualizations)
- Streamlit (for dashboard)
- Jupyter Notebook

---

## 📄 Dataset Source

A sample dataset can be found here:  
👉 [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance)

You can also use any structured dataset in CSV format with similar features.

---

## **Installation (Local)**

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/student-dropout-app.git
cd student-dropout-app
```


## ✅ How to Run the System

1. Install required Python packages:

   ```bash
   pip install pandas scikit-learn matplotlib seaborn streamlit
   ```

2. Load the dataset and run the preprocessing script:

   ```bash
   python data_preprocessing.py
   ```

3. Train the model:

   ```bash
   python model_training.py
   ```

4. Generate predictions and explain top reasons:

   ```bash
   python generate_report.py
   ```

5. (Optional) Run the interactive dashboard:
   ```bash
   streamlit run dashboard.py
   ```

---

## 📞 Contact Information

Developed by **Ashwani Pandey**  
📧 Email: ashwanip0009@gmail.com

---

⭐ Feel free to contribute, raise issues, or provide feedback!

or

# 🌳 Student Dropout Prediction App

This is a **Streamlit-based dashboard** that predicts the risk of a student dropping out of school and explains the top reasons contributing to the risk.

---

## **Features**

- Input student demographic, academic, and socio-economic data
- Predicts dropout probability (Yes/No + %)
- Shows top 3 features contributing to the prediction
- Easy-to-use interactive dashboard

---


