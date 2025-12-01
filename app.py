import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Page config
st.set_page_config(page_title="🧬 Pima Diabetes App with Tabs", layout="wide")

st.title("🧬 Pima Indian Diabetes Prediction")

# Sidebar: Upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Load or use sample
if uploaded_file is not None:
     df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("Using sample data")
    df = pd.read_csv("diabetes.csv")  # <- update with your file name!

# Clean data: replace 0s
cols_to_fix = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for col in cols_to_fix:
    df[col] = df[col].replace(0, df[col].mean())

# Train-test split (for later)
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📄 Data Preview", "📊 EDA & Plots", "🤖 Model Training", "🔍 Predict"])

# ----- Tab 1 -----
with tab1:
    st.subheader("📄 Raw Data")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())

# ----- Tab 2 -----
with tab2:
    st.subheader("📊 Exploratory Data Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Outcome Count")
        fig1, ax1 = plt.subplots()
        df['Outcome'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
        plt.xticks(ticks=[0, 1], labels=['No Diabetes', 'Diabetes'])
        st.pyplot(fig1)

    with col2:
        st.write("### Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        st.pyplot(fig2)

# ----- Tab 3 -----
with tab3:
    st.subheader("🤖 Model Training & Evaluation")

    model_choice = st.selectbox(
        "Choose Classifier",
        ("Logistic Regression", "Random Forest", "SVM")
    )

    if model_choice == "Logistic Regression":
        model = LogisticRegression()
    elif model_choice == "Random Forest":
        model = RandomForestClassifier()
    else:
        model = SVC(kernel='linear')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write(f"### ✅ {model_choice} Results")
    st.write("**Accuracy:**", round(accuracy_score(y_test, y_pred), 2))

    fig3, ax3 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='d')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig3)

    st.text(classification_report(y_test, y_pred))

# ----- Tab 4 -----
with tab4:
    st.subheader("🔍 Make a Prediction")
    st.write("Input patient details below:")

    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 200, 120)
    bp = st.number_input("Blood Pressure", 0, 122, 70)
    skin = st.number_input("Skin Thickness", 0, 99, 20)
    insulin = st.number_input("Insulin", 0, 846, 79)
    bmi = st.number_input("BMI", 0.0, 70.0, 32.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.number_input("Age", 21, 100, 30)

    if st.button("Predict"):
        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        prediction = model.predict(input_data)
        st.write("### 🎉 Prediction Result:")
        if prediction[0] == 1:
            st.error("The model predicts: DIABETIC")
        else:
            st.success("The model predicts: NOT DIABETIC")
