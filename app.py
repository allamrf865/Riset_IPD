# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set Streamlit page configuration
st.set_page_config(
    page_title="AI Research Scientist Evaluation",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #F0F8FF, #ADD8E6);
            font-family: 'Poppins', sans-serif;
        }
        .header {
            background: rgba(255, 255, 255, 0.85);
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 2rem;
        }
        .header h1 {
            font-size: 2.5rem;
            color: #007BFF;
        }
        .header p {
            font-size: 1.2rem;
            color: #333333;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header">
        <h1>📊 AI Research Scientist Evaluation</h1>
        <p>Analyze machine learning models with advanced metrics and stunning visualizations.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for file upload
st.sidebar.title("📂 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"])

# Main layout
if uploaded_file:
    st.markdown("### 📊 Dataset Overview and Analysis")

    # Read the dataset
    df = pd.read_excel(uploaded_file)
    st.dataframe(df.head())

    # Select target column and positive class
    target_col = st.selectbox("Select the target column", options=df.columns)
    pos_label = st.selectbox("Select the positive class", options=df[target_col].unique())

    # Analyze dataset button
    if st.button("Analyze Dataset"):
        with st.spinner("Analyzing data..."):
            # Data preprocessing
            df[target_col] = df[target_col].apply(lambda x: 1 if x == pos_label else 0)
            X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
            y = df[target_col]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            auc = roc_auc_score(y_test, y_pred_prob)
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            accuracy = accuracy_score(y_test, y_pred)
            ppv = tp / (tp + fp)  # Positive Predictive Value
            npv = tn / (tn + fn)  # Negative Predictive Value
            prevalence = (tp + fn) / (tn + fp + fn + tp)
            lr_positive = sensitivity / (1 - specificity)
            lr_negative = (1 - sensitivity) / specificity

            # Display metrics
            st.markdown("### Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Sensitivity", f"{sensitivity:.2f}")
            col2.metric("Specificity", f"{specificity:.2f}")
            col3.metric("Accuracy", f"{accuracy:.2f}")

            col4, col5, col6 = st.columns(3)
            col4.metric("PPV (Precision)", f"{ppv:.2f}")
            col5.metric("NPV", f"{npv:.2f}")
            col6.metric("Prevalence", f"{prevalence:.2f}")

            col7, col8 = st.columns(2)
            col7.metric("Likelihood Ratio +", f"{lr_positive:.2f}")
            col8.metric("Likelihood Ratio -", f"{lr_negative:.2f}")

            # Confusion Matrix Heatmap
            st.markdown("#### Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)

            # ROC Curve
            st.markdown("#### ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            fig, ax = plt.subplots()
            plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
            plt.plot([0, 1], [0, 1], 'r--', label="Random Guess")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            st.pyplot(fig)

else:
    st.markdown("### Please upload a dataset to get started!")
