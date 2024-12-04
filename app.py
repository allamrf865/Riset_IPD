import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
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
            background: linear-gradient(135deg, #F8F9FA, #E9ECEF);
            font-family: 'Poppins', sans-serif;
        }
        .header {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .header h1 {
            font-size: 2.5rem;
            color: #007BFF;
            text-align: center;
        }
        .header p {
            font-size: 1.2rem;
            color: #495057;
            text-align: center;
        }
        .dataset-card {
            border: 1px solid #DEE2E6;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 20px;
            background: white;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        }
        .dataset-card h4 {
            margin: 0;
            color: #007BFF;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header">
        <h1>📊 AI Research Scientist Evaluation</h1>
        <p>Analyze datasets with advanced metrics and a sleek, interactive interface.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for file upload
st.sidebar.title("📂 Upload Your Datasets")
uploaded_files = st.sidebar.file_uploader(
    "Upload up to 10 Excel files",
    type=["xlsx"],
    accept_multiple_files=True,
    help="You can upload multiple Excel files (up to 10)."
)
# Branding AI and Bio Information
st.sidebar.markdown("""
    <div class="header">
        <h1>🌟 About Me - Muhammad Allam Rafi</h1>
        <p>AI Enthusiast | Developer | Researcher</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
    <div class="dataset-card">
        <h4>📋 Biodata</h4>
        <p><strong>Name:</strong> Muhammad Allam Rafi</p>
        <p><strong>Organization:</strong> TBM, FSI, Oxygen, LPP, JIMKI</p>
        <p><strong>Faculty:</strong> FKUI 2022</p>
        <p><strong>Mission:</strong> Dedicated to enhancing AI and machine learning applications to solve real-world problems. Passionate about research and innovation in artificial intelligence, and committed to making impactful contributions in healthcare and other sectors.</p>
        <p><strong>Instagram:</strong> <a href="https://instagram.com/allamrf865" target="_blank">instagram.com/allamrf865</a></p>
    </div>
""", unsafe_allow_html=True)

# Watermark or Footer to give credit
st.sidebar.markdown("""
    <p style="text-align:center; font-size:14px; color:gray;">
        Created by Muhammad Allam Rafi | FKUI 2022
    </p>
""", unsafe_allow_html=True)

# Main layout for datasets
if uploaded_files:
    st.markdown("### 📊 Datasets Overview")
    if len(uploaded_files) > 10:
        st.error("You can only upload a maximum of 10 datasets!")
    else:
        tabs = st.tabs([f"Dataset {i+1}" for i in range(len(uploaded_files))])

        for i, uploaded_file in enumerate(uploaded_files):
            with tabs[i]:
                # Load dataset
                try:
                    df = pd.read_excel(uploaded_file)
                    st.markdown(f"#### Dataset {i + 1}: {uploaded_file.name}")
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error loading {uploaded_file.name}: {str(e)}")

                # Allow selection of target column and positive class
                target_col = st.selectbox(
                    f"Select target column for Dataset {i + 1}",
                    options=df.columns,
                    key=f"target_col_{i}"
                )
                pos_label = st.selectbox(
                    f"Select positive class for Dataset {i + 1}",
                    options=df[target_col].unique(),
                    key=f"pos_label_{i}"
                )

                # Analyze dataset button
                if st.button(f"Analyze Dataset {i + 1}", key=f"analyze_{i}"):
                    with st.spinner("Analyzing data..."):
                        # Data Preprocessing
                        # Handle missing values
                        imputer = SimpleImputer(strategy="most_frequent")
                        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
                        
                        # Convert target column to binary (0, 1)
                        df_imputed[target_col] = df_imputed[target_col].apply(lambda x: 1 if x == pos_label else 0)

                        # Feature scaling (standardize)
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(df_imputed.drop(columns=[target_col]))
                        y = df_imputed[target_col]

                        # Train-test split
                        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

                        # Ensemble Learning Models
                        base_learners = [
                            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
                            ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
                            ("ab", AdaBoostClassifier(n_estimators=100, random_state=42)),
                            ("svm", SVC(probability=True, random_state=42))
                        ]

                        # Stacking Classifier
                        model = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        y_pred_prob = model.predict_proba(X_test)[:, 1]

                        # Confusion Matrix Calculation
                        cm = confusion_matrix(y_test, y_pred)
                        TP, TN, FP, FN = cm.ravel()

                        # Metric Calculations
                        sensitivity = TP / (TP + FN)
                        specificity = TN / (TN + FP)
                        positive_predictive_value = TP / (TP + FP)
                        negative_predictive_value = TN / (TN + FN)
                        prevalence = (TP + FN) / len(y)
                        positive_likelihood_ratio = sensitivity / (1 - specificity) if (1 - specificity) != 0 else float('inf')
                        negative_likelihood_ratio = (1 - sensitivity) / specificity if specificity != 0 else float('inf')
                        accuracy = accuracy_score(y_test, y_pred)
                        auc = roc_auc_score(y_test, y_pred_prob)

                        # Display Metrics
                        st.markdown("#### Model Metrics")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("AUC", f"{auc:.2f}")
                        col2.metric("Sensitivity (Recall)", f"{sensitivity:.2f}")
                        col3.metric("Specificity", f"{specificity:.2f}")
                        col4, col5 = st.columns(2)
                        col4.metric("Accuracy", f"{accuracy:.2f}")

                        # Additional Metrics
                        st.markdown("#### Additional Metrics")
                        st.write(f"**Nilai Duga Positif (Positive Predictive Value)**: {positive_predictive_value:.2f}")
                        st.write(f"**Nilai Duga Negatif (Negative Predictive Value)**: {negative_predictive_value:.2f}")
                        st.write(f"**Prevalensi (Prevalence)**: {prevalence:.2f}")
                        st.write(f"**Rasio Kemungkinan Positif (Positive Likelihood Ratio)**: {positive_likelihood_ratio:.2f}")
                        st.write(f"**Rasio Kemungkinan Negatif (Negative Likelihood Ratio)**: {negative_likelihood_ratio:.2f}")

                        # Confusion Matrix
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

                        # Watermark
                        st.markdown("""
                            <p style="text-align:center; font-size:14px; color:gray;">
                                Created by Allam Rafi FKUI 2022
                            </p>
                        """, unsafe_allow_html=True)
else:
    st.markdown("### Please upload datasets to get started!")
