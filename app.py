# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
# Import matplotlib for 3D plotting
from mpl_toolkits.mplot3d import Axes3D
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
        .metric-box {
            background-color: white;
            border: 2px solid #007BFF;
            color: #333333;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
        }
        .metric-box h1 {
            margin: 0;
            font-size: 1.8rem;
            color: #007BFF;
        }
        .metric-box p {
            margin: 0;
            font-size: 1.2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header">
        <h1>📊 AI Research Scientist Evaluation</h1>
        <p>Analyze machine learning models with advanced metrics and stunning 2D & 3D visualizations.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for file upload
st.sidebar.title("📂 Upload Your Dataset")
uploaded_files = st.sidebar.file_uploader("Upload Excel files", type=["xlsx"], accept_multiple_files=True)

# Main layout
if uploaded_files:
    st.markdown("### 📊 Datasets Overview and Analysis")
    for idx, uploaded_file in enumerate(uploaded_files):
        # Read the dataset
        df = pd.read_excel(uploaded_file)
        st.markdown(f"#### Dataset {idx + 1}: {uploaded_file.name}")
        st.dataframe(df.head())

        # Select target column and positive class
        target_col = st.selectbox(
            f"Select the target column for Dataset {idx + 1}",
            options=df.columns,
            key=f"target_{idx}"
        )
        pos_label = st.selectbox(
            f"Select the positive class for Dataset {idx + 1}",
            options=df[target_col].unique(),
            key=f"pos_label_{idx}"
        )

        # Analyze dataset button
        if st.button(f"Analyze Dataset {idx + 1}", key=f"analyze_{idx}"):
            with st.spinner("Analyzing data..."):
                # Data preprocessing
                df[target_col] = df[target_col].apply(lambda x: 1 if x == pos_label else 0)
                X = df.drop(columns=[target_col])
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
                auc = roc_auc_score(y_test, y_pred_prob)
                sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
                specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
                accuracy = accuracy_score(y_test, y_pred)

                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("AUC", f"{auc:.2f}")
                col2.metric("Sensitivity", f"{sensitivity:.2f}")
                col3.metric("Specificity", f"{specificity:.2f}")
                col4.metric("Accuracy", f"{accuracy:.2f}")

                # Confusion Matrix Heatmap
                st.markdown("#### Confusion Matrix (2D)")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
                plt.title("Confusion Matrix")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                st.pyplot(fig)

                # ROC Curve
                st.markdown("#### ROC Curve (2D)")
                fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                fig, ax = plt.subplots()
                plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
                plt.plot([0, 1], [0, 1], 'r--', label="Random Guess")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend()
                st.pyplot(fig)

                # 3D ROC Curve Visualization
                st.markdown("#### ROC Curve (3D Visualization)")
                thresholds = np.linspace(0, 1, len(fpr))
                fig_3d = go.Figure(data=[
                    go.Surface(
                        z=np.array([tpr, fpr, thresholds]).T,
                        colorscale="Viridis",
                        showscale=True
                    )
                ])
                fig_3d.update_layout(
                    title="3D ROC Curve",
                    scene=dict(
                        xaxis_title="Thresholds",
                        yaxis_title="False Positive Rate",
                        zaxis_title="True Positive Rate"
                    )
                )
                st.plotly_chart(fig_3d)

else:
    st.markdown("### Please upload a dataset to get started!")

