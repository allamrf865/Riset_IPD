# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score
import plotly.express as px
import plotly.graph_objects as go
import time

# Set page configuration
st.set_page_config(
    page_title="AI Research Scientist Evaluation",
    page_icon="🧪",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #FAD961, #F76B1C);
            font-family: 'Poppins', sans-serif;
            color: #333333;
        }
        .reportview-container {
            padding: 1rem 2rem;
        }
        .header {
            background: rgba(255, 255, 255, 0.85);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 2rem;
        }
        .header h1 {
            font-size: 2.5rem;
            color: #FF5733;
        }
        .header p {
            font-size: 1.2rem;
            color: #333333;
        }
        .metric-box {
            background-color: white;
            border: 2px solid #FF5733;
            color: #333333;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
        }
        .metric-box h1 {
            margin: 0;
            font-size: 1.8rem;
            color: #FF5733;
        }
        .metric-box p {
            margin: 0;
            font-size: 1.2rem;
        }
        .watermark {
            font-size: 12px;
            color: white;
            text-align: right;
            margin-top: 50px;
        }
        .dataframe {
            background-color: white;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header">
        <h1>🧪 AI Research Scientist Evaluation 🧪</h1>
        <p>Analyze datasets with advanced metrics and state-of-the-art visualizations, designed for research scientists.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for file upload
st.sidebar.title("📂 Upload Your Dataset")
uploaded_files = st.sidebar.file_uploader("Upload Excel files", type=["xlsx"], accept_multiple_files=True)

# Footer watermark
st.markdown("""
    <div class="watermark">
        AI by Allam Rafi FKUI 2022
    </div>
""", unsafe_allow_html=True)

# Main layout
if uploaded_files:
    st.markdown("### 📊 Datasets Overview and Analysis")
    for idx, uploaded_file in enumerate(uploaded_files):
        # Read the dataset
        df = pd.read_excel(uploaded_file)
        st.markdown(f"#### Dataset {idx + 1}: {uploaded_file.name}")
        st.markdown('<div class="dataframe">', unsafe_allow_html=True)
        st.dataframe(df.head())
        st.markdown('</div>', unsafe_allow_html=True)

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

        # Button to analyze the dataset
        if st.button(f"Analyze Dataset {idx + 1}", key=f"analyze_{idx}"):
            # Show progress bar
            with st.spinner("Analyzing data..."):
                time.sleep(2)  # Simulating computation time

            # Process data
            df[target_col] = df[target_col].apply(lambda x: 1 if x == pos_label else 0)
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Split data and train model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            cm = confusion_matrix(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_prob)

            # Display metrics as cards
            col1, col2, col3 = st.columns(3)
            col1.markdown("""
                <div class="metric-box">
                    <h1>AUC</h1>
                    <p>{:.2f}</p>
                </div>
            """.format(auc), unsafe_allow_html=True)
            col2.markdown("""
                <div class="metric-box">
                    <h1>Sensitivity</h1>
                    <p>{:.2f}</p>
                </div>
            """.format(cm[1, 1] / (cm[1, 1] + cm[1, 0])), unsafe_allow_html=True)
            col3.markdown("""
                <div class="metric-box">
                    <h1>Specificity</h1>
                    <p>{:.2f}</p>
                </div>
            """.format(cm[0, 0] / (cm[0, 0] + cm[0, 1])), unsafe_allow_html=True)

            # Confusion matrix heatmap
            st.markdown("#### Confusion Matrix")
            fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"), x=["Negative", "Positive"], y=["Negative", "Positive"])
            st.plotly_chart(fig)

            # ROC curve
            st.markdown("#### ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(dash='dash')))
            fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig)

else:
    st.markdown("""
        <div style="text-align: center; margin-top: 50px;">
            <h3>👈 Upload your dataset to start analyzing!</h3>
        </div>
    """, unsafe_allow_html=True)
