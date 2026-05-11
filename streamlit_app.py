import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Telecom Churn AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }

    .main-content {
        background: rgba(20, 20, 35, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }

    .hero-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
    }

    .hero-header p {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 400;
    }

    .custom-card {
        background: linear-gradient(145deg, #1e1e30 0%, #252540 100%);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.3s ease;
    }

    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .section-title::before {
        content: '';
        width: 4px;
        height: 24px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        border-radius: 2px;
    }

    .metric-card {
        background: linear-gradient(145deg, #1e1e30 0%, #252540 100%);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .metric-card .label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.5rem;
    }

    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #e2e8f0;
    }

    .metric-card.highlight .value {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .result-card {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }

    .result-card.high-risk {
        background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%);
        border: 2px solid #ef4444;
    }

    .result-card.low-risk {
        background: linear-gradient(135deg, #052e16 0%, #064e3b 100%);
        border: 2px solid #10b981;
    }

    .result-card .title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .result-card.high-risk .title { color: #fca5a5; }
    .result-card.low-risk .title { color: #6ee7b7; }

    .result-card .probability {
        font-size: 2.5rem;
        font-weight: 700;
    }

    .result-card.high-risk .probability { color: #fca5a5; }
    .result-card.low-risk .probability { color: #6ee7b7; }

    [data-testid="stSidebar"] {
        background: rgba(20, 20, 35, 0.98);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        color: white;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.4);
    }

    .info-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #172554 100%);
        border-left: 4px solid #3b82f6;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        color: #bfdbfe;
    }

    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #374151, transparent);
        margin: 2rem 0;
    }

    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.875rem;
    }

    .stRadio > div {
        background: #1e1e30;
        border-radius: 12px;
        padding: 0.5rem;
    }

    .stRadio > div > label {
        color: #e2e8f0;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }

    .stRadio > div > label:has(input:checked) {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
    }

    [data-testid="stDataFrame"] {
        background: #1e1e30;
        border-radius: 12px;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #e2e8f0 !important;
    }

    [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_data():
    df = pd.read_csv("telecom_churn_cleaned.csv")

    x = df[['Tenure', 'MonthlyCharges_Clean', 'TotalCharges_Clean']].copy()
    y = df['Churn']

    x.loc[:, 'TotalCharges_Clean'] = pd.to_numeric(x['TotalCharges_Clean'], errors='coerce')
    median_val = x['TotalCharges_Clean'].median()
    x.loc[:, 'TotalCharges_Clean'] = pd.to_numeric(x['TotalCharges_Clean'].fillna(median_val), errors='coerce')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)

    return {
        'model': model,
        'df': df,
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

model_data = load_model_and_data()
model = model_data['model']
df = model_data['df']
y_test = model_data['y_test']
y_pred = model_data['y_pred']
y_pred_proba = model_data['y_pred_proba']
x_test = model_data['x_test']

st.markdown('<div class="main-content">', unsafe_allow_html=True)

st.markdown("""
    <div class="hero-header" style="text-align: center; padding: 2rem 0 1rem 0;">
        <h1>Telecom Churn Predictor AI</h1>
        <p>Intelligent customer retention forecasting system</p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0 0.5rem 0;">
            <h2 style="color: #e2e8f0; margin: 0; font-size: 1.25rem;">Navigation</h2>
        </div>
    """, unsafe_allow_html=True)
    page = st.radio("Select Page", ["Prediction", "Analytics", "Model Details", "Dataset Explorer"], label_visibility="collapsed")

if page == "Prediction":
    st.markdown('<div class="section-title">Customer Churn Prediction</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.3, 0.7], gap="large")

    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### Customer Details")

        col_a, col_b = st.columns(2)
        with col_a:
            tenure = st.slider("Tenure (months)", 0, 100, 12, 1, help="Customer relationship duration")
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 500.0, 65.0, 0.50)
        with col_b:
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 780.0, 10.0)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### Summary")
        st.metric("Tenure", f"{tenure} months")
        st.markdown("")
        st.metric("Monthly", f"${monthly_charges:,.2f}")
        st.markdown("")
        st.metric("Total", f"${total_charges:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    if st.button("Predict Churn", type="primary", use_container_width=True):
        input_data = np.array([[tenure, monthly_charges, total_charges]])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        churn_prob = probability[1] * 100
        no_churn_prob = probability[0] * 100

        st.markdown("")

        if prediction == 'Yes':
            st.markdown(f"""
                <div class="result-card high-risk">
                    <div class="title">High Risk of Churn</div>
                    <div class="probability">{churn_prob:.1f}%</div>
                    <p style="color: #fca5a5; margin-top: 0.5rem;">Customer is likely to churn</p>
                </div>
            """, unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = churn_prob,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Probability", 'font': {'size': 20, 'color': '#e2e8f0'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                    'bar': {'color': "#ef4444"},
                    'bgcolor': "#1e1e30",
                    'borderwidth': 2,
                    'bordercolor': "#374151",
                    'steps': [
                        {'range': [0, 30], 'color': "#064e3b"},
                        {'range': [30, 60], 'color': "#451a03"},
                        {'range': [60, 100], 'color': "#450a0a"}
                    ],
                },
                number = {'font': {'color': '#fca5a5'}}
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=60, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': '#e2e8f0'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown(f"""
                <div class="result-card low-risk">
                    <div class="title">Low Risk of Churn</div>
                    <div class="probability">{no_churn_prob:.1f}%</div>
                    <p style="color: #6ee7b7; margin-top: 0.5rem;">Customer is likely to stay</p>
                </div>
            """, unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = churn_prob,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Probability", 'font': {'size': 20, 'color': '#e2e8f0'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                    'bar': {'color': "#10b981"},
                    'bgcolor': "#1e1e30",
                    'borderwidth': 2,
                    'bordercolor': "#374151",
                    'steps': [
                        {'range': [0, 30], 'color': "#064e3b"},
                        {'range': [30, 60], 'color': "#451a03"},
                        {'range': [60, 100], 'color': "#450a0a"}
                    ],
                },
                number = {'font': {'color': '#6ee7b7'}}
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=60, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': '#e2e8f0'})
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Key Factors")
        col1, col2, col3 = st.columns(3)

        tenure_risk = "High" if tenure < 12 else "Medium" if tenure < 24 else "Low"
        charges_risk = "High" if monthly_charges > 70 else "Medium" if monthly_charges > 50 else "Low"

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Tenure Risk</div>
                <div class="value" style="font-size: 1.25rem; color: {'#ef4444' if tenure_risk == 'High' else '#f59e0b' if tenure_risk == 'Medium' else '#10b981'};">{tenure_risk}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Charges Risk</div>
                <div class="value" style="font-size: 1.25rem; color: {'#ef4444' if charges_risk == 'High' else '#f59e0b' if charges_risk == 'Medium' else '#10b981'};">{charges_risk}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="label">Recommendation</div>
                <div class="value" style="font-size: 1rem; color: #6366f1;">Review Plan</div>
            </div>
            """, unsafe_allow_html=True)


elif page == "Analytics":
    st.markdown('<div class="section-title">Model Performance Analytics</div>', unsafe_allow_html=True)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="metric-card highlight">
                <div class="label">Accuracy</div>
                <div class="value">{accuracy*100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="label">True Negatives</div>
                <div class="value">{cm[0, 0]}</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="label">True Positives</div>
                <div class="value">{cm[1, 1]}</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="label">False Positives</div>
                <div class="value">{cm[0, 1]}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### Confusion Matrix")

        fig = make_subplots(rows=1, cols=1, specs=[[{"type": "heatmap"}]])

        fig.add_trace(go.Heatmap(
            z=cm,
            x=['Predicted<br>No Churn', 'Predicted<br>Churn'],
            y=['Actual<br>No Churn', 'Actual<br>Churn'],
            colorscale=[[0, '#1e3a5f'], [1, '#6366f1']],
            showscale=True,
            colorbar=dict(title="Count", thickness=20, len=0.8),
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 24, "color": "white"}
        ))

        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': '#e2e8f0'},
            xaxis=dict(tickfont={'color': '#e2e8f0'}),
            yaxis=dict(tickfont={'color': '#e2e8f0'})
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### Prediction Distribution")

        fig = go.Figure(data=[
            go.Histogram(
                x=y_pred_proba[:, 1] * 100,
                nbinsx=30,
                marker_color='#6366f1',
                marker_line_color='#4338ca',
                marker_line_width=1
            )
        ])
        fig.update_layout(
            xaxis_title="Churn Probability (%)",
            yaxis_title="Customer Count",
            height=350,
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': '#e2e8f0'},
            xaxis=dict(tickfont={'color': '#e2e8f0'}, title=dict(font={'color': '#e2e8f0'})),
            yaxis=dict(tickfont={'color': '#e2e8f0'}, title=dict(font={'color': '#e2e8f0'}))
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Classification Report")
    cr = classification_report(y_test, y_pred, output_dict=True)

    report_data = pd.DataFrame({
        'Class': ['No Churn', 'Churn', 'Weighted Avg'],
        'Precision': [f"{cr['No']['precision']:.3f}", f"{cr['Yes']['precision']:.3f}", f"{cr['weighted avg']['precision']:.3f}"],
        'Recall': [f"{cr['No']['recall']:.3f}", f"{cr['Yes']['recall']:.3f}", f"{cr['weighted avg']['recall']:.3f}"],
        'F1-Score': [f"{cr['No']['f1-score']:.3f}", f"{cr['Yes']['f1-score']:.3f}", f"{cr['weighted avg']['f1-score']:.3f}"],
        'Support': [int(cr['No']['support']), int(cr['Yes']['support']), int(cr['weighted avg']['support'])]
    })

    st.dataframe(report_data, use_container_width=True, hide_index=True, height=150)


elif page == "Model Details":
    st.markdown('<div class="section-title">Model Specification</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### Overview")
        st.markdown("""
        **Algorithm:** Logistic Regression
        **Type:** Binary Classification
        **Task:** Customer Churn Prediction
        **Training Samples:** 5,639
        **Test Samples:** 1,410
        **Features:** 3
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### Features")
        st.markdown("""
        **1. Tenure**
        Customer relationship duration in months

        **2. Monthly Charges**
        Monthly service cost in USD

        **3. Total Charges**
        Cumulative service cost in USD
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### Model Coefficients")

    m = model.coef_[0]
    c = model.intercept_[0]

    coef_data = pd.DataFrame({
        'Feature': ['Tenure', 'Monthly Charges', 'Total Charges', 'Intercept (Bias)'],
        'Coefficient': [f"{m[0]:.6f}", f"{m[1]:.6f}", f"{m[2]:.6f}", f"{c:.6f}"],
        'Impact': ['Decreases churn', 'Increases churn', 'Decreases churn', 'Base probability']
    })

    st.dataframe(coef_data, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="info-box">
            <strong>Insight:</strong> Positive coefficients increase churn probability while negative coefficients decrease it.
            Tenure has a strong negative effect (longer customers are more loyal), while higher monthly charges slightly increase churn risk.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### Mathematical Model")
    st.latex(r"P(Churn) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 \cdot Tenure + \beta_2 \cdot MonthlyCharges + \beta_3 \cdot TotalCharges)}}")


elif page == "Dataset Explorer":
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    total_records = len(df)
    churn_count = (df['Churn'] == 'Yes').sum()
    retained = total_records - churn_count
    churn_rate = (churn_count / total_records) * 100

    with col1:
        st.markdown(f"""
            <div class="metric-card highlight">
                <div class="label">Total Records</div>
                <div class="value">{total_records:,}</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="label">Churned</div>
                <div class="value" style="color: #ef4444;">{churn_count:,}</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="label">Retained</div>
                <div class="value" style="color: #10b981;">{retained:,}</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="label">Churn Rate</div>
                <div class="value" style="color: #f59e0b;">{churn_rate:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### Churn Distribution")

        churn_counts = df['Churn'].value_counts()
        fig = go.Figure(data=[
            go.Pie(
                labels=['Retained', 'Churned'],
                values=churn_counts.values,
                marker=dict(colors=['#10b981', '#ef4444']),
                textinfo='percent+label',
                textfont=dict(size=14, color='white'),
                hole=0.6
            )
        ])
        fig.update_layout(
            height=300,
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': '#e2e8f0'}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### Tenure Distribution")

        fig = px.histogram(df, x='Tenure', nbins=20, color_discrete_sequence=['#6366f1'])
        fig.update_layout(
            height=300,
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=40),
            xaxis_title="Tenure (months)",
            yaxis_title="Count",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': '#e2e8f0'},
            xaxis=dict(tickfont={'color': '#e2e8f0'}, title=dict(font={'color': '#e2e8f0'})),
            yaxis=dict(tickfont={'color': '#e2e8f0'}, title=dict(font={'color': '#e2e8f0'}))
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### Monthly Charges Distribution")

    fig = px.histogram(df, x='MonthlyCharges_Clean', nbins=25, color_discrete_sequence=['#8b5cf6'])
    fig.update_layout(
        height=280,
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=40),
        xaxis_title="Monthly Charges ($)",
        yaxis_title="Count",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e2e8f0'},
        xaxis=dict(tickfont={'color': '#e2e8f0'}, title=dict(font={'color': '#e2e8f0'})),
        yaxis=dict(tickfont={'color': '#e2e8f0'}, title=dict(font={'color': '#e2e8f0'}))
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Data Sample")
    col1, col2 = st.columns([1, 3])
    with col1:
        n_rows = st.slider("Rows", 5, 50, 10)
    with col2:
        st.dataframe(df.head(n_rows), use_container_width=True, height=300)

    csv_data = df.to_csv(index=False)
    st.download_button("Download Dataset", csv_data, "telecom_churn_data.csv", "text/csv", use_container_width=True)

st.markdown("""
    <div class="footer">
        <hr>
        <p>Telecom Churn Predictor AI | Powered by Logistic Regression</p>
    </div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)