"""
Milestone 4: Web App Development & Deployment
Professional Visa Processing Time Estimator Web App
"""

import datetime
import os
import sys
import pandas as pd
import streamlit as st

# =========================
# Path Setup
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from predict_processing_days import (
    load_model_and_preprocessing,
    predict_processing_days,
)

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="VisaAI – Processing Time Estimator",
    page_icon="🛂",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# Custom Premium CSS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.main {
    background-color: #f8fafc;
}

h1, h2, h3 {
    color: #0f172a;
}

.hero {
    padding: 2.5rem 1rem 2rem 1rem;
    text-align: center;
}

.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
}

.hero p {
    font-size: 1.1rem;
    color: #475569;
}

.card {
    background: white;
    padding: 1.5rem;
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.05);
    margin-bottom: 1.5rem;
}

.result-card {
    background: linear-gradient(135deg, #2563eb, #1e40af);
    color: white;
    padding: 1.8rem;
    border-radius: 18px;
}

.result-card h2 {
    color: white;
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 800;
}

.footer {
    text-align: center;
    padding: 2rem 0 1rem 0;
    color: #64748b;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Data Loading Helpers
# =========================
@st.cache_data
def get_choice_options():
    try:
        _, prep_info = load_model_and_preprocessing()
        countries = sorted(list(prep_info["country_avg"].keys()))
        visa_types = sorted(list(prep_info["visa_avg"].keys()))
    except Exception:
        countries = ["India", "United States", "Canada", "United Kingdom", "Australia"]
        visa_types = ["Student", "Tourist", "Work", "Business"]
    return {"countries": countries, "visa_types": visa_types}

# =========================
# Main UI
# =========================
def build_ui():

    # ===== HERO SECTION =====
    st.markdown("""
    <div class="hero">
        <h1>🛂 VisaAI – Processing Time Estimator</h1>
        <p>AI-powered predictions to estimate your visa processing timeline with confidence.</p>
    </div>
    """, unsafe_allow_html=True)

    # ===== SIDEBAR =====
    with st.sidebar:
        st.header("About VisaAI")
        st.write(
            "VisaAI uses historical data and machine learning models to provide "
            "approximate estimates of visa processing times."
        )
        st.markdown("### How it works")
        st.markdown(
            "1. Select application date\n"
            "2. Choose country & visa type\n"
            "3. Get instant AI prediction"
        )
        st.info("⚠️ These are estimates only. Actual timelines may vary.")

    options = get_choice_options()
    today = datetime.date.today()

    # ===== INPUT SECTION =====
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📋 Enter Application Details")

    with st.form("visa_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            application_date = st.date_input(
                "Application Date",
                value=today,
                max_value=today + datetime.timedelta(days=365)
            )

        with col2:
            country = st.selectbox(
                "Country of Application",
                options=options["countries"]
            )

        with col3:
            visa_type = st.selectbox(
                "Visa Type",
                options=options["visa_types"]
            )

        submitted = st.form_submit_button("🚀 Estimate Processing Time")

    st.markdown("</div>", unsafe_allow_html=True)

    # ===== RESULT SECTION =====
    if submitted:
        with st.spinner("Analyzing historical data & generating prediction..."):
            try:
                result = predict_processing_days(
                    application_date=str(application_date),
                    country=country,
                    visa_type=visa_type,
                )

                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown("### ✅ Prediction Result")

                colA, colB = st.columns([1, 2])

                with colA:
                    st.markdown(
                        f"<div class='metric-value'>{result['predicted_days']} Days</div>",
                        unsafe_allow_html=True
                    )
                    st.caption("Estimated Processing Time")

                with colB:
                    st.write(f"**Application Date:** {result['application_date']}")
                    st.write(f"**Country:** {result['country']}")
                    st.write(f"**Visa Type:** {result['visa_type']}")
                    st.write(f"**Season:** {result['season']}")
                    st.write(f"**Processing Office:** {result['processing_office']}")
                    st.write(f"**Model Used:** {result['model_type']}")

                st.markdown("</div>", unsafe_allow_html=True)

                # ===== INSIGHTS SECTION =====
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("📊 Application Summary")

                summary_df = pd.DataFrame({
                    "Application Date": [result["application_date"]],
                    "Country": [result["country"]],
                    "Visa Type": [result["visa_type"]],
                    "Estimated Days": [result["predicted_days"]],
                    "Season": [result["season"]],
                    "Processing Office": [result["processing_office"]],
                })

                st.dataframe(summary_df, use_container_width=True)

                st.markdown(
                    "ℹ️ *This prediction is generated using historical trends and seasonal patterns. "
                    "Delays may occur due to policy changes, document verification, or workload variations.*"
                )
                st.markdown("</div>", unsafe_allow_html=True)

            except FileNotFoundError as fnf_err:
                st.error("❌ Model files not found.")
                st.code(str(fnf_err))
                st.info("Please run **Milestone3.py** first to train and generate the model files.")

            except Exception as exc:
                st.error("❌ An unexpected error occurred.")
                st.exception(exc)

    # ===== FOOTER =====
    st.markdown("""
    <div class="footer">
        Milestone 4 – VisaAI Web Application • Built for academic & demonstration purposes  
        <br>© 2026 VisaAI | All rights reserved
    </div>
    """, unsafe_allow_html=True)

# =========================
# Main Entry
# =========================
def main():
    build_ui()

if __name__ == "__main__":
    main()
