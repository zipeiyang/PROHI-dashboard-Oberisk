import streamlit as st

st.set_page_config(
    page_title="Home - Oberisk",
    page_icon="./assets/oberisk_icon.svg",
)

st.logo("./assets/oberisk_sidebar.png", size="large", icon_image="./assets/oberisk_logo_icon.png")

# Sidebar configuration
# st.sidebar.image("./assets/oberisk_centered.svg",)
# st.sidebar.success("Select a tab above.")

# # Page information

st.write("# Welcome to OBERISK")

st.markdown(
"""
    Oberisk is desktop‑based dashboard that enables individuals to estimate their obesity risk using a machine learning model trained on demographic, family‑history, and lifestyle features presented in an accessible and intuitive dashboard.
"""
)

st.markdown("""
<style>
.feature-card {
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 18px;
    background-color: #f9f9f9;
    box-shadow: 1px 1px 3px rgba(0,0,0,0.05);
    height: 100%;
}
.feature-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #333;
    margin-bottom: 8px;
}
.feature-desc {
    font-size: 0.95rem;
    color: #555;
}
</style>
""", unsafe_allow_html=True)

st.markdown("### 💡 What You Can Do with **OBERISK**")

# --- First row (2 columns) ---
st.markdown('<div class="row-container">', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
    <div class="feature-title">🧮 Estimate Your Obesity Risk</div>
    <div class="feature-desc">Input your age, sex, family history, dietary habits, and lifestyle to obtain an individualized obesity risk score.</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
    <div class="feature-title">📊 Visualize Risk Contributors</div>
    <div class="feature-desc">See which factors contribute most to your predicted risk through dynamic charts, powered by explainable AI.</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Second row (2 columns) ---
st.markdown('<div class="row-container">', unsafe_allow_html=True)
col3, col4 = st.columns(2)

with col3:
    st.markdown("""
    <div class="feature-card">
    <div class="feature-title">⚙️ Compare Scenarios</div>
    <div class="feature-desc">Modify one or more habits and instantly see how your risk score changes, a “what-if” simulation to support behavioral change.</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="feature-card">
    <div class="feature-title">🎯 Get Personalized Insights</div>
    <div class="feature-desc">Receive tailored recommendations and educational tips grounded in evidence-based obesity prevention strategies.</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)



if st.button("Calculate risk", help="Input your patient data to predict the risk", type="primary", use_container_width=True):
    st.switch_page("pages/3_🎯_Predictive_Analytics.py")