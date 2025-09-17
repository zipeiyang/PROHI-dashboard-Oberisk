import streamlit as st

st.set_page_config(
    page_title="Home - Oberisk",
    page_icon="./assets/oberisk_icon.svg",
)

st.logo("./assets/oberisk_sidebar.png", size="large", icon_image="./assets/oberisk_icon.svg")

# Sidebar configuration
# st.sidebar.image("./assets/oberisk_centered.svg",)
# st.sidebar.success("Select a tab above.")

# # Page information

st.write("# Welcome to OBERISK")

st.markdown(
"""
    Oberisk is a data-driven tool designed to support clinicians in tackling adult obesity. Our platform helps identify individuals at higher risk of obesity through advanced machine-learning models, presented in an accessible and intuitive dashboard.
    ## What you can do with OBERISK?
    - Describe population trends in obesity and lifestyle factors.
    - Analyze key risk factors and patterns with clear visualizations.
    - Predict individual obesity risk by entering patient data.
    
    Oberisk brings prediction and prevention together, giving healthcare professionals the insight they need to act early and improve long-term health outcomes.
"""
)

if st.button("Calculate risk", help="Input your patient data to predict the risk", type="primary", use_container_width=True):
    st.switch_page("pages/3_🎯_Predictive_Analytics.py")