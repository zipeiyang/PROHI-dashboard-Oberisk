import streamlit as st

st.set_page_config(
    page_title="ℹ️ About - Oberisk",
    page_icon="./assets/oberisk_icon.svg",
)

st.logo("./assets/oberisk_sidebar.png", size="large", icon_image="./assets/oberisk_logo_icon.png")

# # Page information

st.write("# About")

st.image("./assets/oberisk_side.png", use_container_width=False)

st.markdown(
"""

    ## The Project
    Obesity rates are rising globally, leading to increased cases of chronic diseases and higher healthcare costs. There is a growing need for early identification and prevention strategies. However, there is currently no accessible, personalized tool that uses both anthropometric data and lifestyle history (e.g., physical activity, diet) to predict an individual's risk of obesity. Therefore, stakeholders, especially healthcare providers, require clear, visualized statistics to understand obesity trends and contributing factors, which can support targeted health interventions. 
    OBERISK is a desktop‑based dashboard that enables clinicians to estimate an adult patient’s obesity risk using a machine learning model trained on demographic, anthropometric, family‑history, and lifestyle features. The initial demonstration will run in Streamlit with a potential for future EHR integration and commercialization.

    ## The Dataset
    The dataset combines information from individuals in Mexico, Peru, and Colombia. The dataset captures key attributes related to eating habits, physical activity, and health status (4). It consists of 2,111 records and 17 features, with each entry labelled according to the NObesity class variable. This label categorizes individuals into one of seven obesity levels: Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, and Obesity Type III. Notably, 23% of the data was collected directly from users via a web platform, while the remaining 77% was synthetically generated using the SMOTE filter within the Weka tool, ensuring a more balanced distribution across classes.
    
    ## Model Justification

    ## Team OBERISK

    ## References
    
"""
)