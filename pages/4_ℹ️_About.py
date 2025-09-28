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

    What OBERISK offers:
    - Fast risk estimate
    - Visualized feature effects to support counseling on modifiable risks.
    - Configurable thresholds to adapt sensitivity and specificity to clinical priorities, for instances screening and diagnostic.


    ## The Dataset
    The dataset combines information from individuals in Mexico, Peru, and Colombia. The dataset captures key attributes related to eating habits, physical activity, and health status (4). It consists of 2,111 records and 17 features, with each entry labelled according to the NObesity class variable. This label categorizes individuals into one of seven obesity levels: Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, and Obesity Type III. Notably, 23% of the data was collected directly from users via a web platform, while the remaining 77% was synthetically generated using the SMOTE filter within the Weka tool, ensuring a more balanced distribution across classes.
    
    For modeling, we converted the 7-class label to a binary outcome:
    - No risk (0): Insufficient/Normal
    - At risk (1): Overweight I–II and Obesity I–III
    
    ### Important caveats about the label
    The “obesity level” is derived directly from height and weight (BMI categories), it is not a clinical diagnosis. This introduces **label-definition leakage**: models that see height/weight will (correctly) learn BMI boundaries, and lifestyle features will contribute less to predictions of current status.
    
    ## Model Justification
    **Why binary classification?**

    - Clinically, the key fork in decisions is “screen and counsel intensively vs. routine prevention.” Binarizing (Normal/Underweight vs. Overweight/Obesity) aligns with that workflow and improves class balance for evaluation and threshold tuning.
    - It also simplifies calibration (well-calibrated risk probabilities) and operating point selection (e.g., high-recall screening).

    **Why XGBoost (and why compare baselines)?**

    We evaluated Logistic Regression, Random Forest, XGBoost, and LightGBM using a consistent preprocessing pipeline (scaling numeric features; one-hot encoding categoricals) and stratified train/test splits. We selected XGBoost as the primary model because it:
    - Handles non-linearities and feature interactions between lifestyle, family history, and anthropometrics.
    - Is robust to mixed data types and class imbalance.
    - Trains quickly, supports probability calibration, and deploys easily in Streamlit.

    **Evaluation & operating points**

    For each model we report:
    - Accuracy, Precision, Recall, F1, ROC curve and AUC on a held-out test set.
    - Confusion matrix with configurable thresholds so clinics can prioritize:
    - High recall (catch more at-risk patients; acceptable more false positives), or
    - Higher precision (reduce unnecessary referrals).

    ## Team OBERISK

    ## References
    1.	World Health Organization. (2021, June 9). Obesity and overweight. World Health Organization. https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight
    2.	Barquera, S., Pedroza-Tobías, A., Medina, C., Hernández-Barrera, L., Bibbins-Domingo, K., Lozano, R., & Moran, A. E. (2020). Global overview of the epidemiology of obesity. Obesity Reviews, 21(S1), e13229. https://doi.org/10.1111/obr.13229
    3.	DeGregory, K. W., Kuiper, P., DeSilvio, T., Pleuss, J. D., Miller, R., Roginski, J. W., Fisher, C. B., Harness, D., Viswanath, S., Heymsfield, S. B., Dungan, I., & Thomas, D. M. (2018). A review of machine learning in obesity. Obesity Reviews, 19(5), 668–685. https://doi.org/10.1111/obr.12667
    4.	Díaz-Rodríguez, S., & Hernández, F. (2019). Estimation of obesity levels based on eating habits and physical condition. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition

"""
)