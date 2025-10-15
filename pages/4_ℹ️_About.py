import streamlit as st

st.set_page_config(
    page_title="ℹ️ About - Oberisk",
    page_icon="./assets/oberisk_icon.svg",
)

st.logo("./assets/oberisk_sidebar.png", size="large", icon_image="./assets/oberisk_logo_icon.png")

# # Page information

st.write("# About")

st.image("./assets/oberisk_side.png", use_container_width=False, width=300)




st.markdown(
"""

    ## The Project
    Obesity rates are on the rise globally, leading to increased cases of chronic diseases and higher healthcare costs. There is a growing need for early identification and prevention strategies. However, currently personalized tools that uses both anthropometric data (age, sex) and lifestyle factors (e.g., physical activity, diet) to predict an individual's risk of obesity are not widely available. Therefore, the general public requires clear, visualized statistics to understand obesity trends and how their own lifestyle factors contirbute to their obesity risk, which can support and encourage preventive visits to health professionals such as dieticians. 
    OBERISK is a desktop‑based dashboard that enables individuals to estimate their obesity risk using a machine learning model trained on demographic, family‑history, and lifestyle features. The initial demonstration will run in Streamlit. This tool was developed in partnership with Stockholm Region and Karolinska Institutet as part of their wellness initiative to help promote healthier lifestyles for Stockholm Lan's population.

    What OBERISK offers:
    - Fast risk estimates from the comfort of your own home. No appointment needed! No waiting time!
    - Visualized feature effects to support easier understanding.
    - A simple, easy-to-access tools encouraging individuals to evaluate and monitor their risk factors, facilitating a proactive approach to one's health and wellbeing.


    ## The Dataset
    The dataset combines information from individuals in Mexico, Peru, and Colombia. The dataset captures key attributes related to eating habits, physical activity, and health status (4). It consists of 2,111 records and 17 features, with each entry labelled according to the NObesity class variable. This label categorizes individuals into one of seven obesity levels: Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, and Obesity Type III. Notably, 23% of the data was collected directly from users via a web platform, while the remaining 77% was synthetically generated using the SMOTE filter within the Weka tool, ensuring a more balanced distribution across classes.
    
    For modeling, we converted the 7-class label to a binary outcome:
    - No risk (0): Insufficient/Normal
    - At risk (1): Overweight I–II and Obesity I–III
    
"""

)

st.warning(
"""
     ⚠️ **Important caveats about the dataset labels**

    The "obesity level" was derived directly from height and weight (BMI categories), which introduced a **label-definition leakage**: models that see height/weight will (correctly) learn BMI boundaries, and lifestyle features will contribute less to predictions of current status.    
        
    Therefore, to reduce this the Oberisk prediction model has been trained to predict obesity **solely on lifestyle factors, family-history, as well as age and sex**. Weight and height features were dropped from the prediction model to reduce the label-definition leakage, giving you a model that predicts obesity risk using lifestyle factors and not one that learns BMI boundaries from weight and height variables.
"""

)

st.markdown(
"""
    ## Model Justification
    **Why binary classification?**

    - Clinically, the key fork in decisions is "screen and counsel intensively vs. routine prevention." Binarizing (Normal/Underweight vs. Overweight/Obesity) aligns with that workflow and improves class balance for evaluation and threshold tuning.
    - It also simplifies calibration (well-calibrated risk probabilities) and operating point selection (e.g., high-recall screening).
    - Additionally, it makes it simpler for the individuals to understand the model's output and what next steps they can take for their health and wellbeing.

    **Why Random Forest?**

    We evaluated Logistic Regression, Decision Tree, Random Forest, Support Vector Machines, K-Nearest Neighbor, XGBoost, and LightGBM using a consistent preprocessing pipeline (scaling numeric features; one-hot encoding categoricals) and stratified train/test splits. We selected **Random Forest** as the primary model because it:

    - Robust performance: Demonstrated strong predictive accuracy and generalization across diverse patient profiles
    - Handles complex interactions: Naturally captures non-linear relationships and feature interactions between lifestyle factors, family history, and anthropometric measurements
    - Feature importance: Provides interpretable feature importance scores that align with clinical domain knowledge
    - Robust to outliers and noise: Handles mixed data types effectively without extensive preprocessing
    - Balanced performance: Consistently achieved strong results across multiple evaluation metrics while maintaining computational efficiency

    **Model interpretability**

    We employed multiple interpretability approaches to ensure model transparency and clinical relevance:

    SHAP Analysis (Model Development Phase):
    - Global feature importance: Identified key drivers of obesity risk across the entire population, including family history, physical activity levels, and dietary patterns
    - Local explanations: Provided individual-level insights into how each feature contributes to specific predictions
    - Interaction effects: Revealed complex relationships between lifestyle factors and their combined impact on obesity risk
    - Model validation: Confirmed that the model's decision-making aligns with established clinical knowledge

    Counterfactual Analysis (Clinical Application):
    - Actionable insights: Allows users to directly explore "what-if" scenarios by modifying lifestyle factors in real-time
    - Personalized recommendations: Shows how specific behavior changes might impact individual obesity risk
    - Interactive learning: Engages users in understanding the relationship between their choices and health outcomes
    - Clinical relevance: Mimics real-world counseling conversations where providers discuss potential benefits of lifestyle modifications

    **Evaluation points**

    For each model we report:
    - Accuracy, Precision, Recall, F1, ROC curve and AUC on a held-out test set
    - Confusion matrix with configurable thresholds
    - SHAP analysis for model interpretability and feature importance validation
    - Counterfactual analysis for personalized risk modification guidance

    ## Team OBERISK
    *Alex Anna Matija Meilia Zipei* | info@oberisk.se

    ## References
    1.	World Health Organization. (2021, June 9). Obesity and overweight. World Health Organization. https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight
    2.	Barquera, S., Pedroza-Tobías, A., Medina, C., Hernández-Barrera, L., Bibbins-Domingo, K., Lozano, R., & Moran, A. E. (2020). Global overview of the epidemiology of obesity. Obesity Reviews, 21(S1), e13229. https://doi.org/10.1111/obr.13229
    3.	DeGregory, K. W., Kuiper, P., DeSilvio, T., Pleuss, J. D., Miller, R., Roginski, J. W., Fisher, C. B., Harness, D., Viswanath, S., Heymsfield, S. B., Dungan, I., & Thomas, D. M. (2018). A review of machine learning in obesity. Obesity Reviews, 19(5), 668–685. https://doi.org/10.1111/obr.12667
    4.	Díaz-Rodríguez, S., & Hernández, F. (2019). Estimation of obesity levels based on eating habits and physical condition. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition
    5.  Palechor, F.M., & Manotas, A.D. Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico. Data in Brief. 2019(25). https://doi.org/10.1016/j.dib.2019.104344
    6.  World Health Organization. Body mass index (BMI) [Internet]. Geneva: World Health Organization. 2025 [cited 2025 Oct 12]. Available from: https://www.who.int/data/gho/data/themes/topics/topic-details/GHO/body-mass-index
    7.  Rubino, F., Cummings, D. E., Eckel, R. H., Cohen, R. V., Wilding, J. P. H., Brown, W. A., Stanford, F. C., Batterham, R. L., Farooqi, I. S., Farpour-Lambert, N. J., le Roux, C. W., Sattar, N., Baur, L. A., Morrison, K. M., Misra, A., Kadowaki, T., Tham, K. W., Sumithran, P., Garvey, W. T., … Mingrone, G. (2025). Definition and diagnostic criteria of clinical obesity. The Lancet. Diabetes & Endocrinology, 13(3), 221–262. https://doi.org/10.1016/S2213-8587(24)00316-4


"""
)