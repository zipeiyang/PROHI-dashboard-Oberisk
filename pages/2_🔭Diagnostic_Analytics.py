import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
from typing import Dict, List, Tuple, Any
import time

# Set page configuration
st.set_page_config(page_title="Diagnostic Analytics", layout="wide")
st.logo("./assets/oberisk_sidebar.png", size="large")

DEFAULT_CSV_PATH = "./assets/ObesityDataSet_raw_and_data_sinthetic.csv"

# Define constants
CATEGORY_ORDER = [
    'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
    'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
]

COLORS = ['#F7F0D4', '#A5C2A7', '#C3D2E0', '#A5B8D6', '#E8C4B8', '#D49A89', '#B36A5E']

VARIABLE_FEATURES = [
    'age', 'height', 'weight', 'veggie_per_meal', 'meals_daily', 'water_daily',
    'physical_activity', 'technological_devices',
    'sex', 'family_history', 'often_high_calorie_intake', 'freq_snack',
    'smoking', 'monitor_calorie', 'freq_alcohol', 'transport'
]

# Set matplotlib default parameters
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 100
})

# Load data
@st.cache_data
def load_data(path: str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Load and preprocess data"""
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df = df.rename(columns={
            "Age": "age",
            "Gender": "sex",
            "Weight": "weight",
            "Height": "height",
            "family_history_with_overweight": "family_history",
            "FAVC": "often_high_calorie_intake",
            "FCVC": "veggie_per_meal",
            "NCP": "meals_daily",
            "CAEC": "freq_snack",
            "SMOKE": "smoking",
            "CH2O": "water_daily",
            "SCC": "monitor_calorie",
            "FAF": "physical_activity",
            "TUE": "technological_devices",
            "CALC": "freq_alcohol",
            "MTRANS": "transport",
            "NObeyesdad": "bmi_category"
        })
        
        # Ensure correct data types
        numeric_cols = ['age', 'height', 'weight', 'veggie_per_meal', 'meals_daily', 
                       'water_daily', 'physical_activity', 'technological_devices']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Failed to load dataset from '{path}'. Error: {e}")
        st.stop()

# Utility functions
def calculate_cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Calculate Cramér's V coefficient between two categorical variables"""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    
    if n == 0:
        return 0.0
        
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    
    if min((kcorr-1), (rcorr-1)) == 0:
        return 0.0
        
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# Diagnostic analysis functions
@st.cache_data(show_spinner=False)
def perform_diagnostic_analysis(_df: pd.DataFrame, numeric_cols: List[str], 
                               categorical_cols: List[str], target: str) -> Dict[str, Any]:
    """Perform diagnostic analysis and return results"""
    results = {}
    df_analysis = _df[numeric_cols + categorical_cols + [target]].copy()
    
    # Handle missing values
    for col in numeric_cols:
        df_analysis[col] = df_analysis[col].fillna(df_analysis[col].median())
    for col in categorical_cols:
        df_analysis[col] = df_analysis[col].fillna(
            df_analysis[col].mode()[0] if not df_analysis[col].mode().empty else "Missing"
        )
    
    X = df_analysis[numeric_cols + categorical_cols]
    y = df_analysis[target]
    
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Preprocessing pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])
    
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names
    numeric_features = numeric_cols
    categorical_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
    all_features = list(numeric_features) + list(categorical_features)
    
    # Fit logistic regression model
    logreg = LogisticRegression(multi_class='multinomial', max_iter=1000, 
                               penalty='l2', C=1.0, random_state=42)
    logreg.fit(X_processed, y_encoded)
    
    coefficients = logreg.coef_
    coef_df = pd.DataFrame(coefficients, columns=all_features, index=le.classes_)
    results['logreg_coefficients'] = coef_df
    
    # Calculate association strength
    eta_squared_results = {}
    
    # Numeric variables -> eta-squared (ANOVA)
    for col in numeric_cols:
        clean = df_analysis[[col, target]].dropna()
        if len(clean) > 0:
            overall_mean = clean[col].mean()
            ss_total = np.sum((clean[col] - overall_mean) ** 2)
            ss_between = 0.0
            for category in CATEGORY_ORDER:
                group = clean[clean[target] == category][col]
                if len(group) > 0:
                    ss_between += len(group) * (group.mean() - overall_mean) ** 2
            eta_sq = ss_between / ss_total if ss_total != 0 else 0
            eta_squared_results[col] = eta_sq
        else:
            eta_squared_results[col] = 0.0
    
    # Categorical variables -> Cramér's V
    for col in categorical_cols:
        clean = df_analysis[[col, target]].dropna()
        if len(clean) > 0:
            eta_squared_results[col] = calculate_cramers_v(clean[col], clean[target])
        else:
            eta_squared_results[col] = 0.0
    
    eta_df = pd.DataFrame.from_dict(eta_squared_results, orient='index', columns=['Association_Strength'])
    eta_df = eta_df.sort_values('Association_Strength', ascending=False)
    results['association_strength'] = eta_df
    
    # PCA analysis
    try:
        X_for_pca = X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_for_pca)
        feature_weights = pd.DataFrame(pca.components_, columns=all_features, index=['PC1', 'PC2'])
        
        results['pca'] = {
            'X_pca': X_pca,
            'feature_weights': feature_weights,
            'explained_variance_ratio': pca.explained_variance_ratio_
        }
    except Exception as e:
        results['pca_error'] = str(e)
    
    results['label_encoder'] = le
    return results

def render_diagnostic_page(df: pd.DataFrame, numeric_cols: List[str], 
                          categorical_cols: List[str], target: str):
    """Render diagnostic analysis page"""
    st.header("Comprehensive Analysis: Multi-Feature Diagnostics")
    st.markdown("""
    **Explore complex relationships and interactions between multiple variables.**  
    This section provides advanced analytical techniques including diagnostic modeling
    and dimensionality reduction to uncover patterns and key drivers behind obesity levels. 
    Select any diagnostic method below to begin your analysis.
    """)
    
    # Create tabs for different analysis views
    analysis_type = st.radio(
        "Select Analysis Type:",
        ["Logistic Regression Coefficients", "Feature Association Strength", "PCA Visualization"],
        horizontal=True,
        key="analysis_selector"
    )
    
    # Add a slider to control number of features to display
    total_features = len(numeric_cols) + len(categorical_cols)
    n_features = st.slider(
        "Number of top features to display:",
        min_value=5,
        max_value=min(20, total_features),
        value=min(10, total_features),
        key="feature_slider"
    )
    
    # Add progress bar and status message
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Execute diagnostic analysis
    status_text.text("Running diagnostic analysis...")
    progress_bar.progress(25)
    
    results = perform_diagnostic_analysis(df, numeric_cols, categorical_cols, target)
    
    progress_bar.progress(75)
    status_text.text("Analysis complete! Displaying results...")
    
    # Display different analysis results based on user selection
    if analysis_type == "Logistic Regression Coefficients":
        st.subheader("Multiclass Logistic Regression Coefficients")
        
        # Let user select which class to view
        selected_class = st.selectbox(
            "Select obesity class to view:",
            results['logreg_coefficients'].index.tolist(),
            key="class_selector"
        )
        
        # Display coefficients for selected class
        coefs = results['logreg_coefficients'].loc[selected_class]
        top_coefs = coefs.abs().sort_values(ascending=False).head(n_features)
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#A5B8D6' if f in numeric_cols else '#A5C2A7' for f in top_coefs.index]
        ax.barh(range(len(top_coefs)), top_coefs.values, color=colors)
        ax.set_yticks(range(len(top_coefs)))
        ax.set_yticklabels(top_coefs.index)
        ax.invert_yaxis()
        ax.set_xlabel('Coefficient magnitude (abs)')
        ax.set_title(f'Top {n_features} features for {selected_class}')
        legend_elements = [Patch(facecolor='#A5B8D6', label='Numerical'), 
                         Patch(facecolor='#A5C2A7', label='Categorical')]
        ax.legend(handles=legend_elements, loc='lower right')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display data table
        st.dataframe(pd.DataFrame({
            'Feature': top_coefs.index,
            'Coefficient': [coefs[f] for f in top_coefs.index],
            'Absolute Value': top_coefs.values
        }).reset_index(drop=True))
        
    elif analysis_type == "Feature Association Strength":
        st.subheader("Feature Association Strength with Obesity Levels")
        
        # Display association strength
        eta_df = results['association_strength']
        top_features = eta_df.head(n_features)
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#A5B8D6' if c in numeric_cols else '#A5C2A7' for c in top_features.index]
        ax.barh(range(len(top_features)), top_features['Association_Strength'], color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features.index)
        ax.invert_yaxis()
        ax.set_xlabel('Association Strength')
        ax.set_title(f'Top {n_features} features by association strength')
        legend_elements = [Patch(facecolor='#A5B8D6', label='Numeric (Eta^2)'), 
                         Patch(facecolor='#A5C2A7', label="Categorical (Cramer's V)")]
        ax.legend(handles=legend_elements, loc='lower right')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display data table
        st.dataframe(top_features)
        
    elif analysis_type == "PCA Visualization":
        st.subheader("Principal Component Analysis (PCA) Visualization")
        
        if 'pca' in results:
            pca_results = results['pca']
            X_pca = pca_results['X_pca']
            feature_weights = pca_results['feature_weights']
            explained_variance = pca_results['explained_variance_ratio']
            
            # Let user select which classes to highlight
            selected_classes = st.multiselect(
                "Select classes to highlight:",
                results['label_encoder'].classes_.tolist(),
                default=results['label_encoder'].classes_.tolist()[:3],
                key="pca_class_selector"
            )
            
            # Create PCA scatter plot
            fig, ax = plt.subplots(figsize=(10, 8))
            le = results['label_encoder']
            
            # Plot points for each class
            for i, class_name in enumerate(le.classes_):
                mask = (le.transform(df[target]) == i)
                alpha = 0.8 if class_name in selected_classes else 0.2
                size = 60 if class_name in selected_classes else 30
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                          label=class_name, alpha=alpha, s=size)
            
            # Draw arrows for important variables
            important_vars = ['height', 'weight', 'age', 'technological_devices', 'physical_activity']
            for var in important_vars:
                if var in feature_weights.columns:
                    pc1_load = feature_weights.loc['PC1', var]
                    pc2_load = feature_weights.loc['PC2', var]
                    ax.arrow(0, 0, pc1_load*3, pc2_load*3, 
                            head_width=0.05, head_length=0.05, fc='red', ec='red')
                    ax.text(pc1_load*3.2, pc2_load*3.2, var, color='red')
            
            ax.set_xlabel(f'PC1 ({explained_variance[0]:.1%} explained variance)')
            ax.set_ylabel(f'PC2 ({explained_variance[1]:.1%} explained variance)')
            ax.set_title('PCA: 2D projection of observations')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display top loadings for each principal component
            st.write("Top feature loadings per principal component:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("PC1 top features:")
                pc1_weights = feature_weights.loc['PC1'].abs().sort_values(ascending=False).head(n_features)
                st.dataframe(pd.DataFrame({
                    'Feature': pc1_weights.index,
                    'Weight': pc1_weights.values
                }).reset_index(drop=True))
            
            with col2:
                st.write("PC2 top features:")
                pc2_weights = feature_weights.loc['PC2'].abs().sort_values(ascending=False).head(n_features)
                st.dataframe(pd.DataFrame({
                    'Feature': pc2_weights.index,
                    'Weight': pc2_weights.values
                }).reset_index(drop=True))
                
        elif 'pca_error' in results:
            st.error(f"PCA failed: {results['pca_error']}")
    
    # Complete progress bar
    progress_bar.progress(100)
    status_text.text("Analysis complete!")
    
    # Add some additional statistics
    with st.expander("View additional statistics"):
        st.write(f"Total features analyzed: {len(numeric_cols) + len(categorical_cols)}")
        st.write(f"Numerical features: {len(numeric_cols)}")
        st.write(f"Categorical features: {len(categorical_cols)}")
        st.write(f"Total variance explained by PCA: {sum(results['pca']['explained_variance_ratio']):.2%}")

# Main application
def main():
    # Load data
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Failed to load dataset. Make sure '{DEFAULT_CSV_PATH}' exists. Error: {e}")
        return
    
    target = "bmi_category"
    
    # Identify numeric and categorical variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.drop(target).tolist()
    
    # Ensure all variables are in the correct lists
    for col in VARIABLE_FEATURES:
        if col not in numeric_cols and col not in categorical_cols:
            st.warning(f"Column '{col}' not found in numeric or categorical columns. It might have an unexpected data type.")
    
    # Render diagnostic page
    render_diagnostic_page(df, numeric_cols, categorical_cols, target)

if __name__ == "__main__":
    main()