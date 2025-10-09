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
st.set_page_config(page_title="Obesity Risk Factors", layout="wide")
st.logo("./assets/oberisk_sidebar.png", size="large", icon_image="./assets/oberisk_logo_icon.png")

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
    **Explore the complex relationships between various factors and obesity risk.**  
    This section uses statistical methods to identify which factors have the strongest influence on obesity levels.
    """)
    
    # Add a friendly introduction with explanations
    with st.expander("💡 **How to use this page**", expanded=True):
        st.markdown("""
        This page helps you understand which factors matter most for obesity risk. Here's how to use it:
        
        - **Choose an analysis type** - each shows different insights
        - **Adjust the slider** to see more or fewer factors
        - **Click the info icons** 🔍 next to each analysis for explanations
        """)
    
    # Create tabs for different analysis views with user-friendly names
    analysis_type = st.radio(
        "**Choose what you want to explore:**",
        ["Key Factors by Obesity Level", "Most Influential Factors", "Data Patterns Overview"],
        horizontal=True,
        key="analysis_selector",
        help="Select what type of insights you'd like to see"
    )
    
    # Add a slider to control number of features to display
    total_features = len(numeric_cols) + len(categorical_cols)
    n_features = st.slider(
        "**Number of top factors to show:**",
        min_value=5,
        max_value=min(20, total_features),
        value=min(10, total_features),
        key="feature_slider",
        help="Show more or fewer factors in the charts"
    )
    
    # Add progress bar and status message
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Execute diagnostic analysis
    status_text.text("🔬 Analyzing your data...")
    progress_bar.progress(25)
    
    results = perform_diagnostic_analysis(df, numeric_cols, categorical_cols, target)
    
    progress_bar.progress(75)
    status_text.text("📈 Preparing your results...")
    
    # Display different analysis results based on user selection
    if analysis_type == "Key Factors by Obesity Level":
        st.subheader("Key Factors for Each Obesity Level")
        
        with st.expander("🔍 **What does this analysis show?**", expanded=False):
            st.markdown("""
            **In simple terms:** This shows which factors are most strongly linked to **each specific obesity level**.
            
            **How to read the chart:**
            - **Longer bars** = stronger influence on that obesity level
            - **Blue bars** = numerical factors (like age, weight)
            - **Green bars** = categorical factors (like family history, smoking)
            - **Positive values** = factor increases risk of that obesity level
            - **Negative values** = factor decreases risk of that obesity level
            
            **Example:** If "physical activity" has a long negative bar for "Obesity Type III", it means more physical activity strongly reduces the risk of severe obesity.
            """)
        
        # Let user select which class to view
        selected_class = st.selectbox(
            "**Select obesity level to explore:**",
            results['logreg_coefficients'].index.tolist(),
            key="class_selector",
            help="Choose which obesity level you want to understand better"
        )
        
        # Display coefficients for selected class
        coefs = results['logreg_coefficients'].loc[selected_class]
        top_coefs = coefs.abs().sort_values(ascending=False).head(n_features)
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#A5B8D6' if f in numeric_cols else '#A5C2A7' for f in top_coefs.index]
        
        # Use actual coefficient values (not absolute) to show direction
        actual_values = [coefs[f] for f in top_coefs.index]
        bars = ax.barh(range(len(top_coefs)), actual_values, color=colors)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, actual_values)):
            ax.text(bar.get_width() + (0.01 if value >= 0 else -0.01), 
                   bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', 
                   ha='left' if value >= 0 else 'right', 
                   va='center', fontsize=8)
        
        ax.set_yticks(range(len(top_coefs)))
        ax.set_yticklabels(top_coefs.index)
        ax.invert_yaxis()
        ax.set_xlabel('Influence Strength (Positive = Increases Risk, Negative = Decreases Risk)')
        ax.set_title(f'Top Factors Influencing {selected_class}')
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        
        legend_elements = [Patch(facecolor='#A5B8D6', label='Numerical Factors'), 
                         Patch(facecolor='#A5C2A7', label='Lifestyle & Background Factors')]
        ax.legend(handles=legend_elements, loc='lower right')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Interpretation help
        st.info(f"💡 **Interpretation help for {selected_class}:** Factors on the **right side** increase risk, while factors on the **left side** decrease risk. The longer the bar, the stronger the effect.")
        
    elif analysis_type == "Most Influential Factors":
        st.subheader("Overall Most Influential Factors")
        
        with st.expander("🔍 **What does this analysis show?**", expanded=False):
            st.markdown("""
            **In simple terms:** This ranks all factors by their **overall importance** across all obesity levels.
            
            **How to read the chart:**
            - **Longer bars** = more important factors overall
            - **Blue bars** = numerical factors (like age, meals per day)
            - **Green bars** = categorical factors (like family history, smoking)
            - The score shows how strongly each factor is related to obesity in general
            
            **This helps you answer:** "Which feature changes would have the biggest impact on obesity risk?"
            """)
        
        # Display association strength
        eta_df = results['association_strength']
        top_features = eta_df.head(n_features)
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#A5B8D6' if c in numeric_cols else '#A5C2A7' for c in top_features.index]
        bars = ax.barh(range(len(top_features)), top_features['Association_Strength'], color=colors)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_features['Association_Strength'])):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', ha='left', va='center', fontsize=8)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features.index)
        ax.invert_yaxis()
        ax.set_xlabel('Overall Importance Score')
        ax.set_title(f'Top {n_features} Most Important Factors for Obesity Risk')
        
        legend_elements = [Patch(facecolor='#A5B8D6', label='Numerical Factors'), 
                         Patch(facecolor='#A5C2A7', label='Lifestyle & Background Factors')]
        ax.legend(handles=legend_elements, loc='lower right')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Practical implications
        if len(top_features) > 0:
            top_factor = top_features.index[0]
            top_score = top_features.iloc[0, 0]
            st.info(f"💡 **Key Insight:** **{top_factor}** is the most important factor (score: {top_score:.3f}). Focusing on this could have the biggest impact on obesity risk.")
        
    elif analysis_type == "Data Patterns Overview":
        st.subheader("Data Patterns Overview")
        
        with st.expander("🔍 **What does this analysis show?**", expanded=False):
            st.markdown("""
            **In simple terms:** This creates a "map" of all the people in the dataset, showing how different obesity levels group together based on their factors.
            
            **How to read the chart:**
            - **Each dot** = one person in the dataset
            - **Colors** = different obesity levels
            - **Similar positions** = similar feature patterns
            - **Grouped dots** = people with similar obesity levels and features
            - **Arrows** = show which factors influence the pattern the most
            
            **This helps you see:** "Do people with similar factors tend to have similar obesity levels?"
            """)
        
        if 'pca' in results:
            pca_results = results['pca']
            X_pca = pca_results['X_pca']
            feature_weights = pca_results['feature_weights']
            explained_variance = pca_results['explained_variance_ratio']
            
            # Let user select which classes to highlight
            selected_classes = st.multiselect(
                "**Focus on specific obesity levels:**",
                results['label_encoder'].classes_.tolist(),
                default=results['label_encoder'].classes_.tolist()[:3],
                key="pca_class_selector",
                help="Select which obesity levels to highlight (others will be faded)"
            )
            
            # Create PCA scatter plot
            fig, ax = plt.subplots(figsize=(10, 8))
            le = results['label_encoder']
            
            # Plot points for each class
            for i, class_name in enumerate(le.classes_):
                mask = (le.transform(df[target]) == i)
                alpha = 0.8 if class_name in selected_classes else 0.2
                size = 60 if class_name in selected_classes else 30
                label = class_name if class_name in selected_classes else None
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                          label=label, alpha=alpha, s=size, c=COLORS[i % len(COLORS)])
            
            # Draw arrows for important variables
            important_vars = ['height', 'weight', 'age', 'technological_devices', 'physical_activity']
            for var in important_vars:
                if var in feature_weights.columns:
                    pc1_load = feature_weights.loc['PC1', var]
                    pc2_load = feature_weights.loc['PC2', var]
                    ax.arrow(0, 0, pc1_load*3, pc2_load*3, 
                            head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.7)
                    ax.text(pc1_load*3.2, pc2_load*3.2, var, color='red', fontsize=9)
            
            ax.set_xlabel(f'Pattern Dimension 1 ({explained_variance[0]:.1%} of patterns)')
            ax.set_ylabel(f'Pattern Dimension 2 ({explained_variance[1]:.1%} of patterns)')
            ax.set_title('Lifestyle Patterns and Obesity Levels')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Obesity Levels")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Pattern interpretation
            st.info("💡 **Pattern interpretation:** When dots of the same color cluster together, it means people with that obesity level have similar lifestyles. Arrows show which factors create these patterns.")
            
            # Display top loadings for each principal component
            st.write("**Key factors shaping the patterns:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Dimension 1** - Main pattern drivers:")
                pc1_weights = feature_weights.loc['PC1'].abs().sort_values(ascending=False).head(n_features)
                st.dataframe(pd.DataFrame({
                    'Factor': pc1_weights.index,
                    'Influence': pc1_weights.values
                }).reset_index(drop=True))
            
            with col2:
                st.write("**Dimension 2** - Secondary pattern drivers:")
                pc2_weights = feature_weights.loc['PC2'].abs().sort_values(ascending=False).head(n_features)
                st.dataframe(pd.DataFrame({
                    'Factor': pc2_weights.index,
                    'Influence': pc2_weights.values
                }).reset_index(drop=True))
                
        elif 'pca_error' in results:
            st.error(f"Pattern analysis failed: {results['pca_error']}")
    
    # Complete progress bar
    progress_bar.progress(100)
    status_text.text("✅ Analysis complete!")
    
    # Add summary and practical takeaways
    with st.expander(" **Summary**", expanded=False):
        st.markdown("""
        - The most influential factors suggest whose changes could have the biggest impact
        - Different factors matter for different obesity levels  

        **Remember:** These are statistical patterns from group data. Individual results may vary, and professional medical advice should always be sought for health decisions.
        """)
    
    # Add some additional statistics in a compact way
    with st.expander(" **Technical Details**", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Factors Analyzed", f"{len(numeric_cols) + len(categorical_cols)}")
        with col2:
            st.metric("Numerical Factors", f"{len(numeric_cols)}")
        with col3:
            st.metric("Lifestyle Factors", f"{len(categorical_cols)}")
        
        if 'pca' in results:
            st.write(f"**Pattern Coverage:** This analysis captures {sum(results['pca']['explained_variance_ratio']):.1%} of the lifestyle patterns in the data.")

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