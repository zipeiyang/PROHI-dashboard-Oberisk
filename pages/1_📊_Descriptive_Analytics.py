import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from typing import Dict, List, Tuple

# Set page configuration
st.set_page_config(page_title="Descriptive Analytics", layout="wide")
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

def create_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Create age groups"""
    df_copy = df.copy()
    df_copy['age_group'] = pd.cut(
        df_copy['age'], 
        bins=[0, 18, 30, 45, 60, 100], 
        labels=['0-18', '19-30', '31-45', '46-60', '60+']
    )
    return df_copy

# Plotting functions
def create_figure(size: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """Create standardized matplotlib figure"""
    return plt.figure(figsize=size)

def plot_pie_distribution(df: pd.DataFrame) -> plt.Figure:
    """Plot pie chart of target variable distribution"""
    fig = create_figure((6, 6))
    counts = df["bmi_category"].value_counts().reindex(CATEGORY_ORDER)
    explode = (0.05, 0.1, 0, 0, 0, 0, 0)
    plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
           startangle=90, explode=explode, colors=COLORS, shadow=True)
    plt.title('Distribution of Obesity Levels', pad=20)
    return fig

def plot_categorical_distribution(df: pd.DataFrame, col: str, horizontal: bool = False, 
                                 figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """Plot distribution of categorical variable vs target variable"""
    fig, ax = plt.subplots(figsize=figsize)
    cross_tab = pd.crosstab(df[col], df["bmi_category"])
    cross_tab = cross_tab.reindex(columns=CATEGORY_ORDER, fill_value=0)
    
    if horizontal:
        cross_tab.plot(kind='barh', color=COLORS, width=0.8, ax=ax)
        ax.set_xlabel('Count')
        ax.set_ylabel(col)
    else:
        cross_tab.plot(kind='bar', color=COLORS, width=0.8, ax=ax)
        ax.set_xlabel(col)
        ax.set_ylabel('Count')        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    ax.legend(title='Obesity Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(f'Obesity Levels Distribution by {col}')
    plt.tight_layout()
    return fig

def plot_numeric_distribution(df: pd.DataFrame, x_col: str, y_col: str, plot_type: str = 'box') -> plt.Figure:
    """Plot distribution of numeric variables"""
    fig = create_figure((8, 6))
    
    if plot_type == 'box':
        sns.boxplot(x=x_col, y=y_col, data=df, order=CATEGORY_ORDER, palette=COLORS)
    elif plot_type == 'violin':
        sns.violinplot(x=x_col, y=y_col, data=df, order=CATEGORY_ORDER, palette=COLORS)
    
    plt.title(f'{y_col.capitalize()} Distribution by Obesity Level')
    plt.xlabel('Obesity Level')
    plt.ylabel(y_col.capitalize())
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_age_analysis(df: pd.DataFrame, plot_type: str = 'heatmap') -> plt.Figure:
    """Plot age-related analysis"""
    df_with_age_groups = create_age_groups(df)
    
    if plot_type == 'heatmap':
        fig = create_figure((8, 5))
        age_bmi_crosstab = pd.crosstab(df_with_age_groups['age_group'], df_with_age_groups["bmi_category"], normalize='index')
        sns.heatmap(age_bmi_crosstab, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Proportion', 'shrink': 0.8})
        plt.title('Obesity Levels Distribution by Age Group')
        plt.xlabel('Obesity Level')
        plt.ylabel('Age Group')
    elif plot_type == 'smallmultiples':
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()
        for i, category in enumerate(CATEGORY_ORDER):
            age_counts = df_with_age_groups[df_with_age_groups["bmi_category"] == category]['age_group'].value_counts().sort_index()
            axes[i].bar(age_counts.index.astype(str), age_counts.values, color=COLORS[i], alpha=0.8)
            axes[i].set_title(category, fontsize=10)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].set_ylabel('Count')
        for j in range(len(CATEGORY_ORDER), len(axes)):
            axes[j].set_visible(False)
        plt.suptitle('Obesity Levels Distribution by Age Group')
    
    plt.tight_layout()
    return fig

def plot_scatter_age_bmi(df: pd.DataFrame) -> plt.Figure:
    """Plot scatter plot of age vs BMI"""
    fig = create_figure((10, 8))
    df_copy = df.copy()
    df_copy['bmi'] = df_copy['weight'] / (df_copy['height'] ** 2)
    
    for i, category in enumerate(CATEGORY_ORDER):
        subset = df_copy[df_copy["bmi_category"] == category]
        plt.scatter(subset['age'], subset['bmi'], color=COLORS[i], label=category, alpha=0.7, s=60)
    
    for y in [18.5, 25, 30, 35, 40]:
        plt.axhline(y=y, color='gray', linestyle='--', alpha=0.7)
    
    plt.title('Age vs BMI by Obesity Level')
    plt.xlabel('Age')
    plt.ylabel('BMI')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str]) -> plt.Figure:
    """Plot correlation heatmap for numeric variables"""
    fig = create_figure((8, 6))
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', center=0)
    plt.title('Correlation Heatmap of Numerical Variables')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_cramers_heatmap(df: pd.DataFrame, categorical_cols: List[str]) -> plt.Figure:
    """Plot Cramér's V association heatmap for categorical variables"""
    cramer_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols, dtype=float)
    
    for i in categorical_cols:
        for j in categorical_cols:
            try:
                cramer_matrix.loc[i, j] = calculate_cramers_v(df[i], df[j])
            except Exception:
                cramer_matrix.loc[i, j] = 0.0
    
    fig = create_figure((10, 7))
    sns.heatmap(cramer_matrix.astype(float), annot=True, cmap='Blues', fmt='.2f', 
               square=True, cbar_kws={'shrink': .8})
    plt.title("Association Heatmap of Categorical Variables (Cramer's V)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# Page rendering functions
def render_summary_page(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]):
    """Render summary page"""
    st.header("Introduction")
    n_rows, n_cols = df.shape
    st.write(
        f"Our dataset includes **{n_rows} data points** and **{n_cols} features** "
        "for the estimation of obesity levels in individuals from the countries of "
        "Mexico, Peru and Colombia, based on their eating habits and physical condition."
    )
    
    # Obesity level distribution
    st.header("Obesity Level Distribution")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_level = st.radio(
            "Please select an Obesity Level",
            CATEGORY_ORDER,
            key="obesity_level_selector"
        )
    
    counts = df["bmi_category"].value_counts()
    explode = [0.1 if lvl == selected_level else 0 for lvl in CATEGORY_ORDER]
    
    with col2:
        fig = create_figure((6, 6))
        plt.pie(
            counts[CATEGORY_ORDER],
            labels=CATEGORY_ORDER,
            autopct="%1.1f%%",
            explode=explode,
            shadow=True,
            startangle=90,
            colors=COLORS
        )
        plt.title("Obesity Level Distribution")
        st.metric(label=f"Count of {selected_level}", value=int(counts[selected_level]))
        st.pyplot(fig)
    
    # Numeric variable statistics
    st.header("Numerical Variables")
    stats_df = df[numeric_cols].describe().T
    st.dataframe(stats_df, use_container_width=True)
    
    # Categorical variable distribution
    st.header("Categorical Variables")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_cat = st.selectbox(
            "Please choose a categorical variable",
            categorical_cols,
            key="cat_var_selector"
        )
        
        freq_table = pd.DataFrame()
        
        if st.checkbox("Show frequency table", key="freq_table_checkbox"):
            freq_table = df[selected_cat].value_counts(normalize=True).reset_index()
            freq_table.columns = [selected_cat, "Proportion"]
            st.dataframe(freq_table)
    
    with col2:
        fig = create_figure((6, 4))
        sns.countplot(data=df, x=selected_cat, order=df[selected_cat].value_counts().index, palette="pastel")
        plt.title(f"Distribution of {selected_cat}")
        plt.xticks(rotation=45)
        st.pyplot(fig)

def render_single_features_page(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]):
    """Render single feature analysis page"""
    st.header("Drill-Down: Variable Explorer")
    st.markdown("""
    **Understand how each factor correlates with obesity levels.**  
    This view allows you to focus on one variable at a time, examining the specific relationship between individual lifestyle or physical factors 
    and obesity classification through tailored visualizations. Select any variable below to begin your analysis.
    """)
    
    left_col, right_col = st.columns([1, 3])
    
    with left_col:
        var = st.radio("Please select a feature", options=VARIABLE_FEATURES, index=0, key="feature_selector")
        st.markdown("---")
        st.markdown("Legend:")
        st.write(", ".join(CATEGORY_ORDER))
        st.markdown("---")
    
    with right_col:
        st.markdown(f"#### Plots for **{var}**")
        
        # Select appropriate charts based on variable type
        if var in numeric_cols:
            if var == 'age':
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Distribution", "Heatmap", "Small Multiples", "Scatter with BMI", "Correlation"])
                with tab1:
                    fig = plot_numeric_distribution(df, "bmi_category", var, 'box')
                    st.pyplot(fig)
                with tab2:
                    fig = plot_age_analysis(df, 'heatmap')
                    st.pyplot(fig)
                with tab3:
                    fig = plot_age_analysis(df, 'smallmultiples')
                    st.pyplot(fig)
                with tab4:
                    fig = plot_scatter_age_bmi(df)
                    st.pyplot(fig)
                with tab5:
                    fig = plot_correlation_heatmap(df, numeric_cols)
                    st.pyplot(fig)
            else:
                tab1, tab2 = st.tabs(["Distribution", "Correlation"])
                with tab1:
                    plot_type = 'violin' if var in ['veggie_per_meal', 'water_daily'] else 'box'
                    fig = plot_numeric_distribution(df, "bmi_category", var, plot_type)
                    st.pyplot(fig)
                with tab2:
                    fig = plot_correlation_heatmap(df, numeric_cols)
                    st.pyplot(fig)
        
        elif var in categorical_cols:
            # Create two tabs: one for distribution plot, one for Cramer's V heatmap
            tab1, tab2 = st.tabs(["Distribution", "Cramer's V"])
            
            with tab1:
                horizontal = (var == 'transport')
                cross_tab = pd.crosstab(df[var], df["bmi_category"])
                cross_tab = cross_tab.reindex(columns=CATEGORY_ORDER, fill_value=0)
                try:
                    fig = plot_categorical_distribution(df, var, horizontal)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error plotting: {e}")
            
            with tab2:
                with st.spinner("Calculating Cramer's V..."):
                    fig = plot_cramers_heatmap(df, categorical_cols)
                    st.pyplot(fig)

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
    
    # Sidebar navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Data Overview"

    st.sidebar.markdown("#### Navigation")
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("Data Overview", use_container_width=True, 
                     type="primary" if st.session_state.current_page == "Data Overview" else "secondary"):
            st.session_state.current_page = "Data Overview"
            st.rerun()

    with col2:
        if st.button("Variable Explorer", use_container_width=True,
                     type="primary" if st.session_state.current_page == "Variable Explorer" else "secondary"):
            st.session_state.current_page = "Variable Explorer"
            st.rerun()

    subpage = st.session_state.current_page
    
    # Main content area
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    if subpage == "Data Overview":
        render_summary_page(df, numeric_cols, categorical_cols)
    elif subpage == "Variable Explorer":
        render_single_features_page(df, numeric_cols, categorical_cols)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:60px"></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()