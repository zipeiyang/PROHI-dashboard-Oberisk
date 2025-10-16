import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from typing import Dict, List, Tuple

# Set page configuration
st.set_page_config(page_title="Obesity Trends", layout="wide")
st.logo("./assets/oberisk_sidebar.png", size="large", icon_image="./assets/oberisk_logo_icon.png")

DEFAULT_CSV_PATH = "./assets/ObesityDataSet_raw_and_data_sinthetic.csv"

# Define constants
CATEGORY_ORDER = [
    'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
    'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
]

#Display labels 
CATEGORY_LABELS = {
    "Insufficient_Weight": "Underweight",
    "Normal_Weight": "Normal weight",
    "Overweight_Level_I": "Overweight I",
    "Overweight_Level_II": "Overweight II",
    "Obesity_Type_I": "Obesity I",
    "Obesity_Type_II": "Obesity II",
    "Obesity_Type_III": "Obesity III",
}

VAR_LABELS = {
    "age": "Age (years)",
    "height": "Height (m)",
    "weight": "Weight (kg)",
    "veggie_per_meal": "Vegetables per meal",
    "meals_daily": "Meals per day",
    "water_daily": "Water per day (L)",
    "physical_activity": "Physical activity (days/week)",
    "technological_devices": "Technology use (hours/day)",
    "sex": "Sex",
    "family_history": "Family history of overweight",
    "often_high_calorie_intake": "Often high-calorie intake",
    "freq_snack": "Snack frequency",
    "smoking": "Smoking",
    "monitor_calorie": "Monitor calorie intake",
    "freq_alcohol": "Alcohol frequency",
    "transport": "Usual transport mode",
    "bmi_category": "Obesity level"
}

COLORS = [
    "#E05E00",
    "#FFB26F",
    "#FFD0AA",
    "#9BC1FF",
    "#4284FF",
    "#3276C4",
    "#1F9CA3",
]

VARIABLE_FEATURES = [
    'age', 'height', 'weight', 'veggie_per_meal', 'meals_daily', 'water_daily',
    'physical_activity', 'technological_devices',
    'sex', 'family_history', 'often_high_calorie_intake', 'freq_snack',
    'smoking', 'monitor_calorie', 'freq_alcohol', 'transport'
]

def label_cat(name: str) -> str:
    return CATEGORY_LABELS.get(name, name.replace("_", " "))

def label_var(name: str) -> str:
    return VAR_LABELS.get(name, name.replace("_", " "))

def show_centered(fig, middle_ratio=5):
    left = 1.0
    mid  = middle_ratio
    right = 1.0
    c1, c2, c3 = st.columns([left, mid, right])
    with c2:
        st.pyplot(fig, use_container_width=False)
        
# Set matplotlib default parameters
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 110
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
        for col in ['age','height','weight','veggie_per_meal','meals_daily','water_daily',
                'physical_activity','technological_devices']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Pretty category labels for display (keep raw in a parallel column if needed)
        df['bmi_category_display'] = df['bmi_category'].map(label_cat)
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
def plot_obesity_distribution(df: pd.DataFrame, show_percent: bool = True, pie: bool = True) -> plt.Figure:
    order_display = [label_cat(c) for c in CATEGORY_ORDER]
    counts = df["bmi_category_display"].value_counts().reindex(order_display, fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 5 if not pie else 6))
    if pie:
        explode = [0.08 if i < 2 else 0 for i in range(len(order_display))]
        ax.pie(counts.values, labels=order_display, autopct="%1.1f%%", startangle=90,
               explode=explode, colors=COLORS, shadow=True)
        ax.set_title("Distribution of Obesity Levels")
    else:
        vals = counts.values
        if show_percent:
            vals = vals / vals.sum() * 100
            ax.bar(order_display, vals, color=COLORS)
            ax.set_ylabel("Percent of people (%)")
        else:
            ax.bar(order_display, vals, color=COLORS)
            ax.set_ylabel("Number of people")
        ax.set_title("Obesity Levels (counts or %)")
        ax.set_xticklabels(order_display, rotation=30, ha="right")
    fig.tight_layout()
    return fig


def plot_numeric_by_level(df: pd.DataFrame, var: str, kind: str = "box") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = "bmi_category_display"
    order_display = [label_cat(c) for c in CATEGORY_ORDER]

    if kind == "violin":
        sns.violinplot(x=x, y=var, data=df, order=order_display, palette=COLORS, ax=ax, cut=0)
    else:
        sns.boxplot(x=x, y=var, data=df, order=order_display, palette=COLORS, ax=ax)

    ax.set_title(f"{label_var(var)} across obesity levels")
    ax.set_xlabel(label_var("bmi_category"))
    ax.set_ylabel(label_var(var))
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig

def plot_numeric_distribution(df: pd.DataFrame, x_col: str, y_col: str, plot_type: str = 'box') -> plt.Figure:
    fig = plt.figure(figsize=(8, 6))
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
    df_with_age_groups = create_age_groups(df)
    if plot_type == 'heatmap':
        fig = plt.figure(figsize=(8, 5))
        age_bmi_crosstab = pd.crosstab(
            df_with_age_groups['age_group'], df_with_age_groups["bmi_category"], normalize='index'
        )
        sns.heatmap(age_bmi_crosstab, annot=True, fmt='.3f', cmap='YlOrRd',
                    cbar_kws={'label': 'Proportion', 'shrink': 0.8})
        plt.title('Obesity Levels Distribution by Age Group')
        plt.xlabel('Obesity Level')
        plt.ylabel('Age Group')
    elif plot_type == 'smallmultiples':
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()
        for i, category in enumerate(CATEGORY_ORDER):
            age_counts = df_with_age_groups[
                df_with_age_groups["bmi_category"] == category
            ]['age_group'].value_counts().sort_index()
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
    fig = plt.figure(figsize=(10, 8))
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

def plot_categorical_stack(df: pd.DataFrame, var: str, as_percent: bool = True, horizontal: bool = False) -> plt.Figure:
    order_display = [label_cat(c) for c in CATEGORY_ORDER]
    ct = pd.crosstab(df[var], df["bmi_category_display"]).reindex(columns=order_display, fill_value=0)
    if as_percent:
        ct = (ct.T / ct.T.sum()).T * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    if horizontal:
        ct.plot(kind="barh", stacked=True, color=COLORS, ax=ax)
        ax.set_xlabel("Percent of people (%)" if as_percent else "Count")
        ax.set_ylabel(label_var(var))
    else:
        ct.plot(kind="bar", stacked=True, color=COLORS, ax=ax)
        ax.set_ylabel("Percent of people (%)" if as_percent else "Count")
        ax.set_xlabel(label_var(var))
        ax.tick_params(axis="x", rotation=0 if var == "transport" else 30)

    ax.legend(title="Obesity level", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_title(f"{label_var(var)} vs obesity level")
    fig.tight_layout()
    return fig


def plot_age_heatmap(df: pd.DataFrame) -> plt.Figure:
    g = create_age_groups(df)
    tab = pd.crosstab(g["age_group"], g["bmi_category_display"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(tab, annot=True, fmt=".1f", cmap="YlOrRd",
                cbar_kws={"label": "Percent (%)"}, annot_kws={"size": 8}, ax=ax)
    ax.set_title("Age groups vs obesity level", fontsize=11)
    ax.set_xlabel("Obesity level", fontsize=10)
    ax.set_ylabel("Age group", fontsize=10)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig

def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str]) -> plt.Figure:
    fig = plt.figure(figsize=(8, 6))
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', center=0)
    plt.title('Correlation Heatmap of Numerical Variables')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_numeric_correlation(df: pd.DataFrame, numeric_cols: List[str]) -> plt.Figure:
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlOrRd", center=0, ax=ax)
    ax.set_title("How numeric factors move together")
    fig.tight_layout()
    return fig

def plot_categorical_distribution(df: pd.DataFrame, col: str, horizontal: bool = False,
                                  figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
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

def plot_cramers_heatmap(df: pd.DataFrame, cat_cols: List[str]) -> plt.Figure:
    M = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)
    for i in cat_cols:
        for j in cat_cols:
            M.loc[i, j] = calculate_cramers_v(df[i], df[j])

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(M.astype(float), annot=True, cmap="YlOrRd", fmt=".2f",
                square=True, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title("Strength of association (Cramér’s V) between categories")
    ax.set_xticklabels([label_var(c) for c in cat_cols], rotation=30, ha="right")
    ax.set_yticklabels([label_var(c) for c in cat_cols], rotation=0)
    fig.tight_layout()
    return fig

# NEW: scatter Age vs BMI (colored by obesity level)
def plot_scatter_age_bmi(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 7))
    dfc = df.copy()
    dfc['bmi'] = dfc['weight'] / (dfc['height'] ** 2)
    order_display = [label_cat(c) for c in CATEGORY_ORDER]
    for i, cat in enumerate(order_display):
        subset = dfc[dfc["bmi_category_display"] == cat]
        ax.scatter(subset['age'], subset['bmi'], label=cat, alpha=0.7, s=50, color=COLORS[i])
    for y in [18.5, 25, 30, 35, 40]:
        ax.axhline(y=y, color='gray', linestyle='--', alpha=0.6)
    ax.set_title('Age vs BMI by obesity level')
    ax.set_xlabel(label_var("age"))
    ax.set_ylabel("BMI")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

# NEW: generic scatter (feature vs BMI) colored by level
def plot_scatter_with_bmi(df: pd.DataFrame, var: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 6))
    dfc = df.copy()
    dfc['bmi'] = dfc['weight'] / (dfc['height'] ** 2)
    order_display = [label_cat(c) for c in CATEGORY_ORDER]
    for i, cat in enumerate(order_display):
        s = dfc[dfc["bmi_category_display"] == cat]
        ax.scatter(s[var], s['bmi'], label=cat, alpha=0.7, s=40, color=COLORS[i])
    ax.set_title(f"{label_var(var)} vs BMI by obesity level")
    ax.set_xlabel(label_var(var))
    ax.set_ylabel("BMI")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------- Pages ----------
def render_overview(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]):
    st.title("📊 Overview")
    st.caption("Simple charts with plain labels. Hover for details.")

    n_rows, n_cols = df.shape
    c1, c2, c3 = st.columns(3)
    c1.metric("People in dataset", f"{n_rows:,}")
    c2.metric("Features", f"{17}")

    with st.expander("What am I looking at?"):
        st.markdown(
            "- **Obesity level** is based on BMI from self-reported height & weight.\n"
            "- These are **screening categories**, not a clinical diagnosis.\n"
            "- Switch between **counts** and **percentages**."
        )

    st.subheader("Obesity levels in the dataset")
    st.caption("See how many people fall into each BMI-based group. Switch between pie or bar, and counts or percentages to compare sizes easily.")
    left, right = st.columns([1.1, 1.9])
    with left:
        chart_type = st.radio("Chart type", ["Pie", "Bar"], horizontal=True)
        show_percent = st.toggle("Show percentages", value=True)
    with right:
        fig = plot_obesity_distribution(df, show_percent=show_percent, pie=(chart_type == "Pie"))
        st.pyplot(fig, use_container_width=True)
    st.caption("Tip: Percent is easier to compare across groups; counts show the exact number of people.")


    st.divider()
    st.subheader("Numbers at a glance (numeric factors)")
    stats = df[[c for c in numeric_cols if c in df.columns]].describe().T
    stats = stats.rename(index=lambda x: label_var(x)).rename(columns={
        "mean": "Mean", "std": "SD", "min": "Min", "25%": "P25",
        "50%": "Median", "75%": "P75", "max": "Max"
    })
    st.dataframe(stats, use_container_width=True)

    st.divider()
    st.subheader("Age groups vs obesity level")
    st.caption("Each row adds to 100%.")
    fig = plot_age_heatmap(df)
    show_centered(fig, middle_ratio=8)

    #Categorical variables
    st.divider()
    st.subheader("Categorical variables")
    st.caption("Compare how each category splits across obesity levels. You can switch to counts or percentages.")

    left, right = st.columns([1, 2])
    with left:
        cat_cols_for_picker = [
            c for c in df.select_dtypes(include=["object"]).columns
            if c not in ["bmi_category", "bmi_category_display"]
        ]
        chosen_cat = st.selectbox("Pick a categorical variable", cat_cols_for_picker, format_func=label_var)
        as_percent = st.toggle("Show as percentages", value=True)
        horizontal = (chosen_cat == "transport")

    with right:
        st.pyplot(
            plot_categorical_stack(df, chosen_cat, as_percent=as_percent, horizontal=horizontal),
            use_container_width=True
        )

    with st.expander("Association between categories (Cramér’s V)"):
        st.pyplot(plot_cramers_heatmap(df, cat_cols_for_picker), use_container_width=False)



def render_explorer(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]):
    st.title("🔎 Explore a factor")
    st.caption("Pick one factor to see how it relates to obesity level.")

    all_vars = VARIABLE_FEATURES.copy()
    var = st.selectbox("Choose a factor", options=all_vars, format_func=label_var)

    if var in numeric_cols:
        if var == 'age':
            # Original: five tabs for age
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["Distribution", "Heatmap", "Small Multiples", "Scatter with BMI", "Correlation"]
            )
            with tab1:
                fig = plot_numeric_distribution(df, "bmi_category", var, plot_type='box')
                show_centered(fig, middle_ratio=5)
            with tab2:
                fig = plot_age_analysis(df, plot_type='heatmap')
                show_centered(fig, middle_ratio=5)
            with tab3:
                fig = plot_age_analysis(df, plot_type='smallmultiples')
                st.pyplot(fig, use_container_width=True)
            with tab4:
                fig = plot_scatter_age_bmi(df)
                show_centered(fig, middle_ratio=5)
            with tab5:
                fig = plot_correlation_heatmap(df, numeric_cols)
                show_centered(fig, middle_ratio=5)

        else:
            # Original: distribution + correlation
            tab1, tab2 = st.tabs(["Distribution", "Correlation"])
            with tab1:
                plot_type = 'violin' if var in ['veggie_per_meal', 'water_daily'] else 'box'
                fig = plot_numeric_distribution(df, "bmi_category", var, plot_type=plot_type)
                show_centered(fig, middle_ratio=5)
            with tab2:
                fig = plot_correlation_heatmap(df, numeric_cols)
                show_centered(fig, middle_ratio=5)

    elif var in categorical_cols:
        # Original: distribution + Cramér's V
        tab1, tab2 = st.tabs(["Distribution", "Cramér's V"])
        with tab1:
            horizontal = (var == 'transport')
            fig = plot_categorical_distribution(df, var, horizontal=horizontal, figsize=(10, 6))
            show_centered(fig, middle_ratio=5)
        with tab2:
            fig = plot_cramers_heatmap(df, categorical_cols)
            show_centered(fig, middle_ratio=5)



# ---------- Main ----------
def main():
    df = load_data()
    target = "bmi_category"

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = (
        df.select_dtypes(include=["object"])
          .columns.drop([target, "bmi_category_display"])
          .tolist()
    )

    # Sidebar nav
    if "page" not in st.session_state:
        st.session_state.page = "Overview"

    st.sidebar.markdown("#### Navigation")
    b1, b2 = st.sidebar.columns(2)
    if b1.button("Overview",
                 type="primary" if st.session_state.page == "Overview" else "secondary",
                 use_container_width=True):
        st.session_state.page = "Overview"
    if b2.button("Explore factors",
                 type="primary" if st.session_state.page == "Explorer" else "secondary",
                 use_container_width=True):
        st.session_state.page = "Explorer"

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Legend**")
    st.sidebar.write(", ".join([label_cat(c) for c in CATEGORY_ORDER]))

    if st.session_state.page == "Overview":
        render_overview(df, numeric_cols, categorical_cols)
    else:
        render_explorer(df, numeric_cols, categorical_cols)


if __name__ == "__main__":
    main()