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

#st.sidebar.markdown("# Descriptive Analytics 📊")
#st.set_page_config(page_title="Descriptive Analytics", layout="wide")

#Remember user's last tab choice
#if 'page' not in st.session_state:
#    st.session_state['page'] = 'Overview'
#page = st.sidebar.radio("Descriptive Analytics", ["Overview", "Single Features", "Multiple Features"],
#                        index=["Overview","Single Features","Multiple Features"].index(st.session_state['page'])
#                        if st.session_state['page'] in ["Overview","Single Features","Multiple Features"] else 0)
#st.session_state['page'] = page

#Default
#page = st.sidebar.radio(
#    "Descriptive Analytics",
#    ["Overview", "Single Features", "Multiple Features"],
#    index=0
#)




# Page config
st.set_page_config(page_title="Descriptive Analytics", layout="wide")

# ---- Settings / constants ----
target = "bmi_category"
DEFAULT_CSV_PATH = "./assets/ObesityDataSet_raw_and_data_sinthetic.csv"

# ---- Load data (cached) ----
@st.cache_data
def load_data(path=DEFAULT_CSV_PATH):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "Age": "age",
        "Gender":"sex",
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
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load dataset. Make sure '{DEFAULT_CSV_PATH}' exists. Error: {e}")
    st.stop()

# ---- Prepare variable lists ----
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# categorical_cols: drop target if present
categorical_cols = df.select_dtypes(include=["object"]).columns.drop(target).tolist()

# category order used in notebook
category_order = [
    'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
    'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
]

# colors & plotting defaults
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
colors = ['#F7F0D4', '#A5C2A7', '#C3D2E0', '#A5B8D6', '#E8C4B8', '#D49A89', '#B36A5E']

# The exact feature list you asked for (single-select list)
variable_features = [
    'age', 'height', 'weight', 'veggie_per_meal', 'meals_daily', 'water_daily',
    'physical_activity', 'technological_devices',
    'sex', 'family_history', 'often_high_calorie_intake', 'freq_snack',
    'smoking', 'monitor_calorie', 'freq_alcohol', 'transport'
]

# ---- Plot-generating functions (each returns a matplotlib.figure) ----
def plot_pie_distribution(df_local):
    fig, ax = plt.subplots(figsize=(6,6))
    counts = df_local[target].value_counts().reindex(category_order)
    explode = (0.05, 0.1, 0, 0, 0, 0, 0)
    ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
           startangle=90, explode=explode, colors=colors, shadow=True)
    ax.set_title('Distribution of Obesity Levels', pad=20)
    return fig

def plot_bar_by_col(df_local, col, rotated=False, horizontal=False, figsize=(10,6)):
    fig, ax = plt.subplots(figsize=figsize)
    cross_tab = pd.crosstab(df_local[col], df_local[target])
    cross_tab = cross_tab.reindex(columns=category_order, fill_value=0)
    if horizontal:
        cross_tab.plot(kind='barh', ax=ax, color=colors, width=0.8)
        ax.set_xlabel('Count')
        ax.set_ylabel(col)
    else:
        cross_tab.plot(kind='bar', ax=ax, color=colors, width=0.8)
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45 if rotated else 0)
    ax.legend(title='Obesity Level', bbox_to_anchor=(1.05,1), loc='upper left')
    ax.set_title(f'Obesity Levels Distribution by {col}')
    plt.tight_layout()
    return fig

def plot_age_box(df_local):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.boxplot(x=target, y='age', data=df_local, order=category_order, palette=colors, ax=ax)
    ax.set_title('Age Distribution by Obesity Level')
    ax.set_xlabel('Obesity Level')
    ax.set_ylabel('Age')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_age_heatmap(df_local):
    df2 = df_local.copy()
    df2['age_group'] = pd.cut(df2['age'], bins=[0,18,30,45,60,100], labels=['0-18','19-30','31-45','46-60','60+'])
    age_bmi_crosstab = pd.crosstab(df2['age_group'], df2[target], normalize='index')
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(age_bmi_crosstab, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label':'Proportion','shrink':0.8}, ax=ax)
    ax.set_title('Obesity Levels Distribution by Age Group')
    ax.set_xlabel('Obesity Level')
    ax.set_ylabel('Age Group')
    plt.tight_layout()
    return fig

def plot_age_smallmultiples(df_local):
    df2 = df_local.copy()
    df2['age_group'] = pd.cut(df2['age'], bins=[0,18,30,45,60,100], labels=['0-18','19-30','31-45','46-60','60+'])
    fig, axes = plt.subplots(2,4, figsize=(12,6))
    axes = axes.flatten()
    for i, category in enumerate(category_order):
        age_counts = df2[df2[target]==category]['age_group'].value_counts().sort_index()
        axes[i].bar(age_counts.index.astype(str), age_counts.values, color=colors[i], alpha=0.8)
        axes[i].set_title(category, fontsize=10)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_ylabel('Count')
    for j in range(len(category_order), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Obesity Levels Distribution by Age Group')
    plt.tight_layout()
    return fig

def plot_violin_veggie(df_local):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.violinplot(x=target, y='veggie_per_meal', data=df_local, order=category_order, palette=colors, ax=ax)
    ax.set_title('Vegetable Intake by Obesity Level')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_bar_veggie(df_local):
    df2 = df_local.copy()
    df2['veggie_intake_level'] = pd.cut(df2['veggie_per_meal'], bins=[0,1.5,2.5,3], labels=['Low','Medium','High'])
    return plot_bar_by_col(df2, 'veggie_intake_level')

def plot_meals_box(df_local):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.boxplot(x=target, y='meals_daily', data=df_local, order=category_order, palette=colors, ax=ax)
    ax.set_title('Number of Daily Meals by Obesity Level')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_bar_meals(df_local):
    df2 = df_local.copy()
    df2['meals_level'] = pd.cut(df2['meals_daily'], bins=[0,2,3,4], labels=['Few','Normal','Many'])
    return plot_bar_by_col(df2, 'meals_level')

def plot_violin_water(df_local):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.violinplot(x=target, y='water_daily', data=df_local, order=category_order, palette=colors, ax=ax)
    ax.set_title('Water Intake by Obesity Level')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_bar_water(df_local):
    df2 = df_local.copy()
    df2['water_intake_level'] = pd.cut(df2['water_daily'], bins=[0,1.5,2.5,3], labels=['Low','Medium','High'])
    return plot_bar_by_col(df2, 'water_intake_level')

def plot_physical_box(df_local):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.boxplot(x=target, y='physical_activity', data=df_local, order=category_order, palette=colors, ax=ax)
    ax.set_title('Physical Activity by Obesity Level')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_bar_physical(df_local):
    df2 = df_local.copy()
    df2['physical_activity_level'] = pd.cut(df2['physical_activity'], bins=[-1,1,2,3], labels=['Low','Medium','High'])
    return plot_bar_by_col(df2, 'physical_activity_level')

def plot_violin_gadget(df_local):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.violinplot(x=target, y='technological_devices', data=df_local, order=category_order, palette=colors, ax=ax)
    ax.set_title('Gadget Time by Obesity Level')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_bar_gadget(df_local):
    df2 = df_local.copy()
    df2['gadget_time_level'] = pd.cut(df2['technological_devices'], bins=[0,1,2,3], labels=['Low','Medium','High'])
    return plot_bar_by_col(df2, 'gadget_time_level')

def plot_scatter_age_bmi(df_local):
    df2 = df_local.copy()
    df2['bmi'] = df2['weight'] / (df2['height'] ** 2)
    fig, ax = plt.subplots(figsize=(10,8))
    for i, category in enumerate(category_order):
        subset = df2[df2[target]==category]
        ax.scatter(subset['age'], subset['bmi'], color=colors[i], label=category, alpha=0.7, s=60)
    for y in [18.5,25,30,35,40]:
        ax.axhline(y=y, color='gray', linestyle='--', alpha=0.7)
    ax.set_title('Age vs BMI by Obesity Level')
    ax.set_xlabel('Age')
    ax.set_ylabel('BMI')
    ax.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df_local):
    fig, ax = plt.subplots(figsize=(8,6))
    correlation_matrix = df_local[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', center=0, ax=ax)
    ax.set_title('Correlation Heatmap of Numerical Variables')
    plt.tight_layout()
    return fig

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    denom = min((kcorr-1), (rcorr-1))
    if denom <= 0:
        return 0.0
    return np.sqrt(phi2corr / denom)

def plot_cramers_heatmap(df_local):
    categorical_vars = categorical_cols
    cramer_matrix = pd.DataFrame(index=categorical_vars, columns=categorical_vars, dtype=float)
    for i in categorical_vars:
        for j in categorical_vars:
            try:
                cramer_matrix.loc[i,j] = cramers_v(df_local[i], df_local[j])
            except Exception:
                cramer_matrix.loc[i,j] = 0.0
    fig, ax = plt.subplots(figsize=(10,7))
    sns.heatmap(cramer_matrix.astype(float), annot=True, cmap='Blues', fmt='.2f', square=True, cbar_kws={'shrink':.8}, ax=ax)
    ax.set_title("Association Heatmap of Categorical Variables (Cramer's V)")
    plt.tight_layout()
    return fig

# ---- Mapping of variables -> list of plot ids (used in Single Features) ----
plots_map = {
    'age': ['pie','age_box','age_heatmap','age_smallmultiples','scatter','correlation'],
    'height': ['scatter','correlation'],
    'weight': ['scatter','correlation'],
    'veggie_per_meal': ['violin_veggie','bar_veggie'],
    'meals_daily': ['meals_box','meals_bar'],
    'water_daily': ['violin_water','bar_water'],
    'physical_activity': ['physical_box','physical_bar'],
    'technological_devices': ['gadget_violin','gadget_bar'],
    'sex': ['pie','by_sex'],
    'family_history': ['by_family_history'],
    'often_high_calorie_intake': ['by_favc'],
    'freq_snack': ['by_snack'],
    'smoking': ['by_smoke'],
    'monitor_calorie': ['by_monitor'],
    'freq_alcohol': ['by_alcohol'],
    'transport': ['by_transport']
}

# ---- Factory to call functions by id ----
plot_functions = {
    'pie': lambda: plot_pie_distribution(df),
    'by_sex': lambda: plot_bar_by_col(df, 'sex'),
    'by_family_history': lambda: plot_bar_by_col(df, 'family_history'),
    'by_favc': lambda: plot_bar_by_col(df, 'often_high_calorie_intake'),
    'by_snack': lambda: plot_bar_by_col(df, 'freq_snack'),
    'by_smoke': lambda: plot_bar_by_col(df, 'smoking'),
    'by_monitor': lambda: plot_bar_by_col(df, 'monitor_calorie'),
    'by_alcohol': lambda: plot_bar_by_col(df, 'freq_alcohol'),
    'by_transport': lambda: plot_bar_by_col(df, 'transport', horizontal=True),
    'age_box': lambda: plot_age_box(df),
    'age_heatmap': lambda: plot_age_heatmap(df),
    'age_smallmultiples': lambda: plot_age_smallmultiples(df),
    'violin_veggie': lambda: plot_violin_veggie(df),
    'bar_veggie': lambda: plot_bar_veggie(df),
    'meals_box': lambda: plot_meals_box(df),
    'meals_bar': lambda: plot_bar_meals(df),
    'violin_water': lambda: plot_violin_water(df),
    'bar_water': lambda: plot_bar_water(df),
    'physical_box': lambda: plot_physical_box(df),
    'physical_bar': lambda: plot_bar_physical(df),
    'gadget_violin': lambda: plot_violin_gadget(df),
    'gadget_bar': lambda: plot_bar_gadget(df),
    'scatter': lambda: plot_scatter_age_bmi(df),
    'correlation': lambda: plot_correlation_heatmap(df),
    'cramer': lambda: plot_cramers_heatmap(df)
}

# ---- Sidebar: subpage selector (label empty to avoid duplicate top Pages label) ----
subpage = st.sidebar.radio(
    "",  # keep empty so it does not duplicate the Streams Pages label
    ["Overview", "Single Features", "Multiple Features"],
    index=0,
    key="subpage_selector"
)

# ---- Main content ----
st.markdown('<div class="main-content">', unsafe_allow_html=True)

if subpage == "Overview":
    st.title("Overview — Descriptive Statistics & Pie Chart")

    # Numerical/Categorical Variables (as in your notebook)
    numeric_cols_shown = numeric_cols
    categorical_cols_shown = categorical_cols

    st.write("**Data shape:**", df.shape)
    st.write("**Numerical variables:**", numeric_cols_shown)
    st.write("**Categorical variables:**", categorical_cols_shown)

    st.subheader("Missing values")
    st.dataframe(df.isnull().sum())

    st.subheader("Numerical variables statistics")
    st.dataframe(df[numeric_cols_shown].describe().T)

    st.subheader("Categorical variables cardinality")
    for col in categorical_cols_shown:
        st.write(f"{col}: {df[col].nunique()}")

    st.subheader("Obesity Level Distribution (Pie Chart)")
    fig = plot_pie_distribution(df)
    st.pyplot(fig)

elif subpage == "Single Features":
    st.title("Single Features — choose one variable (single-select)")

    left_col, right_col = st.columns([1,3])

    with left_col:
        st.markdown("### Select feature")
        # use the exact feature list requested by you
        var = st.radio("", options=variable_features, index=0, key="feature_selector")
        st.markdown("---")
        st.markdown("Legend:")
        st.write(", ".join(category_order))
        st.markdown("---")
        if st.button("Show Cramer's V (categorical associations)", key="cramer_btn"):
            fig = plot_functions['cramer']()
            right_col.pyplot(fig)

    with right_col:
        st.markdown(f"### Plots for **{var}**")
        selected_plots = plots_map.get(var, [])
        if not selected_plots:
            st.info("No pre-defined plots for this feature.")
        for p in selected_plots:
            if p in plot_functions:
                fig = plot_functions[p]()
                st.pyplot(fig)
            else:
                st.info(f"Plot {p} not implemented")

elif subpage == "Multiple Features":
    st.title("Multiple Features — correlation, diagnostics & PCA")

    # Correlation heatmap for numeric
    st.subheader("Correlation Heatmap (Numerical Variables)")
    fig = plot_correlation_heatmap(df)
    st.pyplot(fig)

    # Cramer's V heatmap for categorical
    st.subheader("Association Heatmap of Categorical Variables (Cramer's V)")
    fig = plot_cramers_heatmap(df)
    st.pyplot(fig)

    # Diagnostic: logistic regression coefficients, ETA/CRAMER, PCA
    st.subheader("Diagnostic: Multiclass Logistic Regression + Association Strength + PCA")
    with st.spinner("Running diagnostic models (may take a few seconds)..."):
        # Prepare df_analysis
        df_analysis = df[numeric_cols + categorical_cols + [target]].copy()

        # Impute missing values
        for col in numeric_cols:
            df_analysis[col] = df_analysis[col].fillna(df_analysis[col].median())
        for col in categorical_cols:
            df_analysis[col] = df_analysis[col].fillna(df_analysis[col].mode()[0] if not df_analysis[col].mode().empty else "Missing")

        X = df_analysis[numeric_cols + categorical_cols]
        y = df_analysis[target]

        # Encode target
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

        # get feature names
        numeric_features = numeric_cols
        try:
            categorical_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
        except Exception:
            # fallback
            categorical_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
        all_features = list(numeric_features) + list(categorical_features)

        # Fit logistic regression (multiclass)
        logreg = LogisticRegression(multi_class='multinomial', max_iter=1000, penalty='l2', C=1.0, random_state=42)
        logreg.fit(X_processed, y_encoded)

        coefficients = logreg.coef_  # shape: (n_classes, n_features)
        coef_df = pd.DataFrame(coefficients, columns=all_features, index=le.classes_)

        st.write("Multiclass Logistic Regression Coefficients (per class)")
        st.dataframe(coef_df)

        # show top features per class (absolute coefficient)
        n_top = 10
        st.write("Top features (by absolute coefficient) per class:")
        for i, class_name in enumerate(le.classes_):
            class_coef = coef_df.loc[class_name].abs().sort_values(ascending=False).head(n_top)
            fig, ax = plt.subplots(figsize=(8, 4))
            colors_bars = ['#4C72B0' if f in numeric_cols else '#55A868' for f in class_coef.index]
            ax.barh(range(len(class_coef)), class_coef.values, color=colors_bars)
            ax.set_yticks(range(len(class_coef)))
            ax.set_yticklabels(class_coef.index)
            ax.invert_yaxis()
            ax.set_xlabel('Coefficient magnitude (abs)')
            ax.set_title(f'Top {n_top} features for {class_name}')
            legend_elements = [Patch(facecolor='#4C72B0', label='Numerical'), Patch(facecolor='#55A868', label='Categorical')]
            ax.legend(handles=legend_elements, loc='lower right')
            plt.tight_layout()
            st.pyplot(fig)

        # Association strength (eta-squared for numeric, Cramer's V for categorical)
        st.write("Association strength (Eta-squared for numeric, Cramer's V for categorical)")
        eta_squared_results = {}

        # numeric -> eta-squared (ANOVA)
        for col in numeric_cols:
            clean = df_analysis[[col, target]].dropna()
            if len(clean) > 0:
                overall_mean = clean[col].mean()
                ss_total = np.sum((clean[col] - overall_mean) ** 2)
                ss_between = 0.0
                for category in category_order:
                    group = clean[clean[target] == category][col]
                    if len(group) > 0:
                        ss_between += len(group) * (group.mean() - overall_mean) ** 2
                eta_sq = ss_between / ss_total if ss_total != 0 else 0
                eta_squared_results[col] = eta_sq
            else:
                eta_squared_results[col] = 0.0

        # categorical -> cramers v with target
        for col in categorical_cols:
            clean = df_analysis[[col, target]].dropna()
            if len(clean) > 0:
                ct = pd.crosstab(clean[col], clean[target])
                chi2 = chi2_contingency(ct)[0]
                n = ct.sum().sum()
                phi2 = chi2 / n
                r, k = ct.shape
                phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
                rcorr = r - ((r-1)**2)/(n-1)
                kcorr = k - ((k-1)**2)/(n-1)
                denom = min((kcorr-1), (rcorr-1))
                cramers_v_val = np.sqrt(phi2corr/denom) if denom > 0 else 0.0
                eta_squared_results[col] = cramers_v_val
            else:
                eta_squared_results[col] = 0.0

        eta_df = pd.DataFrame.from_dict(eta_squared_results, orient='index', columns=['Association_Strength'])
        eta_df = eta_df.sort_values('Association_Strength', ascending=False)
        st.dataframe(eta_df)

        # bar chart of association strengths
        fig, ax = plt.subplots(figsize=(8, max(4, len(eta_df)/3)))
        colors_bars = ['#4C72B0' if c in numeric_cols else '#55A868' for c in eta_df.index]
        ax.barh(range(len(eta_df)), eta_df['Association_Strength'], color=colors_bars)
        ax.set_yticks(range(len(eta_df)))
        ax.set_yticklabels(eta_df.index)
        ax.invert_yaxis()
        ax.set_xlabel('Association Strength')
        ax.set_title("Association Strength with Obesity Level (Eta-sq / Cramer's V)")
        legend_elements = [Patch(facecolor='#4C72B0', label='Numeric (Eta^2)'), Patch(facecolor='#55A868', label="Categorical (Cramer's V)")]
        ax.legend(handles=legend_elements, loc='lower right')
        plt.tight_layout()
        st.pyplot(fig)

        # PCA (2 components) visualization
        st.write("PCA (2 components) of processed features")
        try:
            X_for_pca = X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_for_pca)
            feature_weights = pd.DataFrame(pca.components_, columns=all_features, index=['PC1','PC2'])

            fig, ax = plt.subplots(figsize=(8,6))
            for i, class_name in enumerate(le.classes_):
                mask = (y_encoded == i)
                ax.scatter(X_pca[mask,0], X_pca[mask,1], label=class_name, alpha=0.6, s=40)
            # draw arrows for a few important variables if present
            important_vars = ['height', 'weight', 'age', 'technological_devices', 'physical_activity']
            for var in important_vars:
                if var in feature_weights.columns:
                    pc1_load = feature_weights.loc['PC1', var]
                    pc2_load = feature_weights.loc['PC2', var]
                    ax.arrow(0, 0, pc1_load*3, pc2_load*3, head_width=0.05, head_length=0.05, fc='red', ec='red')
                    ax.text(pc1_load*3.2, pc2_load*3.2, var, color='red')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.set_title('PCA: 2D projection of observations (colored by class)')
            ax.legend(bbox_to_anchor=(1.05,1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)

            # show top loadings per PC
            st.write("Top feature loadings per PC (absolute value)")
            for i in range(2):
                pc_weights = feature_weights.iloc[i].abs().sort_values(ascending=False).head(10)
                st.write(f"PC{i+1} top features:")
                st.dataframe(pd.DataFrame({'Feature': pc_weights.index, 'Weight': pc_weights.values}).reset_index(drop=True))
        except Exception as e:
            st.error(f"PCA / plotting failed: {e}")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div style="height:60px"></div>', unsafe_allow_html=True)