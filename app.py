import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="ML Preprocessing Tool", layout="wide")

st.title("⚙️ Interactive ML Data Preprocessing Tool")
st.caption("EDA → Cleaning → Transformation → Encoding → Reduction")

# ------------------ FILE UPLOAD ------------------
file = st.file_uploader("Upload CSV file", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.success("Dataset Loaded Successfully")

    # ------------------ BASIC INFO ------------------
    st.subheader("📊 Dataset Overview")
    col1, col2 = st.columns(2)
    col1.write("Shape:", df.shape)
    col2.write("Columns:", list(df.columns))

    st.dataframe(df.head())

    # ------------------ COLUMN TYPES ------------------
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    st.subheader("🧠 Auto Detected Columns")
    st.write("Numerical:", num_cols)
    st.write("Categorical:", cat_cols)

    # ------------------ EDA ------------------
    with st.expander("🔍 Exploratory Data Analysis"):
        st.write("Missing Values (%)")
        st.write((df.isnull().mean() * 100).round(2))

        st.write("Statistical Summary")
        st.dataframe(df.describe())

    # ------------------ PREPROCESSING OPTIONS ------------------
    st.subheader("🛠️ Preprocessing Options")

    colA, colB, colC = st.columns(3)

    with colA:
        handle_missing = st.checkbox("Fill Missing Values")
        outlier_cap = st.checkbox("Cap Outliers (IQR)")

    with colB:
        scale_data = st.checkbox("Standardize Numerical Features")
        log_transform = st.checkbox("Log Transform (Skewed Data)")

    with colC:
        encode_cat = st.checkbox("One-Hot Encode Categoricals")
        use_pca = st.checkbox("Apply PCA")

    # ------------------ PCA SLIDER ------------------
    if use_pca:
        variance = st.slider("Variance to Preserve (%)", 80, 99, 95)

    # ------------------ APPLY PREPROCESSING ------------------
    if st.button("🚀 Apply Preprocessing"):

        data = df.copy()

        # Missing Values
        if handle_missing:
            for col in num_cols:
                data[col].fillna(data[col].median(), inplace=True)
            for col in cat_cols:
                data[col].fillna(data[col].mode()[0], inplace=True)

        # Outlier Capping
        if outlier_cap:
            for col in num_cols:
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                data[col] = np.clip(data[col], lower, upper)

        # Log Transform
        if log_transform:
            for col in num_cols:
                if (data[col] > 0).all():
                    data[col] = np.log1p(data[col])

        # Column Transformer
        transformers = []

        if scale_data:
            transformers.append(("num", StandardScaler(), num_cols))

        if encode_cat:
            transformers.append(
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
            )

        preprocessor = ColumnTransformer(transformers, remainder="drop")

        pipeline_steps = [("preprocessor", preprocessor)]

        # PCA
        if use_pca:
            pipeline_steps.append(
                ("pca", PCA(n_components=variance / 100))
            )

        pipeline = Pipeline(pipeline_steps)

        processed_data = pipeline.fit_transform(data)

        st.success("Preprocessing Completed")

        st.subheader("✅ Processed Data Preview")
        st.write("Shape:", processed_data.shape)
        st.dataframe(pd.DataFrame(processed_data).head())

        # ------------------ DOWNLOAD ------------------
        processed_df = pd.DataFrame(processed_data)
        csv = processed_df.to_csv(index=False).encode()

        st.download_button(
            "⬇️ Download Cleaned Dataset",
            csv,
            "processed_data.csv",
            "text/csv"
        )
