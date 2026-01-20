import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="ML Preprocessing Tool",
    layout="wide"
)

st.title("⚙️ Interactive ML Data Preprocessing Tool")
st.caption("EDA → Cleaning → Transformation → Encoding → Reduction")

# ------------------ FILE UPLOAD ------------------
file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if file is not None:
    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")
        st.stop()

    st.success("✅ Dataset Loaded Successfully")

    # ------------------ DATASET OVERVIEW ------------------
    st.subheader("📊 Dataset Overview")

    col1, col2 = st.columns(2)
    col1.markdown(f"**Shape:** {df.shape}")
    col2.markdown(f"**Columns:** {list(df.columns)}")

    st.dataframe(df.head(), use_container_width=True)

    # ------------------ COLUMN TYPE DETECTION ------------------
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    st.subheader("🧠 Auto-Detected Column Types")
    st.write("**Numerical Columns:**", num_cols)
    st.write("**Categorical Columns:**", cat_cols)

    # ------------------ EDA SECTION ------------------
    with st.expander("🔍 Exploratory Data Analysis (EDA)"):
        st.markdown("### Missing Values (%)")
        missing = (df.isnull().mean() * 100).round(2)
        st.dataframe(missing)

        st.markdown("### Statistical Summary")
        st.dataframe(df.describe(include="all"))

    # ------------------ PREPROCESSING OPTIONS ------------------
    st.subheader("🛠️ Preprocessing Options")

    c1, c2, c3 = st.columns(3)

    with c1:
        handle_missing = st.checkbox("Fill Missing Values", value=True)
        cap_outliers = st.checkbox("Cap Outliers (IQR)")

    with c2:
        scale_data = st.checkbox("Standardize Numerical Features")
        log_transform = st.checkbox("Log Transform (Positive Values Only)")

    with c3:
        encode_cat = st.checkbox("One-Hot Encode Categorical Features", value=True)
        use_pca = st.checkbox("Apply PCA (Dimensionality Reduction)")

    if use_pca:
        variance = st.slider("Variance to Preserve (%)", 80, 99, 95)

    # ------------------ APPLY PREPROCESSING ------------------
    if st.button("🚀 Apply Preprocessing"):
        data = df.copy()

        # -------- Missing Values --------
        if handle_missing:
            for col in num_cols:
                data[col] = data[col].fillna(data[col].median())
            for col in cat_cols:
                if not data[col].mode().empty:
                    data[col] = data[col].fillna(data[col].mode()[0])

        # -------- Outlier Capping (IQR) --------
        if cap_outliers:
            for col in num_cols:
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                data[col] = np.clip(data[col], lower, upper)

        # -------- Log Transform --------
        if log_transform:
            for col in num_cols:
                if (data[col] > 0).all():
                    data[col] = np.log1p(data[col])

        # -------- Column Transformer --------
        transformers = []

        if scale_data and num_cols:
            transformers.append(
                ("num", StandardScaler(), num_cols)
            )

        if encode_cat and cat_cols:
            transformers.append(
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
            )

        if not transformers:
            st.error("❌ No preprocessing step selected.")
            st.stop()

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )

        steps = [("preprocessor", preprocessor)]

        if use_pca:
            steps.append(
                ("pca", PCA(n_components=variance / 100))
            )

        pipeline = Pipeline(steps)

        # -------- Fit & Transform --------
        processed_data = pipeline.fit_transform(data)

        processed_df = pd.DataFrame(processed_data)

        # ------------------ OUTPUT ------------------
        st.success("✅ Preprocessing Completed Successfully")

        st.subheader("📦 Processed Dataset")
        st.markdown(f"**Shape:** {processed_df.shape}")
        st.dataframe(processed_df.head(), use_container_width=True)

        # ------------------ DOWNLOAD ------------------
        csv = processed_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Processed Dataset",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )

else:
    st.info("⬆️ Upload a CSV file to begin.")
