"""
streamlit_app/app.py
---------------------
Multi-page Streamlit dashboard for the Customer Churn Prediction project.

Pages:
    * Home
    * About Dataset
    * EDA
    * Prediction (single customer + CSV batch upload)
    * Analytics

Run with:
    streamlit run streamlit_app/app.py

Author: ML Engineering Team
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Make the `src` package importable when running via `streamlit run`
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict import ChurnPredictor
from src.utils import MODEL_FILE, PROCESSED_DATA_FILE, RAW_DATA_FILE

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Cached data / model loaders
# --------------------------------------------------------------------------- #
@st.cache_data
def load_dataset() -> pd.DataFrame | None:
    """Load the processed dataset if available, otherwise the raw one."""
    if PROCESSED_DATA_FILE.exists():
        return pd.read_csv(PROCESSED_DATA_FILE)
    if RAW_DATA_FILE.exists():
        return pd.read_csv(RAW_DATA_FILE)
    return None


@st.cache_resource
def load_predictor() -> ChurnPredictor | None:
    """Load the trained model wrapper, or None if it hasn't been trained yet."""
    if not MODEL_FILE.exists():
        return None
    try:
        return ChurnPredictor()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load model: {exc}")
        return None


# --------------------------------------------------------------------------- #
# Page: Home
# --------------------------------------------------------------------------- #
def page_home() -> None:
    st.title("📊 Customer Churn Prediction Dashboard")
    st.markdown(
        """
        Welcome! This dashboard lets you explore the **IBM Telco Customer
        Churn** dataset, understand what drives customer churn, and predict
        whether a customer is likely to churn — either one at a time or in
        bulk via CSV upload.

        ### What you can do here
        - 📁 **About Dataset** — understand the schema and business context
        - 📈 **EDA** — interactive exploratory data analysis
        - 🔮 **Prediction** — predict churn for a single customer or a batch
        - 📊 **Analytics** — KPI cards and churn analytics

        Use the sidebar to navigate between pages.
        """
    )

    col1, col2, col3 = st.columns(3)
    df = load_dataset()
    predictor = load_predictor()

    with col1:
        st.metric("Total Customers", f"{len(df):,}" if df is not None else "N/A")
    with col2:
        if df is not None and "Churn" in df.columns:
            churn_col = df["Churn"]
            churn_rate = (
                churn_col.mean() * 100
                if pd.api.types.is_numeric_dtype(churn_col)
                else (churn_col == "Yes").mean() * 100
            )
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        else:
            st.metric("Churn Rate", "N/A")
    with col3:
        st.metric("Model Status", "✅ Trained" if predictor else "⚠️ Not Trained")

    if predictor is None:
        st.info(
            "No trained model found yet. Run `python src/train.py` from the "
            "project root, then reload this app."
        )


# --------------------------------------------------------------------------- #
# Page: About Dataset
# --------------------------------------------------------------------------- #
def page_about_dataset() -> None:
    st.title("📁 About the Dataset")
    st.markdown(
        """
        This project uses the **IBM Telco Customer Churn** dataset, a
        widely used benchmark dataset for churn prediction, originally
        published by IBM and mirrored on
        [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

        **Each row represents one customer**, and columns describe:

        | Category | Example Columns |
        |---|---|
        | Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
        | Account info | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
        | Services | `PhoneService`, `InternetService`, `OnlineSecurity`, `StreamingTV`, ... |
        | Charges | `MonthlyCharges`, `TotalCharges` |
        | Target | `Churn` (Yes/No) |
        """
    )

    df = load_dataset()
    if df is not None:
        st.subheader("Sample Records")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Column Summary")
        summary = pd.DataFrame(
            {
                "dtype": df.dtypes.astype(str),
                "missing_values": df.isnull().sum(),
                "unique_values": df.nunique(),
            }
        )
        st.dataframe(summary, use_container_width=True)
    else:
        st.warning(
            "Dataset not found. Place `Telco-Customer-Churn.csv` inside "
            "`data/raw/` and run `python src/train.py`."
        )


# --------------------------------------------------------------------------- #
# Page: EDA
# --------------------------------------------------------------------------- #
def page_eda() -> None:
    st.title("📈 Exploratory Data Analysis")
    df = load_dataset()

    if df is None:
        st.warning("Dataset not found. Please add the raw CSV to `data/raw/`.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Churn Overview", "Demographics", "Contracts & Charges", "Correlations"]
    )

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(
                df, names="Churn", title="Churn Distribution", hole=0.4,
                color="Churn", color_discrete_map={"Yes": "#EF553B", "No": "#00CC96"},
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(
                df, x="tenure", color="Churn", barmode="overlay", nbins=30,
                title="Tenure Distribution by Churn",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                df, x="gender", color="Churn", barmode="group",
                title="Churn by Gender",
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if "SeniorCitizen" in df.columns:
                sc_df = df.copy()
                sc_df["SeniorCitizen"] = sc_df["SeniorCitizen"].map({0: "No", 1: "Yes"})
                fig = px.histogram(
                    sc_df, x="SeniorCitizen", color="Churn", barmode="group",
                    title="Churn by Senior Citizen Status",
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                df, x="Contract", color="Churn", barmode="group",
                title="Churn by Contract Type",
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(
                df, x="Churn", y="MonthlyCharges", color="Churn",
                title="Monthly Charges by Churn",
            )
            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            df, x="tenure", y="MonthlyCharges", color="Churn", opacity=0.6,
            title="Tenure vs Monthly Charges",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        numeric_df = df.select_dtypes(include=["int64", "float64"])
        if not numeric_df.empty:
            corr = numeric_df.corr(numeric_only=True)
            fig = px.imshow(
                corr, text_auto=".2f", aspect="auto",
                title="Correlation Heatmap (Numeric Features)",
                color_continuous_scale="RdBu_r",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for correlation analysis.")


# --------------------------------------------------------------------------- #
# Page: Prediction
# --------------------------------------------------------------------------- #
def page_prediction() -> None:
    st.title("🔮 Churn Prediction")
    predictor = load_predictor()

    if predictor is None:
        st.warning("No trained model found. Run `python src/train.py` first.")
        return

    tab1, tab2 = st.tabs(["Single Customer", "Batch Upload (CSV)"])

    with tab1:
        st.subheader("Enter Customer Details")
        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with c2:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
            online_security = st.selectbox(
                "Online Security", ["Yes", "No", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"]
            )

        with c3:
            device_protection = st.selectbox(
                "Device Protection", ["Yes", "No", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech Support", ["Yes", "No", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["Yes", "No", "No internet service"]
            )
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )

        c4, c5, c6 = st.columns(3)
        with c4:
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        with c5:
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
        with c6:
            monthly_charges = st.number_input(
                "Monthly Charges ($)", min_value=0.0, max_value=500.0, value=70.0
            )

        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(monthly_charges * tenure)
        )

        if st.button("🔍 Predict Churn", type="primary"):
            customer = {
                "gender": gender,
                "SeniorCitizen": 1 if senior == "Yes" else 0,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }

            result = predictor.predict_single(customer)

            st.divider()
            r1, r2, r3 = st.columns(3)
            with r1:
                if result["prediction"] == "Churn":
                    st.error(f"### Prediction: {result['prediction']} ⚠️")
                else:
                    st.success(f"### Prediction: {result['prediction']} ✅")
            with r2:
                st.metric("Churn Probability", f"{result['churn_probability']}%")
            with r3:
                st.metric("Confidence", f"{result['confidence']}%")

            if result["top_factors"]:
                st.subheader("Top Factors Influencing This Prediction")
                factors_df = pd.DataFrame(result["top_factors"])
                fig = px.bar(
                    factors_df, x="impact", y="feature", orientation="h",
                    title="Feature Contribution", color="impact",
                    color_continuous_scale="Oranges",
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Upload a CSV for Batch Prediction")
        st.caption(
            "The CSV should follow the same schema as the Telco dataset "
            "(without the Churn column)."
        )
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("🔍 Predict for All Customers"):
                with st.spinner("Running predictions..."):
                    result_df = predictor.predict_batch(batch_df)

                st.success(f"Predicted churn for {len(result_df)} customers.")
                st.dataframe(result_df, use_container_width=True)

                churn_count = (result_df["Churn_Prediction"] == "Churn").sum()
                st.metric(
                    "Predicted Churners",
                    f"{churn_count} / {len(result_df)} "
                    f"({churn_count / len(result_df) * 100:.1f}%)",
                )

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Predictions as CSV",
                    data=csv_bytes,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                )


# --------------------------------------------------------------------------- #
# Page: Analytics
# --------------------------------------------------------------------------- #
def page_analytics() -> None:
    st.title("📊 Business Analytics")
    df = load_dataset()

    if df is None:
        st.warning("Dataset not found. Please add the raw CSV to `data/raw/`.")
        return

    churn_yes = (
        df["Churn"] == 1
        if pd.api.types.is_numeric_dtype(df["Churn"])
        else df["Churn"] == "Yes"
    )
    churn_rate = churn_yes.mean() * 100
    avg_tenure = df["tenure"].mean()
    avg_monthly = df["MonthlyCharges"].mean()
    total_revenue_at_risk = df.loc[churn_yes, "MonthlyCharges"].sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers", f"{len(df):,}")
    k2.metric("Churn Rate", f"{churn_rate:.1f}%")
    k3.metric("Avg. Tenure", f"{avg_tenure:.1f} months")
    k4.metric("Monthly Revenue at Risk", f"${total_revenue_at_risk:,.0f}")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        contract_churn = (
            df.groupby("Contract")["Churn"]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reset_index(name="Churn Rate (%)")
        )
        fig = px.bar(
            contract_churn, x="Contract", y="Churn Rate (%)",
            title="Churn Rate by Contract Type", color="Churn Rate (%)",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        payment_churn = (
            df.groupby("PaymentMethod")["Churn"]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reset_index(name="Churn Rate (%)")
        )
        fig = px.bar(
            payment_churn, x="PaymentMethod", y="Churn Rate (%)",
            title="Churn Rate by Payment Method", color="Churn Rate (%)",
            color_continuous_scale="Purples",
        )
        fig.update_xaxes(tickangle=25)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("💡 Business Insights & Retention Recommendations")
    st.markdown(
        """
        **Top Churn Drivers (typically observed in this dataset):**
        - Customers on **month-to-month contracts** churn far more than
          those on one/two-year contracts.
        - **Fiber optic** internet customers show higher churn, often tied
          to price sensitivity.
        - Customers **without Online Security or Tech Support** add-ons
          churn more frequently.
        - **Electronic check** payment users tend to churn more than
          automatic payment users.
        - **Low-tenure** customers (new customers) are the highest churn
          risk segment.

        **Retention Recommendations:**
        1. Offer incentives to shift month-to-month customers onto annual
           contracts (discounts, loyalty perks).
        2. Bundle Online Security / Tech Support into fiber plans at a
           reduced rate to increase stickiness.
        3. Encourage automatic payment methods with small billing credits.
        4. Launch a proactive onboarding/retention program for customers
           in their first 3–6 months.
        5. Use the prediction tool here to flag high-risk customers for
           targeted retention campaigns before they churn.
        """
    )


# --------------------------------------------------------------------------- #
# Sidebar Navigation
# --------------------------------------------------------------------------- #
def main() -> None:
    st.sidebar.title("📊 Churn Prediction")
    st.sidebar.markdown("Navigate the dashboard:")

    page = st.sidebar.radio(
        "Go to",
        ["Home", "About Dataset", "EDA", "Prediction", "Analytics"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.caption("Built with Streamlit • Scikit-learn • Plotly")

    pages = {
        "Home": page_home,
        "About Dataset": page_about_dataset,
        "EDA": page_eda,
        "Prediction": page_prediction,
        "Analytics": page_analytics,
    }
    pages[page]()


if __name__ == "__main__":
    main()
