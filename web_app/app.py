from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests


st.set_page_config(
    page_title="Loan Approval Analysis",
    layout="wide"
)


DATA_PATH = Path(__file__).parent.parent / "loan_data.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()
df_clean = df[df["person_age"] <= 100].copy()

st.sidebar.title("Project Navigation")


page = st.sidebar.selectbox(
    "Select a section",
    [
        "About",
        "Dataset",
        "Data Quality",
        "Descriptive Statistics",
        "General Overview",
        "Detailed Overview",
        "Data Transformation",
        "Hypotheses",
        "Discussion",
        "API"
    ]
)


if page == "About":
    st.title("Credit Risk and Loan Approval Analysis")

    st.write(
        """
        This project analyzes a loan approval dataset to understand
        which applicant and loan characteristics are connected with
        loan approval decisions, and how they are connected.

        The main goal of the project is to explore the dataset, check
        data quality and prepare it for analysis, create visualizations,
        engineer new features, and formulate and test hypotheses about
        factors related to loan approval and credit risk.
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Applications", f"{len(df):,}")
    col2.metric("Variables", df.shape[1])
    col3.metric(
        "Approval Rate",
        f"{df['loan_status'].mean() * 100:.1f}%"
    )

    st.subheader("Project Structure")

    st.write(
        """
        The web application presents all main stages of the analysis:
        dataset description, data quality and cleanup, descriptive
        statistics, general and detailed visualizations, data
        transformation, hypothesis testing, discussion, and access to
        the REST API.
        """
    )

    st.subheader("Team Contribution")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Kuznetsova Aleksandra 256")
        st.write(
            """
            Focused on dataset description, data quality check and
            cleanup, general overview, the FastAPI REST API, and the
            Telegram bot.
            """
        )

    with col2:
        st.markdown("#### Tymanian Syuzi 256")
        st.write(
            """
            Focused on descriptive statistics, detailed overview, data
            transformation, and the Streamlit web interface.
            """
        )

    st.write(
        """
        Each participant developed one hypothesis. Both team members
        reviewed each other's code, discussed intermediate results,
        and edited the final report.
        """
    )


elif page == "Dataset":
    st.title("Dataset Description")

    st.write(
        """
        The dataset belongs to the financial field and contains
        information about loan applications, approval decisions,
        and credit risk. It is a synthetic dataset.
        """
    )

    st.subheader("Dataset Size")

    col1, col2 = st.columns(2)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])

    st.subheader("Variables")

    st.write(
        """
        Numerical variables include applicant age, annual income,
        employment experience, loan amount, interest rate,
        loan-to-income ratio, credit history length, and credit score.

        Categorical variables include gender, education, home ownership,
        loan purpose, and previous loan defaults. `loan_status` is the
        target variable: 0 represents a rejected application and
        1 represents an approved application.
        """
    )

    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Non-Null Values": df.notna().sum().values
    })

    st.dataframe(
        column_info,
        hide_index=True,
        width="stretch"
    )

    st.subheader("First Rows")

    st.dataframe(
        df.head(),
        width="stretch"
    )    


elif page == "Data Quality":
    st.title("Data Quality and Cleanup")

    st.write(
        """
        We check the dataset for missing values, duplicated rows,
        incorrect data types, and unusual numerical values.
        """
    )

    missing_values = df.isna().sum()
    duplicated_rows = df.duplicated().sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("Missing Values", int(missing_values.sum()))
    col2.metric("Duplicated Rows", int(duplicated_rows))
    col3.metric(
        "Applicants Over 100",
        int((df["person_age"] > 100).sum())
    )

    st.subheader("Missing Values by Column")

    missing_table = pd.DataFrame({
        "Column": missing_values.index,
        "Missing Values": missing_values.values
    })

    st.dataframe(
        missing_table,
        hide_index=True,
        width="stretch"
    )

    st.subheader("Unusual Values")

    unusual_values = pd.DataFrame({
        "Check": [
            "Age over 100",
            "Age under 18",
            "Income zero or negative",
            "Loan amount zero or negative",
            "Loan percent income over 1",
            "Employment experience greater than age"
        ],
        "Number of Rows": [
            (df["person_age"] > 100).sum(),
            (df["person_age"] < 18).sum(),
            (df["person_income"] <= 0).sum(),
            (df["loan_amnt"] <= 0).sum(),
            (df["loan_percent_income"] > 1).sum(),
            (df["person_emp_exp"] > df["person_age"]).sum()
        ]
    })

    st.dataframe(
        unusual_values,
        hide_index=True,
        width="stretch"
    )

    st.subheader("Data Cleanup")

    st.write(
        """
        The dataset has no missing values or duplicated rows.
        Seven observations contain unrealistic applicant ages over
        100 years, so these rows are removed from further analysis.
        """
    )

    col1, col2 = st.columns(2)
    col1.metric("Rows Before Cleanup", f"{len(df):,}")
    col2.metric("Rows After Cleanup", f"{len(df_clean):,}")


elif page == "Descriptive Statistics":
    st.title("Descriptive Statistics")

    st.write(
        """
        We calculate descriptive statistics for the main numerical
        variables to understand their central values, variation,
        and range.
        """
    )

    numeric_columns = [
        "person_age",
        "person_income",
        "person_emp_exp",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        "credit_score"
    ]

    descriptive_stats = (
        df_clean[numeric_columns]
        .describe()
        .T
        .rename(
            columns={
                "25%": "Q1",
                "50%": "Median",
                "75%": "Q3"
            }
        )
    )

    descriptive_stats["IQR"] = (
        descriptive_stats["Q3"] -
        descriptive_stats["Q1"]
    )

    descriptive_stats = descriptive_stats[
        [
            "count",
            "mean",
            "std",
            "min",
            "Q1",
            "Median",
            "Q3",
            "max",
            "IQR"
        ]
    ].round(2)

    st.dataframe(
        descriptive_stats,
        width="stretch"
    )

    st.subheader("Main Observations")

    st.write(
        """
        The typical applicant is relatively young: the mean age is
        approximately 27.75 years and the median is 26 years.

        Annual income is strongly right-skewed. Its mean is about
        USD 79,908, while its median is about USD 67,046, indicating
        that a small number of high-income applicants increase the mean.

        The average requested loan is approximately USD 9,583 and the
        median is USD 8,000. The average interest rate is about 11%.

        The median requested loan represents approximately 12% of the
        applicant's annual income. Credit scores have a mean of about
        633 and a median of 640.
        """
    )    


elif page == "General Overview":
    st.title("General Overview")

    st.write(
        """
        We examine the distributions of annual income, loan amount,
        interest rate, and credit score.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df_clean["person_income"], bins=60, color="#7AA6C2")
        ax.set_title("Distribution of Annual Income")
        ax.set_xlabel("Annual Income (USD)")
        ax.set_ylabel("Number of Applicants")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df_clean["loan_amnt"], bins=50, color="#83B88B")
        ax.set_title("Distribution of Loan Amount")
        ax.set_xlabel("Loan Amount (USD)")
        ax.set_ylabel("Number of Applicants")
        st.pyplot(fig)
        plt.close(fig)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df_clean["loan_int_rate"], bins=30, color="#D6A76C")
        ax.set_title("Distribution of Interest Rate")
        ax.set_xlabel("Interest Rate (%)")
        ax.set_ylabel("Number of Applicants")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df_clean["credit_score"], bins=40, color="#A7C7E7")
        ax.set_title("Distribution of Credit Score")
        ax.set_xlabel("Credit Score")
        ax.set_ylabel("Number of Applicants")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Distribution Summary")

    st.write(
        """
        Annual income is strongly right-skewed because a small number
        of applicants have very high income values. Requested loan
        amounts are concentrated in the lower and middle ranges.

        Interest rates peak at approximately 11% and become less
        frequent at higher values. Credit scores are mostly
        concentrated around 650.
        """
    )

    st.subheader("Loan Status Distribution")

    loan_status_counts = (
        df_clean["loan_status"]
        .value_counts()
        .sort_index()
    )

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))

        ax.pie(
            loan_status_counts,
            labels=["Rejected", "Approved"],
            colors=["#D98C8C", "#83B88B"],
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title("Loan Status Distribution")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))

        ax.bar(
            ["Rejected", "Approved"],
            loan_status_counts,
            color=["#D98C8C", "#83B88B"]
        )

        ax.set_title("Number of Applications by Loan Status")
        ax.set_ylabel("Number of Applications")
        st.pyplot(fig)
        plt.close(fig)

    st.write(
        """
        Approximately 77.8% of applications were rejected and 22.2%
        were approved. Therefore, the target variable is imbalanced.
        """
    )

    st.subheader("Loan Amount Distribution")

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.boxplot(
        df_clean["loan_amnt"],
        vert=False,
        patch_artist=True,
        showmeans=True,
        boxprops=dict(
            facecolor="#A7C7E7",
            color="#4A6FA5"
        ),
        medianprops=dict(
            color="#D9534F",
            linewidth=2
        ),
        meanprops=dict(
            marker="o",
            markerfacecolor="#83B88B",
            markeredgecolor="#3D7A42",
            markersize=8
        ),
        flierprops=dict(
            marker="o",
            markerfacecolor="#C96B6B",
            markeredgecolor="#8F3F3F",
            markersize=5,
            alpha=0.5
        )
    )

    ax.set_title("Boxplot of Loan Amount")
    ax.set_xlabel("Loan Amount (USD)")
    ax.grid(axis="x", alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.write(
        """
        The median requested loan is approximately USD 8,000, while
        the mean is about USD 9,583. The higher mean indicates that
        large loan requests pull the average upward.
        """
    )

    st.subheader("Annual Income and Loan Amount")

    income_99 = df_clean["person_income"].quantile(0.99)

    filtered_income = df_clean[
        df_clean["person_income"] <= income_99
    ]

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.scatter(
            df_clean["person_income"],
            df_clean["loan_amnt"],
            alpha=0.2,
            s=10,
            color="#7AA6C2"
        )

        ax.set_title("Full Dataset")
        ax.set_xlabel("Annual Income (USD)")
        ax.set_ylabel("Loan Amount (USD)")

        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.scatter(
            filtered_income["person_income"],
            filtered_income["loan_amnt"],
            alpha=0.2,
            s=10,
            color="#7AA6C2"
        )

        ax.set_title("Top 1% of Income Excluded")
        ax.set_xlabel("Annual Income (USD)")
        ax.set_ylabel("Loan Amount (USD)")

        st.pyplot(fig)
        plt.close(fig)

    st.write(
        """
        Very high income values stretch the x-axis in the full plot.
        After excluding the top 1% for visualization, the main
        concentration of applicants and loan amounts becomes clearer.
        """
    )


elif page == "Detailed Overview":
    st.title("Detailed Overview")

    st.write(
        """
        We compare approval patterns across applicant and loan groups.
        These comparisons describe relationships observed in the
        dataset but do not establish causal relationships.
        """
    )

    st.subheader("Approval Rate by Loan Intent")

    approval_by_intent = (
        df_clean.groupby("loan_intent")["loan_status"]
        .mean()
        .sort_values(ascending=False)
        .mul(100)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        approval_by_intent.index,
        approval_by_intent.values,
        color="#7AA6C2"
    )

    ax.set_title("Approval Rate by Loan Intent")
    ax.set_xlabel("Loan Intent")
    ax.set_ylabel("Approval Rate (%)")
    ax.tick_params(axis="x", rotation=25)

    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.write(
        """
        Debt consolidation has the highest approval rate at about
        30.3%, while venture loans have the lowest rate at about 14.4%.
        This shows that approval rates differ across loan purposes.
        """
    )

    st.subheader("Approval Rate by Previous Defaults")

    defaults_summary = (
        df_clean
        .groupby("previous_loan_defaults_on_file")["loan_status"]
        .agg(["count", "sum", "mean"])
        .rename(
            columns={
                "count": "Applications",
                "sum": "Approved Applications",
                "mean": "Approval Rate (%)"
            }
        )
    )

    defaults_summary["Approval Rate (%)"] *= 100
    defaults_summary = defaults_summary.round(2)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(defaults_summary, width="stretch")

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))

        ax.bar(
            defaults_summary.index,
            defaults_summary["Approval Rate (%)"],
            color=["#83B88B", "#D98C8C"]
        )

        ax.set_title("Approval Rate by Previous Defaults")
        ax.set_xlabel("Previous Loan Defaults")
        ax.set_ylabel("Approval Rate (%)")
        ax.grid(axis="y", alpha=0.3)

        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.write(
        """
        Applicants with previous defaults have an approval rate of 0%,
        while applicants without previous defaults have an approval
        rate of approximately 45.2%. Previous default history is
        strongly associated with approval status in this dataset.
        """
    )


    st.subheader("Interest Rate by Loan Status")

    interest_by_status = (
        df_clean
        .groupby("loan_status")["loan_int_rate"]
        .agg(["mean", "median", "std"])
        .round(2)
    )

    interest_by_status.index = ["Rejected", "Approved"]

    approved_rates = df_clean.loc[
        df_clean["loan_status"] == 1,
        "loan_int_rate"
    ]

    rejected_rates = df_clean.loc[
        df_clean["loan_status"] == 0,
        "loan_int_rate"
    ]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(
            interest_by_status,
            width="stretch"
        )

    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.boxplot(
            [rejected_rates, approved_rates],
            tick_labels=["Rejected", "Approved"],
            patch_artist=True,
            boxprops=dict(
                facecolor="#A7C7E7",
                color="#4A6FA5"
            ),
            medianprops=dict(
                color="#D9534F",
                linewidth=2
            )
        )

        ax.set_title("Interest Rate by Loan Status")
        ax.set_xlabel("Loan Status")
        ax.set_ylabel("Interest Rate (%)")
        ax.grid(axis="y", alpha=0.3)

        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.write(
        """
        Approved applications have a higher median interest rate:
        approximately 12.98%, compared with 10.85% for rejected
        applications. The middle 50% of interest rates are also higher
        in the approved group.
        """
    )


    st.subheader("Loan Percent Income by Loan Status")

    loan_percent_by_status = (
        df_clean
        .groupby("loan_status")["loan_percent_income"]
        .agg(["mean", "median", "std"])
        .mul(100)
        .round(2)
    )

    loan_percent_by_status.index = ["Rejected", "Approved"]

    approved_loan_percent = df_clean.loc[
        df_clean["loan_status"] == 1,
        "loan_percent_income"
    ] * 100

    rejected_loan_percent = df_clean.loc[
        df_clean["loan_status"] == 0,
        "loan_percent_income"
    ] * 100

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(
            loan_percent_by_status,
            width="stretch"
        )

    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.boxplot(
            [rejected_loan_percent, approved_loan_percent],
            tick_labels=["Rejected", "Approved"],
            patch_artist=True,
            boxprops=dict(
                facecolor="#A7C7E7",
                color="#4A6FA5"
            ),
            medianprops=dict(
                color="#D9534F",
                linewidth=2
            ),
            flierprops=dict(
                marker="o",
                markerfacecolor="#C96B6B",
                markeredgecolor="#8F3F3F",
                markersize=4,
                alpha=0.4
            )
        )

        ax.set_title("Loan Percent Income by Loan Status")
        ax.set_xlabel("Loan Status")
        ax.set_ylabel(
            "Loan Amount as Percentage of Annual Income (%)"
        )
        ax.grid(axis="y", alpha=0.3)

        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.write(
        """
        The median requested loan represents 20% of annual income for
        approved applications and 11% for rejected applications. The
        approved group also has a wider distribution of this ratio.
        """
    )


    st.subheader("Approval Rate by Home Ownership")

    approval_by_home = (
        df_clean
        .groupby("person_home_ownership")["loan_status"]
        .agg(["count", "mean"])
        .rename(
            columns={
                "count": "Applications",
                "mean": "Approval Rate (%)"
            }
        )
    )

    approval_by_home["Approval Rate (%)"] *= 100

    approval_by_home = (
        approval_by_home
        .sort_values("Approval Rate (%)", ascending=False)
        .round(2)
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(
            approval_by_home,
            width="stretch"
        )

    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.bar(
            approval_by_home.index,
            approval_by_home["Approval Rate (%)"],
            color="#7AA6C2"
        )

        ax.set_title("Approval Rate by Home Ownership")
        ax.set_xlabel("Home Ownership")
        ax.set_ylabel("Approval Rate (%)")
        ax.grid(axis="y", alpha=0.3)

        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.write(
        """
        RENT applicants have an approval rate of approximately 32.4%,
        while MORTGAGE and OWN applicants have considerably lower rates.
        The OTHER category is small, so its result is less representative.
        """
    )


    st.subheader(
        "Annual Income and Loan Amount by Loan Status"
    )

    st.write(
        """
        Annual income is shown on the x-axis, requested loan amount on
        the y-axis, and loan status by color. The top 1% of annual
        income values are excluded to make the main concentration of
        applications easier to examine.
        """
    )

    income_99 = df_clean["person_income"].quantile(0.99)

    filtered_income = df_clean[
        df_clean["person_income"] <= income_99
    ]

    rejected = filtered_income[
        filtered_income["loan_status"] == 0
    ]

    approved = filtered_income[
        filtered_income["loan_status"] == 1
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        rejected["person_income"],
        rejected["loan_amnt"],
        color="#D98C8C",
        alpha=0.25,
        s=12,
        label="Rejected"
    )

    ax.scatter(
        approved["person_income"],
        approved["loan_amnt"],
        color="#83B88B",
        alpha=0.25,
        s=12,
        label="Approved"
    )

    ax.set_title(
        "Applicant Income and Loan Amount by Loan Status"
    )
    ax.set_xlabel("Annual Income (USD)")
    ax.set_ylabel("Loan Amount (USD)")
    ax.legend(title="Loan Status")
    ax.grid(alpha=0.2)

    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.write(
        """
        Applicants with higher incomes request both small and large
        loans. Approved applications are more concentrated among
        low- and middle-income applicants with relatively larger loan
        amounts. However, the groups overlap substantially, so income
        and loan amount do not clearly distinguish approval status.
        """
    )


elif page == "Data Transformation":
    st.title("Data Transformation")

    st.write(
        """
        We create two new columns from existing variables. These
        transformations make credit score and education easier to
        compare and will be used in hypothesis testing.
        """
    )

    st.subheader("Credit Score Group")

    transformed_data = df_clean.copy()

    transformed_data["credit_score_group"] = pd.cut(
        transformed_data["credit_score"],
        bins=[0, 600, 670, float("inf")],
        labels=[
            "Low (<600)",
            "Medium (600-669)",
            "High (670+)"
        ],
        right=False
    )

    st.write(
        """
        The numerical credit score is divided into three ordered
        categories: low, medium, and high.
        """
    )

    credit_score_example = transformed_data[
        ["credit_score", "credit_score_group"]
    ].head(10)

    st.dataframe(
        credit_score_example,
        hide_index=True,
        width="stretch"
    )

    credit_group_counts = (
        transformed_data["credit_score_group"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        credit_group_counts.index.astype(str),
        credit_group_counts.values,
        color=["#D98C8C", "#D6A76C", "#83B88B"]
    )

    ax.set_title("Number of Applicants by Credit Score Group")
    ax.set_xlabel("Credit Score Group")
    ax.set_ylabel("Number of Applicants")

    st.pyplot(fig, width="stretch")
    plt.close(fig)


    st.subheader("Education Level")

    education_order = {
        "High School": 1,
        "Associate": 2,
        "Bachelor": 3,
        "Master": 4,
        "Doctorate": 5
    }

    transformed_data["education_level"] = (
        transformed_data["person_education"]
        .map(education_order)
    )

    st.write(
        """
        Education is originally stored as text. We convert it into an
        ordered numerical variable, where a larger number represents
        a higher education level.
        """
    )

    education_table = (
        transformed_data[
            ["person_education", "education_level"]
        ]
        .drop_duplicates()
        .sort_values("education_level")
        .reset_index(drop=True)
    )

    st.dataframe(
        education_table,
        hide_index=True,
        width="stretch"
    )


elif page == "Hypotheses":
    st.title("Hypothesis Testing")

    hypothesis_data = df_clean.copy()

    hypothesis_data["credit_score_group"] = pd.cut(
        hypothesis_data["credit_score"],
        bins=[0, 600, 670, float("inf")],
        labels=[
            "Low (<600)",
            "Medium (600-669)",
            "High (670+)"
        ],
        right=False
    )

    education_order = {
        "High School": 1,
        "Associate": 2,
        "Bachelor": 3,
        "Master": 4,
        "Doctorate": 5
    }

    hypothesis_data["education_level"] = (
        hypothesis_data["person_education"]
        .map(education_order)
    )


    st.subheader("Hypothesis 1: Credit Score and Loan Intent")

    st.write(
        """
        Among applicants without previous loan defaults, applicants
        with higher credit scores are expected to have higher approval
        rates for every loan purpose.
        """
    )

    no_defaults = hypothesis_data[
        hypothesis_data["previous_loan_defaults_on_file"] == "No"
    ]

    hypothesis_1 = (
        no_defaults
        .pivot_table(
            index="credit_score_group",
            columns="loan_intent",
            values="loan_status",
            aggfunc="mean",
            observed=False
        )
        .mul(100)
    )

    st.dataframe(
        hypothesis_1.round(2),
        width="stretch"
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    for loan_intent in hypothesis_1.columns:
        ax.plot(
            hypothesis_1.index.astype(str),
            hypothesis_1[loan_intent],
            marker="o",
            linewidth=2,
            label=loan_intent
        )

    ax.set_title(
        "Approval Rate by Credit Score Group and Loan Intent\n"
        "Applicants Without Previous Defaults"
    )
    ax.set_xlabel("Credit Score Group")
    ax.set_ylabel("Approval Rate (%)")
    ax.legend(
        title="Loan Intent",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )
    ax.grid(alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.write(
        """
        The hypothesis was not supported. Approval rates decrease
        rather than increase across the credit score groups for every
        loan purpose. Debt consolidation and medical applications
        generally have higher approval rates, while venture
        applications have the lowest.
        """
    )


    st.subheader(
        "Hypothesis 2: Education Level and Home Ownership"
    )

    st.write(
        """
        Among applicants without previous defaults, applicants who
        own their homes are expected to have higher approval rates
        than renters and mortgage holders at each education level.
        """
    )

    no_defaults_home = hypothesis_data[
        (
            hypothesis_data[
                "previous_loan_defaults_on_file"
            ] == "No"
        )
        &
        (
            hypothesis_data["person_home_ownership"]
            .isin(["RENT", "MORTGAGE", "OWN"])
        )
    ]

    hypothesis_2 = (
        no_defaults_home
        .pivot_table(
            index=[
                "education_level",
                "person_education"
            ],
            columns="person_home_ownership",
            values="loan_status",
            aggfunc="mean",
            observed=False
        )
        .mul(100)
        .dropna(how="all")
    )

    hypothesis_2 = hypothesis_2[
        ["RENT", "MORTGAGE", "OWN"]
    ]

    st.dataframe(
        hypothesis_2.round(2),
        width="stretch"
    )

    hypothesis_2_plot = hypothesis_2.copy()

    hypothesis_2_plot.index = (
        hypothesis_2_plot.index
        .get_level_values("person_education")
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "RENT": "#7AA6C2",
        "MORTGAGE": "#D6A76C",
        "OWN": "#83B88B"
    }

    for ownership in hypothesis_2_plot.columns:
        ax.plot(
            hypothesis_2_plot.index,
            hypothesis_2_plot[ownership],
            marker="o",
            linewidth=2,
            color=colors[ownership],
            label=ownership
        )

    ax.set_title(
        "Approval Rate by Education Level and Home Ownership\n"
        "Applicants Without Previous Defaults"
    )
    ax.set_xlabel("Education Level")
    ax.set_ylabel("Approval Rate (%)")
    ax.legend(title="Home Ownership")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.write(
        """
        The hypothesis was not supported. Homeowners have the lowest
        approval rates at every education level, while renters have
        the highest rates. Approval rates generally decrease slightly
        as education level increases. The only exception is the
        MORTGAGE group, where the rate rises from Master to Doctorate.
        """
    )

    
elif page == "Discussion":
    st.title("Discussion")

    st.write(
        """
        The analysis identified several patterns associated with loan
        approval status. Previous default history shows the strongest
        difference: applicants with previous defaults have an approval
        rate of 0%, while applicants without previous defaults have a
        substantially higher approval rate.
        """
    )

    st.subheader("Main Findings")

    findings = [
        (
            "Loan purpose",
            "Debt consolidation and medical applications have higher "
            "approval rates, while venture applications have the "
            "lowest rate."
        ),
        (
            "Interest rate",
            "Approved applications have a higher median interest rate "
            "than rejected applications."
        ),
        (
            "Loan percent income",
            "Approved applications request loans representing a larger "
            "share of annual income."
        ),
        (
            "Home ownership",
            "Renters have higher approval rates than mortgage holders "
            "and homeowners in this dataset."
        ),
        (
            "Credit score",
            "Among applicants without previous defaults, approval rates "
            "decrease across the selected credit score groups."
        ),
        (
            "Education",
            "Higher education levels are not associated with higher "
            "approval rates in the tested groups."
        ),
        (
            "Employment experience",
            "Employment experience is strongly correlated with age "
            "(approximately 0.95), while approval rates differ only "
            "slightly across experience groups. It therefore provides "
            "limited additional insight in this dataset."
        )
    ]

    for title, description in findings:
        st.markdown(f"**{title}:** {description}")

    st.subheader("Hypothesis Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Hypothesis 1")
        st.error("Not supported")
        st.write(
            """
            Higher credit score groups did not have higher approval
            rates for any loan purpose among applicants without
            previous defaults.
            """
        )

    with col2:
        st.markdown("#### Hypothesis 2")
        st.error("Not supported")
        st.write(
            """
            Homeowners did not have higher approval rates. Renters had
            the highest approval rate at every education level.
            """
        )

    st.subheader("Final Conclusion")

    st.write(
        """
        Loan approval patterns in this synthetic dataset differ from
        several initial expectations. Approval status is associated
        with a combination of applicant characteristics and loan
        conditions rather than one variable alone. The results describe
        patterns in this dataset and should not be interpreted as causal
        relationships.
        """
    )


elif page == "API":
    st.title("REST API")

    api_url = "https://loan-approval-api-spni.onrender.com"

    try:
        response = requests.get(api_url, timeout=3)
        response.raise_for_status()
        st.success("FastAPI service is running")
    except requests.RequestException:
        st.error(
            "FastAPI is not running. Start it with: "
            "uvicorn api:app --reload"
        )
        st.stop()

    st.subheader("Get Loan Applications")

    col1, col2 = st.columns(2)

    with col1:
        limit = st.number_input(
            "Number of applications",
            min_value=1,
            max_value=100,
            value=10
        )

        offset = st.number_input(
            "Rows to skip",
            min_value=0,
            value=0
        )

    with col2:
        status_label = st.selectbox(
            "Loan status",
            ["All", "Rejected", "Approved"]
        )

        loan_intent = st.selectbox(
            "Loan intent",
            ["All"] + sorted(df["loan_intent"].unique())
        )

    if st.button("Get applications"):
        parameters = {
            "limit": int(limit),
            "offset": int(offset)
        }

        if status_label == "Rejected":
            parameters["loan_status"] = 0
        elif status_label == "Approved":
            parameters["loan_status"] = 1

        if loan_intent != "All":
            parameters["loan_intent"] = loan_intent

        try:
            response = requests.get(
                f"{api_url}/loans",
                params=parameters,
                timeout=10
            )

            response.raise_for_status()
            result = response.json()

            st.metric(
                "Matching Applications",
                f"{result['total_matching']:,}"
            )

            st.dataframe(
                pd.DataFrame(result["applications"]),
                hide_index=True,
                width="stretch"
            )

        except requests.RequestException as error:
            st.error(f"API request failed: {error}")

    st.subheader("Create a New Loan Application")

    with st.form("new_loan_form"):
        col1, col2 = st.columns(2)

        with col1:
            person_age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30
            )

            person_gender = st.selectbox(
                "Gender",
                ["female", "male"]
            )

            person_education = st.selectbox(
                "Education",
                [
                    "High School",
                    "Associate",
                    "Bachelor",
                    "Master",
                    "Doctorate"
                ]
            )

            person_income = st.number_input(
                "Annual Income (USD)",
                min_value=1.0,
                value=75000.0
            )

            person_emp_exp = st.number_input(
                "Employment Experience",
                min_value=0,
                value=7
            )

            person_home_ownership = st.selectbox(
                "Home Ownership",
                ["RENT", "MORTGAGE", "OWN", "OTHER"]
            )

            loan_amnt = st.number_input(
                "Loan Amount (USD)",
                min_value=1.0,
                value=10000.0
            )

        with col2:
            loan_intent = st.selectbox(
                "Loan Intent",
                sorted(df["loan_intent"].unique())
            )

            loan_int_rate = st.number_input(
                "Interest Rate (%)",
                min_value=0.01,
                value=11.5
            )

            credit_history = st.number_input(
                "Credit History Length",
                min_value=0.0,
                value=6.0
            )

            credit_score = st.number_input(
                "Credit Score",
                min_value=300,
                max_value=850,
                value=650
            )

            previous_defaults = st.selectbox(
                "Previous Loan Defaults",
                ["No", "Yes"]
            )

            status_label_new = st.selectbox(
                "Loan Status",
                ["Rejected", "Approved"]
            )

        submit_application = st.form_submit_button(
            "Create application"
        )

    if submit_application:
        loan_percent_income = loan_amnt / person_income

        new_application = {
            "person_age": float(person_age),
            "person_gender": person_gender,
            "person_education": person_education,
            "person_income": float(person_income),
            "person_emp_exp": int(person_emp_exp),
            "person_home_ownership": person_home_ownership,
            "loan_amnt": float(loan_amnt),
            "loan_intent": loan_intent,
            "loan_int_rate": float(loan_int_rate),
            "loan_percent_income": float(loan_percent_income),
            "cb_person_cred_hist_length": float(credit_history),
            "credit_score": int(credit_score),
            "previous_loan_defaults_on_file": previous_defaults,
            "loan_status": (
                1 if status_label_new == "Approved" else 0
            )
        }

        try:
            response = requests.post(
                f"{api_url}/loans",
                json=new_application,
                timeout=10
            )

            response.raise_for_status()
            result = response.json()

            st.success(result["message"])
            st.json(result["application"])

        except requests.RequestException as error:
            st.error(f"Failed to create application: {error}")

    st.subheader("API Documentation")

    st.markdown(
        "[Open interactive FastAPI documentation]"
        "(https://loan-approval-api-spni.onrender.com/docs)"
    )
