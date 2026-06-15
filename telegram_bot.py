import asyncio
import os
import warnings
from io import BytesIO
from pathlib import Path

import matplotlib
import pandas as pd
import requests
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.warnings import PTBUserWarning
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


warnings.filterwarnings(
    "ignore",
    message="If 'per_message=False'.*",
    category=PTBUserWarning,
)

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
STREAMLIT_URL = os.getenv("STREAMLIT_APP_URL")
API_URL = "https://loan-approval-api-spni.onrender.com"
GITHUB_URL = "https://github.com/schqxs/loan-approval-analysis"
DATA_PATH = Path(__file__).parent / "loan_data.csv"

FORM_INPUT = 1
OFFSET_INPUT = 2

REJECTED_COLOR = "#D98C8C"
APPROVED_COLOR = "#83B88B"
LOAN_INTENTS = [
    "All",
    "DEBTCONSOLIDATION",
    "EDUCATION",
    "HOMEIMPROVEMENT",
    "MEDICAL",
    "PERSONAL",
    "VENTURE",
]

df = pd.read_csv(DATA_PATH)
df_clean = df[df["person_age"] <= 100].copy()


def main_menu():
    website_button = (
        InlineKeyboardButton("Open Streamlit Site", url=STREAMLIT_URL)
        if STREAMLIT_URL
        else InlineKeyboardButton(
            "Open Streamlit Site",
            callback_data="website_missing",
        )
    )

    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("About", callback_data="about"),
            InlineKeyboardButton("Dataset", callback_data="dataset"),
        ],
        [
            InlineKeyboardButton(
                "Data Quality",
                callback_data="data_quality",
            ),
            InlineKeyboardButton(
                "Statistics",
                callback_data="statistics",
            ),
        ],
        [
            InlineKeyboardButton(
                "Main Results",
                callback_data="results",
            ),
            InlineKeyboardButton(
                "Hypotheses",
                callback_data="hypotheses",
            ),
        ],
        [
            InlineKeyboardButton(
                "View Applications",
                callback_data="applications",
            ),
            InlineKeyboardButton(
                "Add New Person",
                callback_data="add_application",
            ),
        ],
        [
            InlineKeyboardButton("Charts", callback_data="charts"),
            InlineKeyboardButton(
                "Check API",
                callback_data="api_status",
            ),
        ],
        [
            website_button,
        ],
        [
            InlineKeyboardButton(
                "API Documentation",
                url=f"{API_URL}/docs",
            ),
            InlineKeyboardButton("GitHub", url=GITHUB_URL),
        ],
    ])


def back_button():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Back to Menu", callback_data="menu")]
    ])


def default_application_filters():
    return {
        "status": "all",
        "intent": "All",
        "limit": 5,
        "offset": 0,
    }


def applications_menu(selected_filters):
    status_labels = {
        "all": "All",
        "0": "Rejected",
        "1": "Approved",
    }

    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                f"Status: {status_labels[selected_filters['status']]}",
                callback_data="filter_status",
            ),
        ],
        [
            InlineKeyboardButton(
                f"Loan intent: {selected_filters['intent']}",
                callback_data="filter_intent",
            ),
        ],
        [
            InlineKeyboardButton(
                f"Number of rows: {selected_filters['limit']}",
                callback_data="filter_limit",
            ),
            InlineKeyboardButton(
                f"Rows to skip: {selected_filters['offset']}",
                callback_data="filter_offset",
            ),
        ],
        [
            InlineKeyboardButton(
                "Show Applications",
                callback_data="apps_show",
            ),
            InlineKeyboardButton(
                "Reset Filters",
                callback_data="filters_reset",
            ),
        ],
        [InlineKeyboardButton("Back to Menu", callback_data="menu")],
    ])


def status_filter_menu():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("All", callback_data="status_all"),
            InlineKeyboardButton(
                "Rejected",
                callback_data="status_0",
            ),
            InlineKeyboardButton(
                "Approved",
                callback_data="status_1",
            ),
        ],
        [
            InlineKeyboardButton(
                "Back to Filters",
                callback_data="applications",
            )
        ],
    ])


def intent_filter_menu():
    rows = []
    for index in range(0, len(LOAN_INTENTS), 2):
        rows.append([
            InlineKeyboardButton(
                intent.title(),
                callback_data=f"intent_{intent}",
            )
            for intent in LOAN_INTENTS[index:index + 2]
        ])

    rows.append([
        InlineKeyboardButton(
            "Back to Filters",
            callback_data="applications",
        )
    ])
    return InlineKeyboardMarkup(rows)


def limit_filter_menu():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("3", callback_data="limit_3"),
            InlineKeyboardButton("5", callback_data="limit_5"),
            InlineKeyboardButton("10", callback_data="limit_10"),
        ],
        [
            InlineKeyboardButton(
                "Back to Filters",
                callback_data="applications",
            )
        ],
    ])


def charts_menu():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "General Overview",
                callback_data="charts_overview",
            ),
            InlineKeyboardButton(
                "Detailed Analysis",
                callback_data="charts_detailed",
            ),
        ],
        [
            InlineKeyboardButton(
                "Hypothesis Graphs",
                callback_data="charts_hypotheses",
            ),
        ],
        [InlineKeyboardButton("Back to Menu", callback_data="menu")],
    ])


def overview_charts_menu():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "Annual Income",
                callback_data="chart_income_dist",
            ),
            InlineKeyboardButton(
                "Loan Amount",
                callback_data="chart_loan_dist",
            ),
        ],
        [
            InlineKeyboardButton(
                "Interest Rate",
                callback_data="chart_interest_dist",
            ),
            InlineKeyboardButton(
                "Credit Score",
                callback_data="chart_credit_dist",
            ),
        ],
        [
            InlineKeyboardButton(
                "Loan Status",
                callback_data="chart_status",
            ),
            InlineKeyboardButton(
                "Loan Boxplot",
                callback_data="chart_loan_box",
            ),
        ],
        [
            InlineKeyboardButton(
                "Income vs Loan",
                callback_data="chart_income_loan",
            ),
        ],
        [InlineKeyboardButton("Back to Charts", callback_data="charts")],
    ])


def detailed_charts_menu():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "Loan Intent",
                callback_data="chart_intent",
            ),
            InlineKeyboardButton(
                "Previous Defaults",
                callback_data="chart_defaults",
            ),
        ],
        [
            InlineKeyboardButton(
                "Interest by Status",
                callback_data="chart_interest_status",
            ),
            InlineKeyboardButton(
                "Loan Burden",
                callback_data="chart_burden",
            ),
        ],
        [
            InlineKeyboardButton(
                "Home Ownership",
                callback_data="chart_home",
            ),
            InlineKeyboardButton(
                "Credit Groups",
                callback_data="chart_credit_groups",
            ),
        ],
        [
            InlineKeyboardButton(
                "Income + Loan + Status",
                callback_data="chart_income_loan_status",
            ),
        ],
        [InlineKeyboardButton("Back to Charts", callback_data="charts")],
    ])


def hypothesis_charts_menu():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "Hypothesis 1",
                callback_data="chart_hypothesis_1",
            ),
            InlineKeyboardButton(
                "Hypothesis 2",
                callback_data="chart_hypothesis_2",
            ),
        ],
        [InlineKeyboardButton("Back to Charts", callback_data="charts")],
    ])


def project_pages():
    return {
        "about": (
            "ABOUT THE PROJECT\n\n"
            "The project studies which applicant and loan characteristics "
            "are associated with loan approval decisions and credit risk.\n\n"
            "It combines data cleaning, exploratory analysis, feature "
            "engineering, hypothesis testing, a Streamlit interface, a "
            "FastAPI REST API, and this Telegram interface."
        ),
        "dataset": (
            "DATASET\n\n"
            "The financial dataset contains 45,000 loan applications and "
            "14 original columns.\n\n"
            "Applicant data: age, gender, education, annual income, "
            "employment experience, and home ownership.\n\n"
            "Loan and credit data: loan amount, purpose, interest rate, "
            "loan-to-income ratio, credit history, credit score, previous "
            "defaults, and approval status."
        ),
        "data_quality": (
            "DATA QUALITY\n\n"
            "No missing values or duplicate rows were found. Seven records "
            "contained unrealistic ages above 100 and were excluded from "
            "the cleaned dataset.\n\n"
            "Categorical values and numerical ranges were also checked for "
            "inconsistencies."
        ),
        "statistics": (
            "DESCRIPTIVE STATISTICS\n\n"
            "The typical applicant is relatively young: median age is about "
            "26 years. Median annual income is about 67,046 USD, and median "
            "requested loan amount is 8,000 USD.\n\n"
            "The median interest rate is about 11.01%, while the median "
            "credit score is 640."
        ),
        "results": (
            "MAIN RESULTS\n\n"
            "Previous default history shows the strongest difference: the "
            "approval rate is 0% for applicants with previous defaults in "
            "this dataset.\n\n"
            "Approval patterns also differ by loan-to-income ratio, interest "
            "rate, home ownership, credit score group, loan purpose, income, "
            "and employment experience."
        ),
        "hypotheses": (
            "HYPOTHESIS RESULTS\n\n"
            "Hypothesis 1 was not supported. Among applicants without "
            "previous defaults, higher credit score groups did not have "
            "higher approval rates for any loan purpose.\n\n"
            "Hypothesis 2 was also not supported. At every education level, "
            "renters had the highest approval rate and homeowners the lowest. "
            "Only the mortgage group increased slightly from Master's to "
            "Doctorate level."
        ),
    }


def format_applications(applications, total, offset):
    if not applications:
        return "No applications were returned for this filter."

    lines = [
        "APPLICATION EXAMPLES\n"
        f"Showing {offset + 1}-{offset + len(applications)} of "
        f"{total:,}\n"
    ]

    for number, application in enumerate(
        applications,
        start=offset + 1,
    ):
        status = (
            "Approved"
            if application["loan_status"] == 1
            else "Rejected"
        )
        loan_share = application["loan_percent_income"] * 100

        lines.append(
            f"{number}. {status} | {application['loan_intent']}\n"
            f"Age: {application['person_age']:.0f}, "
            f"Income: {application['person_income']:,.0f} USD\n"
            f"Loan: {application['loan_amnt']:,.0f} USD, "
            f"Burden: {loan_share:.1f}%\n"
            f"Credit score: {application['credit_score']}, "
            f"Previous defaults: "
            f"{application['previous_loan_defaults_on_file']}"
        )

    return "\n\n".join(lines)


def pagination_keyboard(offset, total, limit):
    buttons = []

    if offset > 0:
        previous_offset = max(0, offset - limit)
        buttons.append(
            InlineKeyboardButton(
                "Previous",
                callback_data=f"apps_page_{previous_offset}",
            )
        )

    if offset + limit < total:
        buttons.append(
            InlineKeyboardButton(
                "Next",
                callback_data=f"apps_page_{offset + limit}",
            )
        )

    rows = []
    if buttons:
        rows.append(buttons)
    rows.append([
        InlineKeyboardButton(
            "Change Filter",
            callback_data="applications",
        )
    ])
    rows.append([
        InlineKeyboardButton("Back to Menu", callback_data="menu")
    ])

    return InlineKeyboardMarkup(rows)


def create_chart(chart_name):
    fig, ax = plt.subplots(figsize=(8, 5))

    if chart_name == "income_dist":
        ax.hist(
            df_clean["person_income"],
            bins=60,
            color="#7AA6C2",
        )
        ax.set_title("Distribution of Annual Income")
        ax.set_xlabel("Annual Income (USD)")
        ax.set_ylabel("Number of Applicants")
        caption = (
            "Annual income is strongly right-skewed because a small "
            "number of applicants have very high incomes."
        )

    elif chart_name == "loan_dist":
        ax.hist(
            df_clean["loan_amnt"],
            bins=50,
            color=APPROVED_COLOR,
        )
        ax.set_title("Distribution of Loan Amount")
        ax.set_xlabel("Loan Amount (USD)")
        ax.set_ylabel("Number of Applicants")
        caption = (
            "Requested loan amounts are concentrated in the lower and "
            "middle ranges."
        )

    elif chart_name == "interest_dist":
        ax.hist(
            df_clean["loan_int_rate"],
            bins=30,
            color="#D6A76C",
        )
        ax.set_title("Distribution of Interest Rate")
        ax.set_xlabel("Interest Rate (%)")
        ax.set_ylabel("Number of Applicants")
        caption = (
            "Interest rates peak at approximately 11% and become less "
            "frequent at higher values."
        )

    elif chart_name == "credit_dist":
        ax.hist(
            df_clean["credit_score"],
            bins=40,
            color="#A7C7E7",
        )
        ax.set_title("Distribution of Credit Score")
        ax.set_xlabel("Credit Score")
        ax.set_ylabel("Number of Applicants")
        caption = (
            "Credit scores are mostly concentrated around 650."
        )

    elif chart_name == "status":
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        counts = (
            df_clean["loan_status"]
            .value_counts()
            .reindex([0, 1])
        )
        axes[0].pie(
            counts,
            labels=["Rejected", "Approved"],
            colors=[REJECTED_COLOR, APPROVED_COLOR],
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[0].set_title("Loan Status Distribution")
        axes[1].bar(
            ["Rejected", "Approved"],
            counts,
            color=[REJECTED_COLOR, APPROVED_COLOR],
        )
        axes[1].set_title("Number of Applications")
        axes[1].set_ylabel("Number of Applications")
        axes[1].grid(axis="y", alpha=0.25)
        caption = (
            "Rejected applications form the larger group: approximately "
            "77.8% of all observations."
        )

    elif chart_name == "loan_box":
        ax.boxplot(
            df_clean["loan_amnt"],
            vert=False,
            patch_artist=True,
            showmeans=True,
            boxprops={
                "facecolor": "#A7C7E7",
                "color": "#4A6FA5",
            },
            medianprops={
                "color": "#D9534F",
                "linewidth": 2,
            },
            meanprops={
                "marker": "o",
                "markerfacecolor": APPROVED_COLOR,
                "markeredgecolor": "#3D7A42",
                "markersize": 7,
            },
            flierprops={
                "marker": "o",
                "markerfacecolor": "#C96B6B",
                "markeredgecolor": "#8F3F3F",
                "markersize": 4,
                "alpha": 0.4,
            },
        )
        ax.set_title("Boxplot of Loan Amount")
        ax.set_xlabel("Loan Amount (USD)")
        caption = (
            "The median requested loan is about 8,000 USD. Larger loan "
            "requests pull the mean upward."
        )

    elif chart_name == "income_loan":
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        income_99 = df_clean["person_income"].quantile(0.99)
        filtered_income = df_clean[
            df_clean["person_income"] <= income_99
        ]
        axes[0].scatter(
            df_clean["person_income"],
            df_clean["loan_amnt"],
            alpha=0.2,
            s=10,
            color="#7AA6C2",
        )
        axes[0].set_title("Full Dataset")
        axes[1].scatter(
            filtered_income["person_income"],
            filtered_income["loan_amnt"],
            alpha=0.2,
            s=10,
            color="#7AA6C2",
        )
        axes[1].set_title("Top 1% of Income Excluded")
        for current_ax in axes:
            current_ax.set_xlabel("Annual Income (USD)")
            current_ax.set_ylabel("Loan Amount (USD)")
            current_ax.grid(alpha=0.2)
        caption = (
            "Very high incomes stretch the full plot. The second panel "
            "makes the main concentration of applications easier to see."
        )

    elif chart_name == "intent":
        approval_by_intent = (
            df_clean
            .groupby("loan_intent")["loan_status"]
            .mean()
            .sort_values(ascending=False)
            .mul(100)
        )
        ax.bar(
            approval_by_intent.index,
            approval_by_intent.values,
            color="#7AA6C2",
        )
        ax.set_title("Approval Rate by Loan Intent")
        ax.set_xlabel("Loan Intent")
        ax.set_ylabel("Approval Rate (%)")
        ax.tick_params(axis="x", rotation=25)
        caption = (
            "Debt consolidation has the highest approval rate, while "
            "venture loans have the lowest."
        )

    elif chart_name == "defaults":
        approval_by_defaults = (
            df_clean
            .groupby("previous_loan_defaults_on_file")["loan_status"]
            .mean()
            .mul(100)
            .reindex(["No", "Yes"])
        )
        ax.bar(
            approval_by_defaults.index,
            approval_by_defaults.values,
            color=[APPROVED_COLOR, REJECTED_COLOR],
        )
        ax.set_title("Approval Rate by Previous Defaults")
        ax.set_xlabel("Previous Loan Defaults")
        ax.set_ylabel("Approval Rate (%)")
        caption = (
            "Applicants with previous defaults have a 0% approval rate, "
            "compared with about 45.2% for applicants without defaults."
        )

    elif chart_name == "interest_status":
        rejected = df_clean.loc[
            df_clean["loan_status"] == 0,
            "loan_int_rate",
        ]
        approved = df_clean.loc[
            df_clean["loan_status"] == 1,
            "loan_int_rate",
        ]
        boxplot = ax.boxplot(
            [rejected, approved],
            tick_labels=["Rejected", "Approved"],
            patch_artist=True,
        )
        boxplot["boxes"][0].set_facecolor(REJECTED_COLOR)
        boxplot["boxes"][1].set_facecolor(APPROVED_COLOR)
        ax.set_title("Interest Rate by Loan Status")
        ax.set_xlabel("Loan Status")
        ax.set_ylabel("Interest Rate (%)")
        caption = (
            "Approved applications have a higher median interest rate: "
            "about 12.98%, compared with 10.85% for rejected applications."
        )

    elif chart_name == "burden":
        rejected = df_clean.loc[
            df_clean["loan_status"] == 0,
            "loan_percent_income",
        ] * 100
        approved = df_clean.loc[
            df_clean["loan_status"] == 1,
            "loan_percent_income",
        ] * 100
        boxplot = ax.boxplot(
            [rejected, approved],
            tick_labels=["Rejected", "Approved"],
            patch_artist=True,
        )
        boxplot["boxes"][0].set_facecolor(REJECTED_COLOR)
        boxplot["boxes"][1].set_facecolor(APPROVED_COLOR)
        ax.set_title("Loan Percent Income by Loan Status")
        ax.set_xlabel("Loan Status")
        ax.set_ylabel("Loan Amount as Percentage of Income (%)")
        caption = (
            "The median loan burden is approximately 11% for rejected and "
            "20% for approved applications in this dataset."
        )

    elif chart_name == "home":
        approval_by_home = (
            df_clean
            .groupby("person_home_ownership")["loan_status"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )
        ax.bar(
            approval_by_home.index,
            approval_by_home.values,
            color="#7AA6C2",
        )
        ax.set_title("Approval Rate by Home Ownership")
        ax.set_xlabel("Home Ownership")
        ax.set_ylabel("Approval Rate (%)")
        caption = (
            "Renters have a substantially higher approval rate than "
            "mortgage holders and homeowners in this dataset."
        )

    elif chart_name == "income_loan_status":
        income_99 = df_clean["person_income"].quantile(0.99)
        filtered = df_clean[
            df_clean["person_income"] <= income_99
        ]
        rejected = filtered[filtered["loan_status"] == 0]
        approved = filtered[filtered["loan_status"] == 1]
        ax.scatter(
            rejected["person_income"],
            rejected["loan_amnt"],
            color=REJECTED_COLOR,
            alpha=0.25,
            s=12,
            label="Rejected",
        )
        ax.scatter(
            approved["person_income"],
            approved["loan_amnt"],
            color=APPROVED_COLOR,
            alpha=0.25,
            s=12,
            label="Approved",
        )
        ax.set_title("Income and Loan Amount by Loan Status")
        ax.set_xlabel("Annual Income (USD)")
        ax.set_ylabel("Loan Amount (USD)")
        ax.legend(title="Loan Status")
        caption = (
            "Income, requested loan amount, and approval status overlap "
            "substantially, so no single clear boundary separates the groups."
        )

    elif chart_name == "credit_groups":
        credit_groups = pd.cut(
            df_clean["credit_score"],
            bins=[0, 600, 670, float("inf")],
            labels=[
                "Low (<600)",
                "Medium (600-669)",
                "High (670+)",
            ],
            right=False,
        )
        counts = credit_groups.value_counts().sort_index()
        ax.bar(
            counts.index.astype(str),
            counts.values,
            color=[REJECTED_COLOR, "#D6A76C", APPROVED_COLOR],
        )
        ax.set_title("Number of Applicants by Credit Score Group")
        ax.set_xlabel("Credit Score Group")
        ax.set_ylabel("Number of Applicants")
        caption = (
            "Credit scores were transformed into three ordered groups for "
            "comparison and hypothesis testing."
        )

    elif chart_name == "hypothesis_1":
        hypothesis_data = df_clean.copy()
        hypothesis_data["credit_score_group"] = pd.cut(
            hypothesis_data["credit_score"],
            bins=[0, 600, 670, float("inf")],
            labels=[
                "Low (<600)",
                "Medium (600-669)",
                "High (670+)",
            ],
            right=False,
        )
        no_defaults = hypothesis_data[
            hypothesis_data[
                "previous_loan_defaults_on_file"
            ] == "No"
        ]
        hypothesis_1 = (
            no_defaults
            .pivot_table(
                index="credit_score_group",
                columns="loan_intent",
                values="loan_status",
                aggfunc="mean",
                observed=False,
            )
            .mul(100)
        )
        for loan_intent in hypothesis_1.columns:
            ax.plot(
                hypothesis_1.index.astype(str),
                hypothesis_1[loan_intent],
                marker="o",
                linewidth=2,
                label=loan_intent,
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
            loc="upper left",
        )
        caption = (
            "Hypothesis 1 was not supported: approval rates decrease across "
            "credit score groups for every loan purpose."
        )

    elif chart_name == "hypothesis_2":
        hypothesis_data = df_clean.copy()
        education_order = {
            "High School": 1,
            "Associate": 2,
            "Bachelor": 3,
            "Master": 4,
            "Doctorate": 5,
        }
        hypothesis_data["education_level"] = (
            hypothesis_data["person_education"]
            .map(education_order)
        )
        no_defaults = hypothesis_data[
            (
                hypothesis_data[
                    "previous_loan_defaults_on_file"
                ] == "No"
            )
            & (
                hypothesis_data["person_home_ownership"]
                .isin(["RENT", "MORTGAGE", "OWN"])
            )
        ]
        hypothesis_2 = (
            no_defaults
            .pivot_table(
                index=["education_level", "person_education"],
                columns="person_home_ownership",
                values="loan_status",
                aggfunc="mean",
                observed=False,
            )
            .mul(100)
            .dropna(how="all")
        )[["RENT", "MORTGAGE", "OWN"]]
        hypothesis_2.index = (
            hypothesis_2.index
            .get_level_values("person_education")
        )
        colors = {
            "RENT": "#7AA6C2",
            "MORTGAGE": "#D6A76C",
            "OWN": APPROVED_COLOR,
        }
        for ownership in hypothesis_2.columns:
            ax.plot(
                hypothesis_2.index,
                hypothesis_2[ownership],
                marker="o",
                linewidth=2,
                color=colors[ownership],
                label=ownership,
            )
        ax.set_title(
            "Approval Rate by Education and Home Ownership\n"
            "Applicants Without Previous Defaults"
        )
        ax.set_xlabel("Education Level")
        ax.set_ylabel("Approval Rate (%)")
        ax.legend(title="Home Ownership")
        caption = (
            "Hypothesis 2 was not supported: renters have the highest and "
            "homeowners the lowest approval rates at every education level."
        )

    else:
        plt.close(fig)
        raise ValueError(f"Unknown chart: {chart_name}")

    if chart_name not in {"status", "income_loan"}:
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    image = BytesIO()
    fig.savefig(image, format="png", dpi=150, bbox_inches="tight")
    image.seek(0)
    plt.close(fig)

    return image, caption


FORM_FIELDS = [
    (
        "person_age",
        "Enter applicant age (18-100):",
        lambda value: float(value)
        if 18 <= float(value) <= 100
        else None,
    ),
    (
        "person_gender",
        "Enter gender: female or male",
        lambda value: value.lower()
        if value.lower() in {"female", "male"}
        else None,
    ),
    (
        "person_education",
        "Enter education: High School, Associate, Bachelor, Master, "
        "or Doctorate",
        lambda value: next(
            (
                choice
                for choice in [
                    "High School",
                    "Associate",
                    "Bachelor",
                    "Master",
                    "Doctorate",
                ]
                if choice.lower() == value.lower()
            ),
            None,
        ),
    ),
    (
        "person_income",
        "Enter annual income in USD:",
        lambda value: float(value) if float(value) > 0 else None,
    ),
    (
        "person_emp_exp",
        "Enter employment experience in years:",
        lambda value: int(value) if int(value) >= 0 else None,
    ),
    (
        "person_home_ownership",
        "Enter home ownership: RENT, MORTGAGE, OWN, or OTHER",
        lambda value: value.upper()
        if value.upper() in {"RENT", "MORTGAGE", "OWN", "OTHER"}
        else None,
    ),
    (
        "loan_amnt",
        "Enter requested loan amount in USD:",
        lambda value: float(value) if float(value) > 0 else None,
    ),
    (
        "loan_intent",
        "Enter loan purpose: PERSONAL, EDUCATION, MEDICAL, VENTURE, "
        "HOMEIMPROVEMENT, or DEBTCONSOLIDATION",
        lambda value: value.upper()
        if value.upper() in {
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION",
        }
        else None,
    ),
    (
        "loan_int_rate",
        "Enter interest rate in percent, for example 11.5:",
        lambda value: float(value) if float(value) > 0 else None,
    ),
    (
        "cb_person_cred_hist_length",
        "Enter credit history length in years:",
        lambda value: float(value) if float(value) >= 0 else None,
    ),
    (
        "credit_score",
        "Enter credit score (300-850):",
        lambda value: int(value)
        if 300 <= int(value) <= 850
        else None,
    ),
    (
        "previous_loan_defaults_on_file",
        "Were there previous loan defaults? Enter Yes or No:",
        lambda value: value.title()
        if value.title() in {"Yes", "No"}
        else None,
    ),
    (
        "loan_status",
        "Enter loan status: 0 for Rejected or 1 for Approved:",
        lambda value: int(value) if value in {"0", "1"} else None,
    ),
]


async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.message.reply_text(
        "LOAN APPROVAL CLASSIFICATION ANALYSIS\n\n"
        "Explore the project, view dataset records, add a new application "
        "through the API, or display charts directly in Telegram.",
        reply_markup=main_menu(),
    )


def get_application_filters(context):
    if "application_filters" not in context.user_data:
        context.user_data["application_filters"] = (
            default_application_filters()
        )
    return context.user_data["application_filters"]


def filters_summary(selected_filters):
    status_labels = {
        "all": "All",
        "0": "Rejected",
        "1": "Approved",
    }
    return (
        f"Status: {status_labels[selected_filters['status']]}\n"
        f"Loan intent: {selected_filters['intent']}\n"
        f"Rows per page: {selected_filters['limit']}\n"
        f"Rows to skip: {selected_filters['offset']}"
    )


async def show_applications(query, selected_filters):
    params = {
        "limit": selected_filters["limit"],
        "offset": selected_filters["offset"],
    }
    if selected_filters["status"] != "all":
        params["loan_status"] = int(selected_filters["status"])
    if selected_filters["intent"] != "All":
        params["loan_intent"] = selected_filters["intent"]

    try:
        response = await asyncio.to_thread(
            requests.get,
            f"{API_URL}/loans",
            params=params,
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        total = result["total_matching"]
        text = format_applications(
            result["applications"],
            total,
            selected_filters["offset"],
        )
        text = f"{filters_summary(selected_filters)}\n\n{text}"
        keyboard = pagination_keyboard(
            selected_filters["offset"],
            total,
            selected_filters["limit"],
        )
    except (requests.RequestException, KeyError, ValueError) as error:
        text = (
            "The API is temporarily unavailable. "
            f"Please try again later.\n\nDetails: {error}"
        )
        keyboard = back_button()

    await query.edit_message_text(text, reply_markup=keyboard)


async def show_chart(query, chart_name):
    image, caption = await asyncio.to_thread(
        create_chart,
        chart_name,
    )
    await query.message.reply_photo(
        photo=image,
        caption=caption,
        reply_markup=back_button(),
    )


async def answer_callback_safely(query):
    try:
        await query.answer()
    except BadRequest as error:
        if "Query is too old" not in str(error):
            raise


async def check_api(query):
    await query.edit_message_text(
        "CHECKING API\n\n"
        "The Render service may need up to a minute to wake up."
    )

    try:
        response = await asyncio.to_thread(
            requests.get,
            API_URL,
            timeout=60,
        )
        response.raise_for_status()
        api_data = response.json()
        text = (
            "API STATUS\n\n"
            "Status: online\n"
            f"Message: {api_data['message']}\n"
            f"Applications available: "
            f"{api_data['number_of_applications']:,}"
        )
    except (requests.RequestException, KeyError, ValueError) as error:
        text = (
            "API STATUS\n\n"
            "Status: unavailable\n"
            f"Details: {error}"
        )

    await query.edit_message_text(text, reply_markup=back_button())


async def start_add_application(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    query = update.callback_query
    if query:
        await answer_callback_safely(query)

    context.user_data["new_application"] = {}
    context.user_data["form_step"] = 0

    text = (
        "ADD A NEW LOAN APPLICATION\n\n"
        "The bot will ask for the fields one by one and then send the "
        "completed record to the FastAPI POST endpoint.\n\n"
        "Use /cancel at any time to stop.\n\n"
        f"{FORM_FIELDS[0][1]}"
    )

    if query:
        await query.edit_message_text(text)
    else:
        await update.message.reply_text(text)

    return FORM_INPUT


async def start_offset_input(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    query = update.callback_query
    await answer_callback_safely(query)
    await query.edit_message_text(
        "ROWS TO SKIP\n\n"
        "Enter a non-negative whole number, for example 0, 10, or 50.\n\n"
        "Use /cancel to stop."
    )
    return OFFSET_INPUT


async def process_offset_input(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    try:
        offset = int(update.message.text.strip())
        if offset < 0:
            raise ValueError
    except ValueError:
        await update.message.reply_text(
            "Please enter a non-negative whole number."
        )
        return OFFSET_INPUT

    selected_filters = get_application_filters(context)
    selected_filters["offset"] = offset

    await update.message.reply_text(
        "VIEW APPLICATIONS\n\n"
        f"{filters_summary(selected_filters)}",
        reply_markup=applications_menu(selected_filters),
    )
    return ConversationHandler.END


async def process_form_input(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    step = context.user_data["form_step"]
    field_name, prompt, parser = FORM_FIELDS[step]

    try:
        parsed_value = parser(update.message.text.strip())
    except (TypeError, ValueError):
        parsed_value = None

    if parsed_value is None:
        await update.message.reply_text(
            f"Invalid value. Please try again.\n\n{prompt}"
        )
        return FORM_INPUT

    context.user_data["new_application"][field_name] = parsed_value
    step += 1
    context.user_data["form_step"] = step

    if step < len(FORM_FIELDS):
        await update.message.reply_text(FORM_FIELDS[step][1])
        return FORM_INPUT

    application_data = context.user_data["new_application"]
    application_data["loan_percent_income"] = (
        application_data["loan_amnt"]
        / application_data["person_income"]
    )

    if application_data["person_emp_exp"] > application_data["person_age"]:
        await update.message.reply_text(
            "Employment experience cannot be greater than age. "
            "The form was cancelled; please start again.",
            reply_markup=back_button(),
        )
        context.user_data.clear()
        return ConversationHandler.END

    try:
        response = await asyncio.to_thread(
            requests.post,
            f"{API_URL}/loans",
            json=application_data,
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        status = (
            "Approved"
            if application_data["loan_status"] == 1
            else "Rejected"
        )
        await update.message.reply_text(
            "APPLICATION CREATED\n\n"
            f"Application ID: {result['application_id']}\n"
            f"Status: {status}\n"
            f"Loan burden: "
            f"{application_data['loan_percent_income'] * 100:.1f}%\n\n"
            "The record was sent through the FastAPI POST method.",
            reply_markup=back_button(),
        )
    except (requests.RequestException, KeyError, ValueError) as error:
        await update.message.reply_text(
            "The application could not be created.\n\n"
            f"Details: {error}",
            reply_markup=back_button(),
        )

    context.user_data.clear()
    return ConversationHandler.END


async def cancel_form(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    context.user_data.clear()
    await update.message.reply_text(
        "Application form cancelled.",
        reply_markup=main_menu(),
    )
    return ConversationHandler.END


async def return_to_main_menu(query):
    text = (
        "LOAN APPROVAL CLASSIFICATION ANALYSIS\n\n"
        "Select a project section or interactive feature:"
    )

    if query.message.text is not None:
        await query.edit_message_text(
            text,
            reply_markup=main_menu(),
        )
        return

    await query.message.reply_text(
        text,
        reply_markup=main_menu(),
    )
    await query.message.delete()


async def handle_menu(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    query = update.callback_query
    await answer_callback_safely(query)

    if query.data == "menu":
        await return_to_main_menu(query)
    elif query.data == "applications":
        selected_filters = get_application_filters(context)
        await query.edit_message_text(
            "VIEW APPLICATIONS\n\n"
            "Set the same parameters used on the Streamlit API page.\n\n"
            f"{filters_summary(selected_filters)}",
            reply_markup=applications_menu(selected_filters),
        )
    elif query.data == "filter_status":
        await query.edit_message_text(
            "SELECT LOAN STATUS",
            reply_markup=status_filter_menu(),
        )
    elif query.data.startswith("status_"):
        selected_filters = get_application_filters(context)
        selected_filters["status"] = query.data.removeprefix(
            "status_"
        )
        selected_filters["offset"] = 0
        await query.edit_message_text(
            "VIEW APPLICATIONS\n\n"
            f"{filters_summary(selected_filters)}",
            reply_markup=applications_menu(selected_filters),
        )
    elif query.data == "filter_intent":
        await query.edit_message_text(
            "SELECT LOAN INTENT",
            reply_markup=intent_filter_menu(),
        )
    elif query.data.startswith("intent_"):
        selected_filters = get_application_filters(context)
        selected_filters["intent"] = query.data.removeprefix(
            "intent_"
        )
        selected_filters["offset"] = 0
        await query.edit_message_text(
            "VIEW APPLICATIONS\n\n"
            f"{filters_summary(selected_filters)}",
            reply_markup=applications_menu(selected_filters),
        )
    elif query.data == "filter_limit":
        await query.edit_message_text(
            "SELECT NUMBER OF APPLICATIONS",
            reply_markup=limit_filter_menu(),
        )
    elif query.data.startswith("limit_"):
        selected_filters = get_application_filters(context)
        selected_filters["limit"] = int(
            query.data.removeprefix("limit_")
        )
        selected_filters["offset"] = 0
        await query.edit_message_text(
            "VIEW APPLICATIONS\n\n"
            f"{filters_summary(selected_filters)}",
            reply_markup=applications_menu(selected_filters),
        )
    elif query.data == "filters_reset":
        context.user_data["application_filters"] = (
            default_application_filters()
        )
        selected_filters = get_application_filters(context)
        await query.edit_message_text(
            "FILTERS RESET\n\n"
            f"{filters_summary(selected_filters)}",
            reply_markup=applications_menu(selected_filters),
        )
    elif query.data == "apps_show":
        selected_filters = get_application_filters(context)
        selected_filters["offset"] = 0
        await show_applications(query, selected_filters)
    elif query.data.startswith("apps_page_"):
        selected_filters = get_application_filters(context)
        selected_filters["offset"] = int(
            query.data.removeprefix("apps_page_")
        )
        await show_applications(query, selected_filters)
    elif query.data == "charts":
        await query.edit_message_text(
            "PROJECT CHARTS\n\n"
            "All project visualizations are grouped by section:",
            reply_markup=charts_menu(),
        )
    elif query.data == "charts_overview":
        await query.edit_message_text(
            "GENERAL OVERVIEW CHARTS",
            reply_markup=overview_charts_menu(),
        )
    elif query.data == "charts_detailed":
        await query.edit_message_text(
            "DETAILED ANALYSIS CHARTS",
            reply_markup=detailed_charts_menu(),
        )
    elif query.data == "charts_hypotheses":
        await query.edit_message_text(
            "HYPOTHESIS GRAPHS",
            reply_markup=hypothesis_charts_menu(),
        )
    elif query.data.startswith("chart_"):
        await show_chart(query, query.data.removeprefix("chart_"))
    elif query.data == "api_status":
        await check_api(query)
    elif query.data == "website_missing":
        await query.edit_message_text(
            "The Streamlit URL has not been configured yet.",
            reply_markup=back_button(),
        )
    else:
        await query.edit_message_text(
            project_pages()[query.data],
            reply_markup=back_button(),
        )


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.message.reply_text(
        "Available commands:\n\n"
        "/start - open the project menu\n"
        "/add - create a new loan application\n"
        "/cancel - cancel the application form\n"
        "/help - show this message"
    )


def main():
    if not TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN is missing in .env")

    application = Application.builder().token(TOKEN).build()

    add_conversation = ConversationHandler(
        entry_points=[
            CallbackQueryHandler(
                start_add_application,
                pattern="^add_application$",
            ),
            CallbackQueryHandler(
                start_offset_input,
                pattern="^filter_offset$",
            ),
            CommandHandler("add", start_add_application),
        ],
        states={
            FORM_INPUT: [
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND,
                    process_form_input,
                )
            ],
            OFFSET_INPUT: [
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND,
                    process_offset_input,
                )
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel_form)],
        allow_reentry=True,
    )

    application.add_handler(add_conversation)
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(handle_menu))

    print("Telegram bot is running...")
    application.run_polling()


if __name__ == "__main__":
    main()
