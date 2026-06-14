from pathlib import Path
import pandas as pd
from fastapi import FastAPI, Query
import json
from pydantic import BaseModel, Field


app = FastAPI(
    title="Loan Approval API",
    description=(
        "REST API for accessing and adding loan applications."
    ),
    version="1.0.0"
)


DATA_PATH = Path(__file__).parent.parent / "loan_data.csv"


df = pd.read_csv(DATA_PATH)

class LoanApplication(BaseModel):
    person_age: float = Field(gt=0, le=100)
    person_gender: str
    person_education: str
    person_income: float = Field(gt=0)
    person_emp_exp: int = Field(ge=0)
    person_home_ownership: str
    loan_amnt: float = Field(gt=0)
    loan_intent: str
    loan_int_rate: float = Field(gt=0)
    loan_percent_income: float = Field(ge=0, le=1)
    cb_person_cred_hist_length: float = Field(ge=0)
    credit_score: int = Field(ge=300, le=850)
    previous_loan_defaults_on_file: str
    loan_status: int = Field(ge=0, le=1)


@app.get("/")
def root():
    return {
        "message": "Loan Approval API is running",
        "number_of_applications": len(df)
    }

@app.get("/loans")
def get_loans(
    limit: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of returned applications"
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Number of applications to skip"
    ),
    loan_status: int | None = Query(
        default=None,
        ge=0,
        le=1,
        description="0 for rejected, 1 for approved"
    ),
    loan_intent: str | None = Query(
        default=None,
        description="Filter by loan purpose"
    )
):
    filtered_data = df.copy()

    if loan_status is not None:
        filtered_data = filtered_data[
            filtered_data["loan_status"] == loan_status
        ]

    if loan_intent is not None:
        filtered_data = filtered_data[
            filtered_data["loan_intent"].str.upper()
            == loan_intent.upper()
        ]

    total_matching = len(filtered_data)

    result = filtered_data.iloc[
        offset:offset + limit
    ]

    records = json.loads(
        result.to_json(orient="records")
    )

    return {
        "total_matching": total_matching,
        "limit": limit,
        "offset": offset,
        "applications": records
    }

@app.post("/loans", status_code=201)
def create_loan(application: LoanApplication):
    new_application = application.model_dump()

    new_index = len(df)

    df.loc[new_index] = new_application

    return {
        "message": "Loan application created successfully",
        "application_id": new_index,
        "application": new_application
    }
