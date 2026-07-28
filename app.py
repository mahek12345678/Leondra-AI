import json
import traceback
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


# -------------------------------------------------------------------
# Application configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Lendora AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parent / "loan_pipeline.pkl"

MODEL_FEATURES = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
    "total_assets",
    "loan_income_ratio",
    "asset_coverage_ratio",
    "monthly_income",
    "emi_proxy",
]

RAW_FEATURES = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

ENGINEERED_FEATURES = [
    "total_assets",
    "loan_income_ratio",
    "asset_coverage_ratio",
    "monthly_income",
    "emi_proxy",
]


# -------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------

if "assessment" not in st.session_state:
    st.session_state.assessment = None

if "prediction_error" not in st.session_state:
    st.session_state.prediction_error = None


# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model(model_path: Path) -> Any:
    return joblib.load(model_path)


# -------------------------------------------------------------------
# Feature engineering
# -------------------------------------------------------------------

def build_model_input(
    no_of_dependents: int,
    education: str,
    self_employed: str,
    income_annum: float,
    loan_amount: float,
    loan_term: int,
    cibil_score: int,
    residential_assets_value: float,
    commercial_assets_value: float,
    luxury_assets_value: float,
    bank_asset_value: float,
) -> pd.DataFrame:
    total_assets = (
        residential_assets_value
        + commercial_assets_value
        + luxury_assets_value
        + bank_asset_value
    )

    monthly_income = income_annum / 12
    loan_income_ratio = loan_amount / income_annum
    asset_coverage_ratio = total_assets / loan_amount
    emi_proxy = loan_amount / (loan_term * 12)

    input_data = {
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
        "total_assets": total_assets,
        "loan_income_ratio": loan_income_ratio,
        "asset_coverage_ratio": asset_coverage_ratio,
        "monthly_income": monthly_income,
        "emi_proxy": emi_proxy,
    }

    return pd.DataFrame([input_data], columns=MODEL_FEATURES)


# -------------------------------------------------------------------
# Prediction utilities
# -------------------------------------------------------------------

def normalise_class_label(class_label: Any) -> str:
    return str(class_label).strip().lower().replace("_", " ")


def identify_approval_class(classes: list[Any]) -> Any:
    approval_labels = {
        "1",
        "approved",
        "approve",
        "yes",
        "true",
        "eligible",
        "accepted",
        "accept",
        "loan approved",
    }

    for class_label in classes:
        normalised_label = normalise_class_label(class_label)

        if normalised_label in approval_labels:
            return class_label

    numeric_classes = []

    for class_label in classes:
        try:
            numeric_classes.append((float(class_label), class_label))
        except (TypeError, ValueError):
            continue

    if numeric_classes:
        return max(numeric_classes, key=lambda item: item[0])[1]

    return classes[-1]


def is_approval_label(class_label: Any) -> bool:
    approval_labels = {
        "1",
        "approved",
        "approve",
        "yes",
        "true",
        "eligible",
        "accepted",
        "accept",
        "loan approved",
    }

    return normalise_class_label(class_label) in approval_labels


def get_prediction_result(
    model: Any,
    model_input: pd.DataFrame,
) -> dict[str, Any]:
    predicted_class = model.predict(model_input)[0]

    approval_probability = 0.0
    confidence = 1.0
    approval_class = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_input)[0]
        classes = list(model.classes_)

        approval_class = identify_approval_class(classes)
        approval_index = classes.index(approval_class)
        predicted_index = classes.index(predicted_class)

        approval_probability = float(probabilities[approval_index])
        confidence = float(probabilities[predicted_index])
        approved = predicted_class == approval_class
    else:
        approved = is_approval_label(predicted_class)
        approval_probability = 1.0 if approved else 0.0

    decision = "Approved" if approved else "Review Required"

    return {
        "decision": decision,
        "is_approved": approved,
        "confidence": confidence,
        "approval_probability": approval_probability,
        "predicted_class": predicted_class,
        "approval_class": approval_class,
    }


# -------------------------------------------------------------------
# Risk grading
# -------------------------------------------------------------------

def calculate_risk_grade(
    approval_probability: float,
    cibil_score: int,
    loan_income_ratio: float,
    asset_coverage_ratio: float,
) -> str:
    risk_score = 0

    if approval_probability >= 0.85:
        risk_score += 4
    elif approval_probability >= 0.70:
        risk_score += 3
    elif approval_probability >= 0.55:
        risk_score += 2
    elif approval_probability >= 0.40:
        risk_score += 1

    if cibil_score >= 750:
        risk_score += 3
    elif cibil_score >= 700:
        risk_score += 2
    elif cibil_score >= 650:
        risk_score += 1

    if loan_income_ratio <= 2.5:
        risk_score += 2
    elif loan_income_ratio <= 4:
        risk_score += 1

    if asset_coverage_ratio >= 1.5:
        risk_score += 2
    elif asset_coverage_ratio >= 1:
        risk_score += 1

    if risk_score >= 10:
        return "A"

    if risk_score >= 8:
        return "B"

    if risk_score >= 6:
        return "C"

    if risk_score >= 4:
        return "D"

    return "E"


# -------------------------------------------------------------------
# Rule-based AI insights
# -------------------------------------------------------------------

def generate_insights(
    model_input: pd.DataFrame,
) -> list[tuple[str, str]]:
    row = model_input.iloc[0]
    insights: list[tuple[str, str]] = []

    cibil_score = int(row["cibil_score"])
    loan_income_ratio = float(row["loan_income_ratio"])
    asset_coverage_ratio = float(row["asset_coverage_ratio"])
    loan_term = int(row["loan_term"])
    total_assets = float(row["total_assets"])
    loan_amount = float(row["loan_amount"])
    monthly_income = float(row["monthly_income"])
    emi_proxy = float(row["emi_proxy"])

    if cibil_score >= 750:
        insights.append(
            (
                "success",
                "Strong credit profile: the applicant's CIBIL score indicates excellent creditworthiness.",
            )
        )
    elif cibil_score >= 700:
        insights.append(
            (
                "success",
                "Good credit profile: the CIBIL score supports a favourable underwriting assessment.",
            )
        )
    elif cibil_score >= 650:
        insights.append(
            (
                "warning",
                "Moderate credit profile: review the credit score alongside leverage and collateral.",
            )
        )
    else:
        insights.append(
            (
                "error",
                "Credit risk concern: the CIBIL score is below the typical lower-risk lending range.",
            )
        )

    if loan_income_ratio <= 2.5:
        insights.append(
            (
                "success",
                "Healthy borrowing level: the requested loan is proportionate to annual income.",
            )
        )
    elif loan_income_ratio <= 4:
        insights.append(
            (
                "warning",
                "Elevated leverage: the requested loan is relatively large compared with annual income.",
            )
        )
    else:
        insights.append(
            (
                "error",
                "High leverage: the requested loan is substantially larger than annual income.",
            )
        )

    if asset_coverage_ratio >= 1.5:
        insights.append(
            (
                "success",
                "Good collateral position: declared assets provide strong coverage of the requested loan.",
            )
        )
    elif asset_coverage_ratio >= 1:
        insights.append(
            (
                "warning",
                "Moderate asset cover: declared assets cover the loan with a limited buffer.",
            )
        )
    else:
        insights.append(
            (
                "error",
                "Limited asset cover: total declared assets are below the requested loan amount.",
            )
        )

    if loan_term >= 15:
        insights.append(
            (
                "warning",
                "Long repayment term: the extended tenure may reduce instalments but increases long-term exposure.",
            )
        )
    elif loan_term <= 5:
        insights.append(
            (
                "success",
                "Short repayment term: the proposed tenure limits long-term credit exposure.",
            )
        )
    else:
        insights.append(
            (
                "success",
                "Balanced repayment term: the proposed tenure is within a moderate underwriting range.",
            )
        )

    if monthly_income > 0:
        estimated_monthly_burden = emi_proxy / monthly_income
    else:
        estimated_monthly_burden = float("inf")

    if estimated_monthly_burden <= 0.30:
        insights.append(
            (
                "success",
                "Manageable repayment proxy: the estimated principal-only monthly burden is low relative to income.",
            )
        )
    elif estimated_monthly_burden <= 0.50:
        insights.append(
            (
                "warning",
                "Moderate repayment burden: affordability should be checked against existing obligations.",
            )
        )
    else:
        insights.append(
            (
                "error",
                "High repayment burden: the estimated monthly loan burden is large relative to income.",
            )
        )

    if total_assets >= loan_amount * 2:
        insights.append(
            (
                "success",
                "Strong balance sheet: total declared assets are at least twice the requested loan amount.",
            )
        )

    return insights


# -------------------------------------------------------------------
# Report generation
# -------------------------------------------------------------------

def convert_to_serialisable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()

    return value


def create_report(
    model_input: pd.DataFrame,
    prediction_result: dict[str, Any],
    risk_grade: str,
) -> str:
    row = model_input.iloc[0]

    report = {
        "assessment": {
            "decision": prediction_result["decision"],
            "confidence": round(
                float(prediction_result["confidence"]),
                6,
            ),
            "confidence_percent": round(
                float(prediction_result["confidence"]) * 100,
                2,
            ),
            "approval_probability": round(
                float(prediction_result["approval_probability"]),
                6,
            ),
            "approval_probability_percent": round(
                float(prediction_result["approval_probability"]) * 100,
                2,
            ),
            "risk_grade": risk_grade,
            "predicted_class": str(
                prediction_result["predicted_class"]
            ),
        },
        "input_features": {
            feature: convert_to_serialisable(row[feature])
            for feature in RAW_FEATURES
        },
        "engineered_features": {
            feature: convert_to_serialisable(row[feature])
            for feature in ENGINEERED_FEATURES
        },
    }

    return json.dumps(report, indent=2)


# -------------------------------------------------------------------
# Formatting
# -------------------------------------------------------------------

def format_currency(value: float) -> str:
    return f"₹{value:,.0f}"


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

st.title("Lendora AI")
st.subheader("AI-Powered Loan Approval & Risk Assessment Platform")
st.caption(
    "An intelligent underwriting workspace for evaluating applicant "
    "affordability, credit strength, leverage and collateral coverage."
)

metric_column_1, metric_column_2, metric_column_3, metric_column_4 = (
    st.columns(4)
)

with metric_column_1:
    st.metric("Model", "Gradient Boosting")

with metric_column_2:
    st.metric("Features", "16")

with metric_column_3:
    st.metric("Inference", "Real Time")

with metric_column_4:
    st.metric("Pipeline", "Scikit-Learn")


# -------------------------------------------------------------------
# Load model safely
# -------------------------------------------------------------------

model = None
model_error = None

with st.status(
    "Initialising underwriting model",
    expanded=False,
) as model_status:
    try:
        model = load_model(MODEL_PATH)

        model_status.update(
            label="Underwriting model ready",
            state="complete",
            expanded=False,
        )
    except Exception:
        model_error = traceback.format_exc()

        model_status.update(
            label="Underwriting model unavailable",
            state="error",
            expanded=True,
        )

if model_error:
    st.error(
        "The model could not be loaded. Ensure that loan_pipeline.pkl "
        "is in the same directory as app.py and that all required "
        "Python packages are installed."
    )

    with st.expander(
        "Model loading traceback",
        expanded=False,
    ):
        st.code(
            model_error,
            language="text",
        )


# -------------------------------------------------------------------
# Main application layout
# -------------------------------------------------------------------

left_column, right_column = st.columns(
    [1.05, 0.95],
    gap="large",
)


# -------------------------------------------------------------------
# Left column: applicant form
# -------------------------------------------------------------------

with left_column:
    with st.container(border=True):
        st.subheader("Applicant Information")
        st.caption(
            "Enter the applicant's profile, requested facility and assets."
        )

        profile_tab, loan_tab, assets_tab = st.tabs(
            [
                "Profile",
                "Loan",
                "Assets",
            ]
        )

        with profile_tab:
            profile_left, profile_right = st.columns(2)

            with profile_left:
                no_of_dependents = st.number_input(
                    "Dependents",
                    min_value=0,
                    max_value=20,
                    value=0,
                    step=1,
                )

                education = st.selectbox(
                    "Education",
                    options=[
                        "Graduate",
                        "Not Graduate",
                    ],
                    index=0,
                )

            with profile_right:
                self_employed = st.selectbox(
                    "Employment",
                    options=[
                        "No",
                        "Yes",
                    ],
                    index=0,
                    help=(
                        "Select Yes when the applicant is "
                        "self-employed."
                    ),
                )

                income_annum = st.number_input(
                    "Annual Income",
                    min_value=1.0,
                    value=1_200_000.0,
                    step=50_000.0,
                    format="%.2f",
                )

        with loan_tab:
            loan_left, loan_right = st.columns(2)

            with loan_left:
                loan_amount = st.number_input(
                    "Loan Amount",
                    min_value=1.0,
                    value=2_500_000.0,
                    step=50_000.0,
                    format="%.2f",
                )

                loan_term = st.number_input(
                    "Loan Term",
                    min_value=1,
                    max_value=40,
                    value=10,
                    step=1,
                    help="Loan term in years.",
                )

            with loan_right:
                cibil_score = st.number_input(
                    "CIBIL Score",
                    min_value=300,
                    max_value=900,
                    value=750,
                    step=1,
                )

                estimated_monthly_principal = (
                    float(loan_amount)
                    / (int(loan_term) * 12)
                )

                st.metric(
                    "Monthly Principal Proxy",
                    format_currency(
                        estimated_monthly_principal
                    ),
                )

        with assets_tab:
            assets_left, assets_right = st.columns(2)

            with assets_left:
                residential_assets_value = st.number_input(
                    "Residential Assets",
                    min_value=0.0,
                    value=2_000_000.0,
                    step=50_000.0,
                    format="%.2f",
                )

                commercial_assets_value = st.number_input(
                    "Commercial Assets",
                    min_value=0.0,
                    value=500_000.0,
                    step=50_000.0,
                    format="%.2f",
                )

            with assets_right:
                luxury_assets_value = st.number_input(
                    "Luxury Assets",
                    min_value=0.0,
                    value=300_000.0,
                    step=50_000.0,
                    format="%.2f",
                )

                bank_asset_value = st.number_input(
                    "Bank Assets",
                    min_value=0.0,
                    value=400_000.0,
                    step=50_000.0,
                    format="%.2f",
                )

        generate_assessment = st.button(
            "Generate Risk Assessment",
            type="primary",
            use_container_width=True,
            disabled=model is None,
        )

        if generate_assessment:
            st.session_state.prediction_error = None

            try:
                with st.status(
                    "Running underwriting assessment",
                    expanded=True,
                ) as assessment_status:
                    assessment_status.write(
                        "Validating applicant data"
                    )

                    model_input = build_model_input(
                        no_of_dependents=int(
                            no_of_dependents
                        ),
                        education=education,
                        self_employed=self_employed,
                        income_annum=float(
                            income_annum
                        ),
                        loan_amount=float(
                            loan_amount
                        ),
                        loan_term=int(
                            loan_term
                        ),
                        cibil_score=int(
                            cibil_score
                        ),
                        residential_assets_value=float(
                            residential_assets_value
                        ),
                        commercial_assets_value=float(
                            commercial_assets_value
                        ),
                        luxury_assets_value=float(
                            luxury_assets_value
                        ),
                        bank_asset_value=float(
                            bank_asset_value
                        ),
                    )

                    assessment_status.write(
                        "Computing affordability and "
                        "collateral features"
                    )

                    prediction_result = (
                        get_prediction_result(
                            model=model,
                            model_input=model_input,
                        )
                    )

                    assessment_status.write(
                        "Assigning risk grade and "
                        "generating insights"
                    )

                    risk_grade = calculate_risk_grade(
                        approval_probability=float(
                            prediction_result[
                                "approval_probability"
                            ]
                        ),
                        cibil_score=int(
                            model_input.iloc[0][
                                "cibil_score"
                            ]
                        ),
                        loan_income_ratio=float(
                            model_input.iloc[0][
                                "loan_income_ratio"
                            ]
                        ),
                        asset_coverage_ratio=float(
                            model_input.iloc[0][
                                "asset_coverage_ratio"
                            ]
                        ),
                    )

                    insights = generate_insights(
                        model_input
                    )

                    report = create_report(
                        model_input=model_input,
                        prediction_result=prediction_result,
                        risk_grade=risk_grade,
                    )

                    st.session_state.assessment = {
                        "model_input": model_input,
                        "prediction_result": prediction_result,
                        "risk_grade": risk_grade,
                        "insights": insights,
                        "report": report,
                    }

                    assessment_status.update(
                        label="Risk assessment complete",
                        state="complete",
                        expanded=False,
                    )

            except Exception:
                st.session_state.assessment = None
                st.session_state.prediction_error = (
                    traceback.format_exc()
                )


# -------------------------------------------------------------------
# Right column: assessment result
# -------------------------------------------------------------------

with right_column:
    with st.container(border=True):
        st.subheader("Assessment Result")

        if st.session_state.prediction_error:
            st.error(
                "The prediction could not be completed. "
                "Review the technical details below."
            )

            with st.expander(
                "Prediction traceback",
                expanded=False,
            ):
                st.code(
                    st.session_state.prediction_error,
                    language="text",
                )

        elif st.session_state.assessment is None:
            st.info(
                "Complete the applicant information and "
                "generate a risk assessment to view the "
                "underwriting decision."
            )

            placeholder_left, placeholder_right = (
                st.columns(2)
            )

            with placeholder_left:
                st.metric(
                    "Decision",
                    "Pending",
                )

                st.metric(
                    "Confidence",
                    "—",
                )

            with placeholder_right:
                st.metric(
                    "Risk Grade",
                    "—",
                )

                st.metric(
                    "Approval Probability",
                    "—",
                )

            st.progress(
                0,
                text="Awaiting assessment",
            )

        else:
            assessment = st.session_state.assessment
            prediction_result = assessment[
                "prediction_result"
            ]
            model_input = assessment["model_input"]
            row = model_input.iloc[0]

            if prediction_result["is_approved"]:
                st.success("Approved")
            else:
                st.warning("Review Required")

            result_column_1, result_column_2, result_column_3 = (
                st.columns(3)
            )

            with result_column_1:
                st.metric(
                    "Decision",
                    prediction_result["decision"],
                )

            with result_column_2:
                st.metric(
                    "Confidence",
                    (
                        f"{prediction_result['confidence']:.1%}"
                    ),
                )

            with result_column_3:
                st.metric(
                    "Risk Grade",
                    assessment["risk_grade"],
                )

            ratio_column_1, ratio_column_2 = (
                st.columns(2)
            )

            with ratio_column_1:
                st.metric(
                    "Loan Income Ratio",
                    (
                        f"{float(row['loan_income_ratio']):.2f}x"
                    ),
                )

            with ratio_column_2:
                st.metric(
                    "Asset Coverage Ratio",
                    (
                        f"{float(row['asset_coverage_ratio']):.2f}x"
                    ),
                )

            approval_probability = float(
                prediction_result[
                    "approval_probability"
                ]
            )

            progress_value = int(
                max(
                    0,
                    min(
                        100,
                        round(
                            approval_probability * 100
                        ),
                    ),
                )
            )

            st.progress(
                progress_value,
                text=(
                    "Approval probability: "
                    f"{approval_probability:.1%}"
                ),
            )


# -------------------------------------------------------------------
# Detailed assessment tabs
# -------------------------------------------------------------------

if st.session_state.assessment is not None:
    assessment = st.session_state.assessment
    model_input = assessment["model_input"]
    row = model_input.iloc[0]

    overview_tab, insights_tab, model_input_tab = st.tabs(
        [
            "Financial Overview",
            "AI Insights",
            "Model Input",
        ]
    )

    with overview_tab:
        with st.container(border=True):
            st.subheader("Financial Overview")

            (
                overview_column_1,
                overview_column_2,
                overview_column_3,
                overview_column_4,
            ) = st.columns(4)

            with overview_column_1:
                st.metric(
                    "Annual Income",
                    format_currency(
                        float(row["income_annum"])
                    ),
                )

            with overview_column_2:
                st.metric(
                    "Loan Amount",
                    format_currency(
                        float(row["loan_amount"])
                    ),
                )

            with overview_column_3:
                st.metric(
                    "Total Assets",
                    format_currency(
                        float(row["total_assets"])
                    ),
                )

            with overview_column_4:
                st.metric(
                    "Loan Term",
                    f"{int(row['loan_term'])} years",
                )

            st.subheader("Asset Composition")

            asset_chart_data = pd.DataFrame(
                {
                    "Asset Value": [
                        float(
                            row[
                                "residential_assets_value"
                            ]
                        ),
                        float(
                            row[
                                "commercial_assets_value"
                            ]
                        ),
                        float(
                            row[
                                "luxury_assets_value"
                            ]
                        ),
                        float(
                            row[
                                "bank_asset_value"
                            ]
                        ),
                    ]
                },
                index=[
                    "Residential",
                    "Commercial",
                    "Luxury",
                    "Bank",
                ],
            )

            st.bar_chart(
                asset_chart_data,
                y="Asset Value",
                use_container_width=True,
            )

    with insights_tab:
        with st.container(border=True):
            st.subheader("AI Insights")
            st.caption(
                "Rule-based underwriting observations "
                "derived from the applicant's financial "
                "and credit profile."
            )

            for (
                insight_type,
                insight_text,
            ) in assessment["insights"]:
                if insight_type == "success":
                    st.success(insight_text)
                elif insight_type == "warning":
                    st.warning(insight_text)
                else:
                    st.error(insight_text)

    with model_input_tab:
        with st.container(border=True):
            st.subheader("Model Input")
            st.caption(
                "Complete feature vector supplied to the "
                "Scikit-learn pipeline."
            )

            display_input = model_input.copy()

            numeric_columns = (
                display_input.select_dtypes(
                    include=["number"]
                ).columns
            )

            display_input[numeric_columns] = (
                display_input[
                    numeric_columns
                ].round(4)
            )

            st.dataframe(
                display_input,
                use_container_width=True,
                hide_index=True,
            )

            with st.expander(
                "Engineered Features",
                expanded=True,
            ):
                engineered_data = pd.DataFrame(
                    {
                        "Feature": [
                            "Total Assets",
                            "Monthly Income",
                            "Loan Income Ratio",
                            "Asset Coverage Ratio",
                            "EMI Proxy",
                        ],
                        "Value": [
                            format_currency(
                                float(
                                    row[
                                        "total_assets"
                                    ]
                                )
                            ),
                            format_currency(
                                float(
                                    row[
                                        "monthly_income"
                                    ]
                                )
                            ),
                            (
                                f"{float(row['loan_income_ratio']):.4f}"
                            ),
                            (
                                f"{float(row['asset_coverage_ratio']):.4f}"
                            ),
                            format_currency(
                                float(
                                    row["emi_proxy"]
                                )
                            ),
                        ],
                    }
                )

                st.dataframe(
                    engineered_data,
                    use_container_width=True,
                    hide_index=True,
                )

            st.download_button(
                label="Download Assessment Report",
                data=assessment["report"],
                file_name=(
                    "lendora_ai_assessment_report.json"
                ),
                mime="application/json",
                type="primary",
                use_container_width=True,
            )