import os
import sys
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

# Ensure current directory and site-packages are in sys.path for unpickling custom transformers
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SITE_PKG = r"D:\Lib\site-packages"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if SITE_PKG not in sys.path:
    sys.path.insert(0, SITE_PKG)

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict, model_validator, AliasChoices

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("credit_risk_api")

# Constants
MODEL_VERSION = "1.0.0"
DEFAULT_CHAMPION_MODEL = "Logistic Regression"
MODEL_FEATURES = ['FICO_score', 'income', 'loan_amount', 'dti', 'employment_length']

# Global state container for model artifact and metadata
app_state: Dict[str, Any] = {
    "model": None,
    "metadata": None,
    "bucketer": None,
    "champion_model": DEFAULT_CHAMPION_MODEL,
    "model_path": None
}

def get_risk_tier(pd_val: float) -> str:
    """
    Classifies calibrated Probability of Default (PD) into intuitive credit risk tiers:
      - Low Risk: PD < 0.20 (Prime borrowers)
      - Moderate Risk: 0.20 <= PD < 0.45 (Good / Near-prime borrowers)
      - High Risk: 0.45 <= PD < 0.70 (Higher-risk / Speculative borrowers)
      - Very High Risk: PD >= 0.70 (Subprime / Distressed borrowers)
    """
    if pd_val < 0.20:
        return "Low Risk"
    elif pd_val < 0.45:
        return "Moderate Risk"
    elif pd_val < 0.70:
        return "High Risk"
    else:
        return "Very High Risk"

def get_fico_bucket_label(fico_score: float, bucketer=None) -> str:
    """
    Determines the DP-discovered optimal FICO bucket range label for a given score.
    """
    boundaries = None
    if bucketer is not None and hasattr(bucketer, 'boundaries_'):
        boundaries = bucketer.boundaries_
    elif app_state["metadata"] and "fico_boundaries" in app_state["metadata"]:
        boundaries = app_state["metadata"]["fico_boundaries"]
    else:
        # Fallback to default optimal boundaries discovered by M1 DP bucketer
        boundaries = [-float('inf'), 600.0, 660.0, 695.0, 741.0, float('inf')]

    for i in range(len(boundaries) - 1):
        low, high = boundaries[i], boundaries[i + 1]
        if low <= fico_score < high:
            return f"[{low}, {high})"
    return f"[{boundaries[-2]}, {boundaries[-1]})"

def load_model_artifact(artifact_path: Optional[str] = None, metadata_path: Optional[str] = None) -> bool:
    """
    Locates and deserializes the champion model pipeline and optional metadata.
    Searches current working directory and module directory.
    """
    # 1. Resolve pipeline artifact path
    candidate_paths = []
    if artifact_path:
        candidate_paths.append(artifact_path)
    candidate_paths.extend([
        os.path.join(CURRENT_DIR, "models", "model_artifact.joblib"),
        "models/model_artifact.joblib",
        os.path.join(CURRENT_DIR, "models", "models/model.joblib"),
        "models/model.joblib"
    ])

    resolved_model_path = None
    for p in candidate_paths:
        if p and os.path.exists(p):
            resolved_model_path = os.path.abspath(p)
            break

    if not resolved_model_path:
        logger.warning("No model artifact found in candidate paths. Model will remain unloaded.")
        app_state["model"] = None
        app_state["bucketer"] = None
        return False

    # 2. Resolve metadata artifact path
    candidate_meta_paths = []
    if metadata_path:
        candidate_meta_paths.append(metadata_path)
    candidate_meta_paths.extend([
        os.path.join(CURRENT_DIR, "models", "model_artifact_metadata.joblib"),
        "models/model_artifact_metadata.joblib",
        resolved_model_path.replace(".joblib", "_metadata.joblib")
    ])

    resolved_meta_path = None
    for mp in candidate_meta_paths:
        if mp and os.path.exists(mp):
            resolved_meta_path = os.path.abspath(mp)
            break

    try:
        pipeline = joblib.load(resolved_model_path)
        app_state["model"] = pipeline
        app_state["model_path"] = resolved_model_path
        logger.info(f"Loaded model pipeline from '{resolved_model_path}'.")

        # Extract DP bucketer if present in pipeline
        try:
            preprocessor = pipeline.named_steps.get('preprocessor')
            if preprocessor and hasattr(preprocessor, 'transformers_'):
                fico_pipe = preprocessor.transformers_[0][1]
                app_state["bucketer"] = fico_pipe.named_steps.get('bucketer')
        except Exception as e:
            logger.warning(f"Could not extract FicoDPBucketer from pipeline: {e}")
            app_state["bucketer"] = None

        # Load metadata if available
        if resolved_meta_path and os.path.exists(resolved_meta_path):
            meta = joblib.load(resolved_meta_path)
            app_state["metadata"] = meta
            app_state["champion_model"] = meta.get("champion_model", DEFAULT_CHAMPION_MODEL)
            logger.info(f"Loaded model metadata from '{resolved_meta_path}': champion={app_state['champion_model']}.")
        else:
            app_state["metadata"] = None
            app_state["champion_model"] = DEFAULT_CHAMPION_MODEL

        return True
    except Exception as e:
        logger.error(f"Error loading model artifact from '{resolved_model_path}': {e}", exc_info=True)
        app_state["model"] = None
        app_state["bucketer"] = None
        return False

def unload_model_artifact():
    """Unloads the model artifact from memory (used for testing 503 error handling)."""
    app_state["model"] = None
    app_state["bucketer"] = None
    app_state["metadata"] = None
    app_state["model_path"] = None
    logger.info("Model artifact unloaded from memory.")

# Pydantic Request & Response Schemas
class LoanApplication(BaseModel):
    """
    Loan application schema containing financial features required for credit risk scoring.
    Extra metadata fields (e.g. customer_id, loan_id, timestamp) pass through cleanly.
    """
    FICO_score: float = Field(
        ...,
        ge=300.0,
        le=850.0,
        validation_alias=AliasChoices("FICO_score", "fico_score", "fico"),
        description="Credit bureau FICO score (standard range: 300 to 850)"
    )
    income: float = Field(
        ...,
        gt=0.0,
        description="Annual borrower gross income in USD (must be strictly positive)"
    )
    loan_amount: float = Field(
        ...,
        gt=0.0,
        description="Requested principal loan amount in USD (must be strictly positive)"
    )
    dti: float = Field(
        ...,
        ge=0.0,
        description="Debt-to-income ratio (must be non-negative, e.g. 0.28 for 28%)"
    )
    employment_length: float = Field(
        ...,
        ge=0.0,
        description="Employment duration in years (must be non-negative)"
    )
    customer_id: Optional[str] = Field(
        default=None,
        description="Optional unique customer identifier (passthrough metadata)"
    )
    loan_id: Optional[str] = Field(
        default=None,
        description="Optional loan application identifier (passthrough metadata)"
    )

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "customer_id": "CUST-98214",
                "loan_id": "LN-40291",
                "FICO_score": 720.0,
                "income": 75000.0,
                "loan_amount": 15000.0,
                "dti": 0.22,
                "employment_length": 6.0
            }
        }
    )

class LoanPredictionResponse(BaseModel):
    """
    Credit risk prediction output containing calibrated default probability,
    binary classification, categorical risk tier, and assigned FICO bucket.
    """
    probability_of_default: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calibrated probability of default (PD) bounded in [0.0, 1.0]"
    )
    predicted_default: int = Field(
        ...,
        ge=0,
        le=1,
        description="Binary default decision (0: Non-default, 1: Default, threshold 0.50)"
    )
    predicted_class: int = Field(
        ...,
        ge=0,
        le=1,
        description="Binary classification label (synonym for predicted_default)"
    )
    risk_tier: str = Field(
        ...,
        description="Credit risk category ('Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk')"
    )
    fico_bucket: str = Field(
        ...,
        description="Assigned FICO partition bucket range discovered by dynamic programming"
    )
    customer_id: Optional[str] = Field(
        default=None,
        description="Customer identifier if provided in request"
    )
    loan_id: Optional[str] = Field(
        default=None,
        description="Loan application identifier if provided in request"
    )

    model_config = ConfigDict(extra="allow", populate_by_name=True)

class BatchLoanApplication(BaseModel):
    """
    Batch evaluation request supporting multiple loan applications.
    Accepts list under either 'loans' or 'applications' key.
    """
    loans: Optional[List[LoanApplication]] = Field(
        default=None,
        description="List of loan applications for batch credit risk evaluation"
    )
    applications: Optional[List[LoanApplication]] = Field(
        default=None,
        description="Alternative field name for loan applications list"
    )

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def unify_loans_list(self):
        if self.loans is None and self.applications is None:
            raise ValueError("Either 'loans' or 'applications' list must be provided.")
        if self.loans is None:
            self.loans = self.applications
        return self

class BatchLoanPredictionResponse(BaseModel):
    """
    Batch prediction output containing individual predictions and portfolio metadata.
    """
    predictions: List[LoanPredictionResponse] = Field(
        ...,
        description="List of prediction responses corresponding to submitted applications"
    )
    total_count: int = Field(..., description="Total count of evaluated applications")
    model_version: str = Field(default=MODEL_VERSION, description="Serving model pipeline version")
    champion_model: str = Field(default=DEFAULT_CHAMPION_MODEL, description="Champion model family name")

    model_config = ConfigDict(extra="allow")

class HealthResponse(BaseModel):
    """
    Health check response model reporting system status and serving model metadata.
    """
    status: str = Field(default="healthy", description="API operational health status")
    model_version: str = Field(default=MODEL_VERSION, description="Serving model version")
    champion_model: str = Field(default=DEFAULT_CHAMPION_MODEL, description="Champion model family name")

    model_config = ConfigDict(extra="allow")

# FastAPI Lifespan Context Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure model artifact is loaded at startup
    if app_state["model"] is None:
        load_model_artifact()
    yield
    # Cleanup / shutdown logic if needed
    logger.info("Serving API shutting down.")

# Initialize FastAPI Application
app = FastAPI(
    title="Quantitative Credit Risk Serving API",
    description=(
        "Production-ready REST API for estimating the Probability of Default (PD) "
        "and credit risk tiers using an end-to-end scikit-learn pipeline featuring "
        "Dynamic Programming optimal FICO discretization."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan
)

# Attempt immediate load upon module import for direct TestClient / CLI usage
load_model_artifact()

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Service Health Check & Model Metadata",
    description="Returns the operational status, serving model version, and champion model algorithm."
)
def health_check():
    """
    GET /health
    Returns HTTP 200 with model version and champion model name if loaded,
    or HTTP 503 if model artifact is unavailable.
    """
    if app_state["model"] is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "model_version": MODEL_VERSION,
                "detail": "Model artifact is not loaded. Train and serialize model_artifact.joblib."
            }
        )

    return HealthResponse(
        status="healthy",
        model_version=MODEL_VERSION,
        champion_model=app_state.get("champion_model", DEFAULT_CHAMPION_MODEL)
    )

@app.get("/", tags=["System"], summary="API Root Overview")
def root():
    """Returns basic service descriptor and documentation links."""
    return {
        "service": "Quantitative Credit Risk Serving API",
        "version": MODEL_VERSION,
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/predict",
        "batch_predict_url": "/predict/batch"
    }

@app.post(
    "/predict",
    response_model=LoanPredictionResponse,
    tags=["Scoring"],
    summary="Score Single Loan Application",
    description=(
        "Accepts applicant and loan features, processes them through the optimal "
        "DP FICO bucketer and preprocessor, and returns a calibrated Probability of Default (PD)."
    )
)
def predict_single(application: LoanApplication):
    """
    POST /predict
    Evaluates a single customer loan application.
    """
    model = app_state.get("model")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded. Service is currently unable to score applications."
        )

    # Convert application data to DataFrame matching pipeline feature contract
    input_df = pd.DataFrame([{
        'FICO_score': application.FICO_score,
        'income': application.income,
        'loan_amount': application.loan_amount,
        'dti': application.dti,
        'employment_length': application.employment_length
    }])

    try:
        # Predict calibrated probability of default
        prob_default = float(model.predict_proba(input_df)[0, 1])
        pred_class = int(model.predict(input_df)[0])
    except Exception as e:
        logger.error(f"Inference execution error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation failure: {str(e)}"
        )

    risk_tier = get_risk_tier(prob_default)
    fico_bucket = get_fico_bucket_label(application.FICO_score, app_state.get("bucketer"))

    return LoanPredictionResponse(
        probability_of_default=prob_default,
        predicted_default=pred_class,
        predicted_class=pred_class,
        risk_tier=risk_tier,
        fico_bucket=fico_bucket,
        customer_id=application.customer_id,
        loan_id=application.loan_id
    )

@app.post(
    "/predict/batch",
    response_model=BatchLoanPredictionResponse,
    tags=["Scoring"],
    summary="Score Batch of Loan Applications",
    description="Vectorized batch scoring for multiple loan applications returning individual predictions."
)
def predict_batch(batch: BatchLoanApplication):
    """
    POST /predict/batch
    Performs vectorized batch inference across multiple loan records.
    """
    model = app_state.get("model")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded. Service is currently unable to score applications."
        )

    loans = batch.loans or []
    if len(loans) == 0:
        return BatchLoanPredictionResponse(
            predictions=[],
            total_count=0,
            model_version=MODEL_VERSION,
            champion_model=app_state.get("champion_model", DEFAULT_CHAMPION_MODEL)
        )

    # Vectorized DataFrame construction
    batch_df = pd.DataFrame([
        {
            'FICO_score': app.FICO_score,
            'income': app.income,
            'loan_amount': app.loan_amount,
            'dti': app.dti,
            'employment_length': app.employment_length
        }
        for app in loans
    ])

    try:
        probs = model.predict_proba(batch_df)[:, 1]
        preds = model.predict(batch_df)
    except Exception as e:
        logger.error(f"Batch inference execution error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference computation failure: {str(e)}"
        )

    bucketer = app_state.get("bucketer")
    predictions = []
    for i, app in enumerate(loans):
        pd_val = float(probs[i])
        pred_cls = int(preds[i])
        tier = get_risk_tier(pd_val)
        bucket = get_fico_bucket_label(app.FICO_score, bucketer)

        predictions.append(LoanPredictionResponse(
            probability_of_default=pd_val,
            predicted_default=pred_cls,
            predicted_class=pred_cls,
            risk_tier=tier,
            fico_bucket=bucket,
            customer_id=app.customer_id,
            loan_id=app.loan_id
        ))

    return BatchLoanPredictionResponse(
        predictions=predictions,
        total_count=len(predictions),
        model_version=MODEL_VERSION,
        champion_model=app_state.get("champion_model", DEFAULT_CHAMPION_MODEL)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
