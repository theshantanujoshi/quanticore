import pytest

import numpy as np
from fastapi.testclient import TestClient
from api import (
    app,
    app_state,
    load_model_artifact,
    unload_model_artifact,
    MODEL_VERSION,
    DEFAULT_CHAMPION_MODEL,
    get_risk_tier,
    get_fico_bucket_label
)

@pytest.fixture(scope="module")
def client():
    """Provides a TestClient with model loaded, restoring state after tests."""
    load_model_artifact()
    with TestClient(app) as c:
        yield c
    # Re-ensure model is loaded after tests complete
    load_model_artifact()

@pytest.fixture
def prime_application():
    return {
        "customer_id": "CUST-PRIME-01",
        "loan_id": "LN-1001",
        "FICO_score": 780.0,
        "income": 120000.0,
        "loan_amount": 10000.0,
        "dti": 0.15,
        "employment_length": 8.0
    }

@pytest.fixture
def subprime_application():
    return {
        "customer_id": "CUST-SUBPRIME-01",
        "loan_id": "LN-9001",
        "FICO_score": 550.0,
        "income": 30000.0,
        "loan_amount": 25000.0,
        "dti": 0.55,
        "employment_length": 1.0
    }

def test_health_endpoint_healthy(client):
    """
    Validates GET /health returns HTTP 200, operational status, model version,
    and champion model name.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_version"] == MODEL_VERSION
    assert "champion_model" in data
    assert data["champion_model"] == "Logistic Regression"

def test_root_endpoint(client):
    """
    Validates GET / root returns HTTP 200 with service metadata and documentation links.
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["version"] == MODEL_VERSION
    assert "docs_url" in data
    assert "health_url" in data

def test_acceptance_criteria_predict_verbatim(client, prime_application):
    """
    Acceptance Criteria Test:
    A test script (test_api.py) can successfully send a request to the API
    with sample features and receive a valid JSON response containing a probability
    (float between 0 and 1).
    """
    response = client.post("/predict", json=prime_application)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    
    # 1. Valid JSON response structure
    assert isinstance(data, dict)
    assert "probability_of_default" in data
    
    # 2. Probability of default is a float between 0 and 1
    pd_val = data["probability_of_default"]
    assert isinstance(pd_val, float), f"Expected float, got {type(pd_val)}"
    assert 0.0 <= pd_val <= 1.0, f"PD {pd_val} is out of bounds [0.0, 1.0]"
    
    # 3. Accompanying decision fields
    assert "predicted_default" in data
    assert data["predicted_default"] in [0, 1]
    assert "predicted_class" in data
    assert data["predicted_class"] == data["predicted_default"]
    assert "risk_tier" in data
    assert "fico_bucket" in data

def test_predict_single_prime_borrower(client, prime_application):
    """
    Validates that a prime borrower profile yields a low probability of default (< 0.20),
    predicted non-default (0), Low Risk tier, and top FICO bucket.
    """
    response = client.post("/predict", json=prime_application)
    assert response.status_code == 200
    data = response.json()
    
    assert data["probability_of_default"] < 0.20
    assert data["predicted_default"] == 0
    assert data["risk_tier"] == "Low Risk"
    assert data["fico_bucket"] == "[741.0, inf)"
    assert data["customer_id"] == prime_application["customer_id"]
    assert data["loan_id"] == prime_application["loan_id"]

def test_predict_single_subprime_borrower(client, subprime_application):
    """
    Validates that a subprime borrower profile yields a high probability of default (> 0.70),
    predicted default (1), Very High Risk tier, and lowest FICO bucket.
    """
    response = client.post("/predict", json=subprime_application)
    assert response.status_code == 200
    data = response.json()
    
    assert data["probability_of_default"] > 0.70
    assert data["predicted_default"] == 1
    assert data["risk_tier"] == "Very High Risk"
    assert data["fico_bucket"] == "[-inf, 600.0)"
    assert data["customer_id"] == subprime_application["customer_id"]
    assert data["loan_id"] == subprime_application["loan_id"]

def test_borrower_profiles_monotonicity(client):
    """
    Tests financial and credit monotonicity across borrower profiles:
    As borrower risk factors worsen (lower FICO, higher DTI, lower income),
    predicted Probability of Default (PD) must strictly increase.
    """
    profiles = [
        # Prime
        {"FICO_score": 790.0, "income": 130000.0, "loan_amount": 10000.0, "dti": 0.12, "employment_length": 10.0},
        # Good / Near-prime
        {"FICO_score": 710.0, "income": 80000.0, "loan_amount": 15000.0, "dti": 0.25, "employment_length": 5.0},
        # Fair / High Risk
        {"FICO_score": 670.0, "income": 50000.0, "loan_amount": 20000.0, "dti": 0.38, "employment_length": 3.0},
        # Subprime / Distressed
        {"FICO_score": 520.0, "income": 25000.0, "loan_amount": 28000.0, "dti": 0.60, "employment_length": 0.5}
    ]
    
    pds = []
    tiers = []
    for p in profiles:
        res = client.post("/predict", json=p)
        assert res.status_code == 200
        d = res.json()
        pds.append(d["probability_of_default"])
        tiers.append(d["risk_tier"])
        
    # Strictly monotonic increase in probability of default
    for i in range(len(pds) - 1):
        assert pds[i] < pds[i + 1], f"Monotonicity violation: Profile {i} PD ({pds[i]}) >= Profile {i+1} PD ({pds[i+1]})"
        
    # Verify risk tiers progress from Low to Very High
    assert tiers[0] == "Low Risk"
    assert tiers[-1] == "Very High Risk"

def test_predict_batch_vectorized(client, prime_application, subprime_application):
    """
    Validates batch prediction endpoint:
    - Accepts list of applications under 'loans' key
    - Performs vectorized scoring
    - Results match individual /predict calls with high precision
    """
    moderate_app = {
        "customer_id": "CUST-MOD-01",
        "loan_id": "LN-5001",
        "FICO_score": 710.0,
        "income": 80000.0,
        "loan_amount": 15000.0,
        "dti": 0.25,
        "employment_length": 5.0
    }
    
    batch_payload = {"loans": [prime_application, moderate_app, subprime_application]}
    batch_res = client.post("/predict/batch", json=batch_payload)
    assert batch_res.status_code == 200
    batch_data = batch_res.json()
    
    assert batch_data["total_count"] == 3
    assert len(batch_data["predictions"]) == 3
    assert batch_data["model_version"] == MODEL_VERSION
    assert batch_data["champion_model"] == "Logistic Regression"
    
    # Compare with individual single predictions
    for i, app_payload in enumerate(batch_payload["loans"]):
        single_res = client.post("/predict", json=app_payload)
        assert single_res.status_code == 200
        single_data = single_res.json()
        
        batch_pred = batch_data["predictions"][i]
        assert np.isclose(batch_pred["probability_of_default"], single_data["probability_of_default"], atol=1e-6)
        assert batch_pred["predicted_default"] == single_data["predicted_default"]
        assert batch_pred["risk_tier"] == single_data["risk_tier"]
        assert batch_pred["fico_bucket"] == single_data["fico_bucket"]
        assert batch_pred["customer_id"] == app_payload["customer_id"]
        assert batch_pred["loan_id"] == app_payload["loan_id"]

def test_predict_batch_empty(client):
    """
    Validates batch prediction with an empty loans list.
    """
    response = client.post("/predict/batch", json={"loans": []})
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 0
    assert data["predictions"] == []

def test_predict_batch_applications_alias(client, prime_application):
    """
    Validates batch prediction accepting 'applications' key as an alternative to 'loans'.
    """
    response = client.post("/predict/batch", json={"applications": [prime_application]})
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 1
    assert len(data["predictions"]) == 1

def test_unmodeled_extra_fields_ignored_and_passthrough(client):
    """
    Validates that unmodeled extra fields (e.g. metadata, IDs, notes, timestamps)
    pass validation cleanly and are safely dropped or passed through without causing 422.
    """
    payload = {
        "customer_id": "CUST-EXTRA-99",
        "loan_id": "LN-EXTRA-42",
        "timestamp": "2026-09-05T12:00:00Z",
        "branch_code": "NYC-04",
        "underwriter_notes": "Prime applicant with clean repayment history",
        "internal_score": 98.5,
        "FICO_score": 750.0,
        "income": 95000.0,
        "loan_amount": 12000.0,
        "dti": 0.18,
        "employment_length": 7.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["probability_of_default"] <= 1.0
    assert data["customer_id"] == "CUST-EXTRA-99"
    assert data["loan_id"] == "LN-EXTRA-42"

def test_validation_negative_income(client):
    """Validates that negative income returns HTTP 422 Unprocessable Entity."""
    payload = {
        "FICO_score": 720.0,
        "income": -50000.0,
        "loan_amount": 10000.0,
        "dti": 0.25,
        "employment_length": 5.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    err = response.json()
    assert "detail" in err

def test_validation_zero_income(client):
    """Validates that zero income returns HTTP 422 (income must be strictly > 0)."""
    payload = {
        "FICO_score": 720.0,
        "income": 0.0,
        "loan_amount": 10000.0,
        "dti": 0.25,
        "employment_length": 5.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_validation_negative_loan_amount(client):
    """Validates that negative loan amount returns HTTP 422."""
    payload = {
        "FICO_score": 720.0,
        "income": 50000.0,
        "loan_amount": -5000.0,
        "dti": 0.25,
        "employment_length": 5.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_validation_zero_loan_amount(client):
    """Validates that zero loan amount returns HTTP 422."""
    payload = {
        "FICO_score": 720.0,
        "income": 50000.0,
        "loan_amount": 0.0,
        "dti": 0.25,
        "employment_length": 5.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_validation_negative_dti(client):
    """Validates that negative debt-to-income ratio returns HTTP 422."""
    payload = {
        "FICO_score": 720.0,
        "income": 50000.0,
        "loan_amount": 10000.0,
        "dti": -0.15,
        "employment_length": 5.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_validation_negative_employment_length(client):
    """Validates that negative employment length returns HTTP 422."""
    payload = {
        "FICO_score": 720.0,
        "income": 50000.0,
        "loan_amount": 10000.0,
        "dti": 0.25,
        "employment_length": -2.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_validation_fico_below_minimum(client):
    """Validates that FICO score below 300 returns HTTP 422."""
    payload = {
        "FICO_score": 280.0,
        "income": 50000.0,
        "loan_amount": 10000.0,
        "dti": 0.25,
        "employment_length": 5.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_validation_fico_above_maximum(client):
    """Validates that FICO score above 850 returns HTTP 422."""
    payload = {
        "FICO_score": 890.0,
        "income": 50000.0,
        "loan_amount": 10000.0,
        "dti": 0.25,
        "employment_length": 5.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_validation_missing_required_fields(client):
    """Validates that omitting required fields returns HTTP 422."""
    # Missing FICO_score
    res1 = client.post("/predict", json={"income": 50000.0, "loan_amount": 10000.0, "dti": 0.25, "employment_length": 5.0})
    assert res1.status_code == 422
    
    # Missing income
    res2 = client.post("/predict", json={"FICO_score": 700.0, "loan_amount": 10000.0, "dti": 0.25, "employment_length": 5.0})
    assert res2.status_code == 422
    
    # Missing loan_amount
    res3 = client.post("/predict", json={"FICO_score": 700.0, "income": 50000.0, "dti": 0.25, "employment_length": 5.0})
    assert res3.status_code == 422

def test_validation_invalid_data_types(client):
    """Validates that non-numeric types for numeric fields return HTTP 422."""
    payload = {
        "FICO_score": "excellent",
        "income": 50000.0,
        "loan_amount": 10000.0,
        "dti": 0.25,
        "employment_length": 5.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_boundary_fico_scores(client):
    """
    Boundary condition testing:
    Exact legal extremes for FICO score (300.0 and 850.0) must score cleanly.
    """
    for boundary_fico in [300.0, 850.0]:
        payload = {
            "FICO_score": boundary_fico,
            "income": 60000.0,
            "loan_amount": 12000.0,
            "dti": 0.25,
            "employment_length": 4.0
        }
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        d = res.json()
        assert 0.0 <= d["probability_of_default"] <= 1.0

def test_zero_dti_and_zero_employment(client):
    """
    Edge case: borrower with 0.0 DTI and 0.0 employment length.
    """
    payload = {
        "FICO_score": 720.0,
        "income": 80000.0,
        "loan_amount": 10000.0,
        "dti": 0.0,
        "employment_length": 0.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["probability_of_default"] <= 1.0

def test_service_unavailable_when_model_unloaded(client, prime_application):
    """
    Validates robust 503 error handling when model artifact is unloaded:
    - GET /health returns 503 Service Unavailable
    - POST /predict returns 503 Service Unavailable
    - POST /predict/batch returns 503 Service Unavailable
    - Re-loading model restores full operational status (HTTP 200)
    """
    try:
        # Unload model from memory
        unload_model_artifact()
        
        # Verify 503 responses
        h_res = client.get("/health")
        assert h_res.status_code == 503
        assert h_res.json()["status"] == "unhealthy"
        
        p_res = client.post("/predict", json=prime_application)
        assert p_res.status_code == 503
        assert "not loaded" in p_res.json()["detail"].lower()
        
        b_res = client.post("/predict/batch", json={"loans": [prime_application]})
        assert b_res.status_code == 503
        assert "not loaded" in b_res.json()["detail"].lower()
        
    finally:
        # Restore model
        reloaded = load_model_artifact()
        assert reloaded is True
        
        # Verify recovery
        recover_res = client.get("/health")
        assert recover_res.status_code == 200
        assert recover_res.json()["status"] == "healthy"

def test_openapi_schema(client):
    """
    Validates OpenAPI schema is auto-generated with expected paths and operations.
    """
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "paths" in schema
    assert "/health" in schema["paths"]
    assert "/predict" in schema["paths"]
    assert "/predict/batch" in schema["paths"]

if __name__ == "__main__":
    pytest.main(["-v", __file__])
