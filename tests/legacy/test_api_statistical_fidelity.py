"""
test_api_statistical_fidelity.py

Adversarial empirical challenge test suite for the Quantitative Credit Risk Serving API (api.py).
Focus areas:
1. Single vs Batch Prediction Parity (|Delta p| < 10^-12) across diverse borrower profiles, boundary conditions, and batch sizes.
2. Credit Tier Monotonicity: Tier 0 > Tier 1 > Tier 2 > Tier 3 > Tier 4 default probability across varied borrower covariates and exact DP bin boundaries.
3. Unmodeled Metadata Passthrough Invariance: customer_id, loan_id, timestamp, and arbitrary extra fields having bit-for-bit zero impact on probability calculations (|Delta p| == 0.0).
4. Inference Fidelity and Statistical Calibration: Strict probability bounds (0, 1), correct risk tier mapping, accurate FICO bucket labels, and decision threshold consistency.
"""

import os
import sys
import uuid
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

# Ensure current directory and site-packages are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SITE_PKG = r"D:\Lib\site-packages"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if SITE_PKG not in sys.path:
    sys.path.insert(0, SITE_PKG)

from api import (
    app,
    load_model_artifact,
    app_state,
    MODEL_VERSION,
    get_risk_tier,
    get_fico_bucket_label
)


@pytest.fixture(scope="module")
def client():
    """Initializes TestClient and ensures model artifact is loaded."""
    loaded = load_model_artifact()
    assert loaded is True, "Model artifact failed to load."
    with TestClient(app) as c:
        yield c


class TestSingleVsBatchPredictionParity:
    """
    Validates that scoring loan applications individually via /predict yields
    numerically identical probabilities to vectorized batch scoring via /predict/batch (|Delta p| < 10^-12).
    """

    def test_parity_curated_profiles(self, client):
        """Tests parity on distinct borrower credit profiles."""
        profiles = [
            {
                "customer_id": "CUST-001",
                "loan_id": "LN-001",
                "FICO_score": 780.0,
                "income": 125000.0,
                "loan_amount": 10000.0,
                "dti": 0.12,
                "employment_length": 9.0
            },
            {
                "customer_id": "CUST-002",
                "loan_id": "LN-002",
                "FICO_score": 715.0,
                "income": 82000.0,
                "loan_amount": 16000.0,
                "dti": 0.28,
                "employment_length": 5.0
            },
            {
                "customer_id": "CUST-003",
                "loan_id": "LN-003",
                "FICO_score": 675.0,
                "income": 54000.0,
                "loan_amount": 22000.0,
                "dti": 0.38,
                "employment_length": 3.0
            },
            {
                "customer_id": "CUST-004",
                "loan_id": "LN-004",
                "FICO_score": 630.0,
                "income": 41000.0,
                "loan_amount": 25000.0,
                "dti": 0.45,
                "employment_length": 2.0
            },
            {
                "customer_id": "CUST-005",
                "loan_id": "LN-005",
                "FICO_score": 520.0,
                "income": 28000.0,
                "loan_amount": 30000.0,
                "dti": 0.58,
                "employment_length": 0.5
            }
        ]

        # 1. Single scoring
        single_pds = []
        for p in profiles:
            res = client.post("/predict", json=p)
            assert res.status_code == 200
            single_pds.append(res.json()["probability_of_default"])

        # 2. Batch scoring
        batch_res = client.post("/predict/batch", json={"loans": profiles})
        assert batch_res.status_code == 200
        batch_data = batch_res.json()
        assert batch_data["total_count"] == len(profiles)

        # 3. Precision assertion: |Delta p| < 10^-12
        max_delta = 0.0
        for i, pred in enumerate(batch_data["predictions"]):
            delta = abs(pred["probability_of_default"] - single_pds[i])
            if delta > max_delta:
                max_delta = delta
            assert delta < 1e-12, (
                f"Parity failure at index {i}: single={single_pds[i]}, "
                f"batch={pred['probability_of_default']}, delta={delta:.2e}"
            )
        assert max_delta < 1e-12

    def test_parity_boundary_ficos(self, client):
        """Tests parity on FICO boundary values and cutoffs."""
        boundary_ficos = [300.0, 599.99, 600.0, 659.99, 660.0, 694.99, 695.0, 740.99, 741.0, 850.0]
        applications = [
            {
                "customer_id": f"CUST-BOUND-{i}",
                "loan_id": f"LN-BOUND-{i}",
                "FICO_score": score,
                "income": 65000.0,
                "loan_amount": 15000.0,
                "dti": 0.25,
                "employment_length": 4.0
            }
            for i, score in enumerate(boundary_ficos)
        ]

        single_resps = [client.post("/predict", json=app).json() for app in applications]
        batch_res = client.post("/predict/batch", json={"loans": applications}).json()

        for i, s_res in enumerate(single_resps):
            b_res = batch_res["predictions"][i]
            delta = abs(b_res["probability_of_default"] - s_res["probability_of_default"])
            assert delta < 1e-12, f"Boundary FICO {applications[i]['FICO_score']} failed parity: delta={delta:.2e}"
            assert b_res["predicted_default"] == s_res["predicted_default"]
            assert b_res["risk_tier"] == s_res["risk_tier"]
            assert b_res["fico_bucket"] == s_res["fico_bucket"]

    def test_parity_large_synthetic_corpus(self, client):
        """
        Stress test: Generates 150 randomized borrower profiles spanning the entire feature domain.
        Verifies machine-precision parity |Delta p| < 10^-12 across all 150 evaluations.
        """
        np.random.seed(42)
        n_samples = 150
        corpus = []
        for i in range(n_samples):
            corpus.append({
                "customer_id": f"SYNTH-{i:04d}",
                "loan_id": f"LN-SYNTH-{i:04d}",
                "FICO_score": float(np.random.uniform(300.0, 850.0)),
                "income": float(np.random.uniform(15000.0, 350000.0)),
                "loan_amount": float(np.random.uniform(1000.0, 100000.0)),
                "dti": float(np.random.uniform(0.01, 1.2)),
                "employment_length": float(np.random.uniform(0.0, 30.0))
            })

        # Score individually
        single_probs = []
        for app in corpus:
            res = client.post("/predict", json=app)
            assert res.status_code == 200
            single_probs.append(res.json()["probability_of_default"])

        # Score in a single batch
        batch_res = client.post("/predict/batch", json={"loans": corpus})
        assert batch_res.status_code == 200
        batch_preds = batch_res.json()["predictions"]

        deltas = [abs(batch_preds[i]["probability_of_default"] - single_probs[i]) for i in range(n_samples)]
        max_delta = max(deltas)
        assert max_delta < 1e-12, f"Large synthetic corpus max delta exceeded 10^-12: {max_delta:.2e}"

    def test_parity_variable_batch_sizes(self, client):
        """
        Tests that batch chunking does not alter numerical probabilities.
        Splits 40 records into batch sizes of [1, 5, 8, 20, 40].
        """
        np.random.seed(99)
        records = [
            {
                "FICO_score": float(np.random.uniform(350.0, 820.0)),
                "income": float(np.random.uniform(25000.0, 180000.0)),
                "loan_amount": float(np.random.uniform(3000.0, 45000.0)),
                "dti": float(np.random.uniform(0.05, 0.65)),
                "employment_length": float(np.random.uniform(0.5, 15.0))
            }
            for _ in range(40)
        ]

        # Baseline: single scores
        baseline_pds = [client.post("/predict", json=r).json()["probability_of_default"] for r in records]

        # Chunk tests
        for chunk_size in [1, 5, 8, 20, 40]:
            chunked_pds = []
            for start_idx in range(0, len(records), chunk_size):
                chunk = records[start_idx:start_idx + chunk_size]
                b_res = client.post("/predict/batch", json={"loans": chunk})
                assert b_res.status_code == 200
                chunked_pds.extend([p["probability_of_default"] for p in b_res.json()["predictions"]])

            for i in range(len(records)):
                delta = abs(chunked_pds[i] - baseline_pds[i])
                assert delta < 1e-12, f"Chunk size {chunk_size} parity failure at idx {i}: delta={delta:.2e}"


class TestCreditTierMonotonicity:
    """
    Validates strict monotonicity of Probability of Default across credit tiers:
    Tier 0 (FICO < 600) > Tier 1 (600 <= FICO < 660) > Tier 2 (660 <= FICO < 695) >
    Tier 3 (695 <= FICO < 741) > Tier 4 (FICO >= 741) default probability.
    """

    # Representative mid-points for the 5 tiers discovered by DP bucketer
    TIER_REPRESENTATIVE_FICO = {
        0: 550.0,  # [-inf, 600.0)
        1: 630.0,  # [600.0, 660.0)
        2: 675.0,  # [660.0, 695.0)
        3: 715.0,  # [695.0, 741.0)
        4: 780.0   # [741.0, inf)
    }

    def test_monotonicity_fixed_covariates(self, client):
        """
        Holds continuous covariates constant and verifies:
        PD(Tier 0) > PD(Tier 1) > PD(Tier 2) > PD(Tier 3) > PD(Tier 4).
        """
        base_features = {
            "income": 75000.0,
            "loan_amount": 18000.0,
            "dti": 0.30,
            "employment_length": 5.0
        }

        pds = []
        for tier_idx in range(5):
            payload = dict(base_features)
            payload["FICO_score"] = self.TIER_REPRESENTATIVE_FICO[tier_idx]
            res = client.post("/predict", json=payload)
            assert res.status_code == 200
            pds.append(res.json()["probability_of_default"])

        # Strict monotonic decrease as credit quality improves
        for i in range(len(pds) - 1):
            assert pds[i] > pds[i + 1], (
                f"Strict monotonicity violation: Tier {i} PD ({pds[i]:.6f}) "
                f"not greater than Tier {i+1} PD ({pds[i+1]:.6f})"
            )

    def test_monotonicity_across_diverse_covariate_profiles(self, client):
        """
        Tests 30 randomized combinations of (income, loan_amount, dti, employment_length).
        For every combination, verifies strict tier monotonicity: Tier 0 > Tier 1 > Tier 2 > Tier 3 > Tier 4.
        """
        np.random.seed(123)
        n_profiles = 30

        for profile_idx in range(n_profiles):
            inc = float(np.random.uniform(20000.0, 200000.0))
            loan = float(np.random.uniform(2000.0, 60000.0))
            dti = float(np.random.uniform(0.05, 0.70))
            emp = float(np.random.uniform(0.5, 20.0))

            tier_apps = [
                {
                    "FICO_score": self.TIER_REPRESENTATIVE_FICO[t],
                    "income": inc,
                    "loan_amount": loan,
                    "dti": dti,
                    "employment_length": emp
                }
                for t in range(5)
            ]

            batch_res = client.post("/predict/batch", json={"loans": tier_apps})
            assert batch_res.status_code == 200
            pds = [p["probability_of_default"] for p in batch_res.json()["predictions"]]

            for t in range(4):
                assert pds[t] > pds[t + 1], (
                    f"Profile {profile_idx} monotonicity failed: "
                    f"Tier {t} ({pds[t]:.5f}) <= Tier {t+1} ({pds[t+1]:.5f})"
                )

    def test_monotonicity_across_exact_dp_boundaries(self, client):
        """
        Tests infinitesimal step epsilon across exact DP partition cutoffs:
        Boundaries: 600.0, 660.0, 695.0, 741.0.
        Verifies:
        PD(599.999) > PD(600.000)
        PD(659.999) > PD(660.000)
        PD(694.999) > PD(695.000)
        PD(740.999) > PD(741.000)
        """
        cutoffs = [600.0, 660.0, 695.0, 741.0]
        base_loan = {
            "income": 60000.0,
            "loan_amount": 12000.0,
            "dti": 0.22,
            "employment_length": 4.0
        }

        for cutoff in cutoffs:
            below_score = cutoff - 0.001
            above_score = cutoff

            app_below = dict(base_loan, FICO_score=below_score)
            app_above = dict(base_loan, FICO_score=above_score)

            res_below = client.post("/predict", json=app_below).json()
            res_above = client.post("/predict", json=app_above).json()

            pd_below = res_below["probability_of_default"]
            pd_above = res_above["probability_of_default"]

            assert pd_below > pd_above, (
                f"Boundary transition failure at cutoff {cutoff}: "
                f"PD({below_score})={pd_below:.6f} not > PD({above_score})={pd_above:.6f}"
            )
            # Ensure bucket label changes across boundary
            assert res_below["fico_bucket"] != res_above["fico_bucket"]

    def test_intra_bucket_flatness(self, client):
        """
        Verifies that two different FICO scores falling in the SAME bucket
        produce identical probability of default when continuous features are identical.
        (Discretization invariance within bucket).
        """
        pairs_in_same_bucket = [
            (350.0, 550.0, "[-inf, 600.0)"),
            (610.0, 650.0, "[600.0, 660.0)"),
            (665.0, 690.0, "[660.0, 695.0)"),
            (700.0, 735.0, "[695.0, 741.0)"),
            (750.0, 830.0, "[741.0, inf)")
        ]

        base_loan = {
            "income": 70000.0,
            "loan_amount": 15000.0,
            "dti": 0.25,
            "employment_length": 6.0
        }

        for f1, f2, expected_label in pairs_in_same_bucket:
            res1 = client.post("/predict", json=dict(base_loan, FICO_score=f1)).json()
            res2 = client.post("/predict", json=dict(base_loan, FICO_score=f2)).json()

            assert abs(res1["probability_of_default"] - res2["probability_of_default"]) < 1e-12, (
                f"Intra-bucket variation found between FICO {f1} and {f2}"
            )
            assert res1["fico_bucket"] == expected_label
            assert res2["fico_bucket"] == expected_label


class TestMetadataPassthroughInvariance:
    """
    Validates that unmodeled metadata fields (customer_id, loan_id, timestamp, arbitrary extra attributes)
    have bit-for-bit zero impact on model predictions and probability calculations (|Delta p| == 0.0).
    """

    def test_metadata_invariance_single_predict(self, client):
        """
        Evaluates a loan with no metadata, standard metadata, and heavy unmodeled extra fields.
        Verifies exact bitwise probability equivalence.
        """
        core_features = {
            "FICO_score": 710.0,
            "income": 85000.0,
            "loan_amount": 20000.0,
            "dti": 0.28,
            "employment_length": 6.0
        }

        # 1. Clean core features only
        res_clean = client.post("/predict", json=core_features)
        assert res_clean.status_code == 200
        pd_clean = res_clean.json()["probability_of_default"]

        # 2. Standard IDs
        payload_standard = dict(core_features, customer_id="CUST-ALPHA-1", loan_id="LN-BETA-2")
        res_standard = client.post("/predict", json=payload_standard)
        assert res_standard.status_code == 200
        data_standard = res_standard.json()
        assert data_standard["probability_of_default"] == pd_clean
        assert data_standard["customer_id"] == "CUST-ALPHA-1"
        assert data_standard["loan_id"] == "LN-BETA-2"

        # 3. Heavy metadata injection (timestamps, UUIDs, nested fields, notes)
        payload_heavy = dict(
            core_features,
            customer_id=str(uuid.uuid4()),
            loan_id=f"LN-{uuid.uuid4()}",
            timestamp="2026-09-05T14:30:00.123456Z",
            application_source="partner_portal",
            underwriter="AutoUnderwriter_v2",
            branch_id="US-WEST-90210",
            notes="Applicant has 10 years clean tenure at Fortune 500 employer",
            client_ip="192.168.1.100",
            session_token="sec_token_999a88b77c",
            internal_risk_rating=1.05
        )
        res_heavy = client.post("/predict", json=payload_heavy)
        assert res_heavy.status_code == 200
        data_heavy = res_heavy.json()
        assert data_heavy["probability_of_default"] == pd_clean
        assert data_heavy["customer_id"] == payload_heavy["customer_id"]
        assert data_heavy["loan_id"] == payload_heavy["loan_id"]

    def test_metadata_invariance_batch_predict(self, client):
        """
        Tests that batch prediction is completely invariant to heterogeneous metadata across rows.
        Row 0: no metadata
        Row 1: basic IDs
        Row 2: timestamp + extra tags
        """
        core1 = {"FICO_score": 750.0, "income": 95000.0, "loan_amount": 15000.0, "dti": 0.18, "employment_length": 7.0}
        core2 = {"FICO_score": 580.0, "income": 35000.0, "loan_amount": 25000.0, "dti": 0.50, "employment_length": 1.0}

        # Baseline single scores
        single1 = client.post("/predict", json=core1).json()["probability_of_default"]
        single2 = client.post("/predict", json=core2).json()["probability_of_default"]

        # Batch with varied unmodeled metadata
        batch_payload = {
            "loans": [
                dict(core1, customer_id="CID-1", loan_id="LID-1", timestamp="2026-09-01T00:00:00Z"),
                dict(core2, customer_id="CID-2", loan_id="LID-2", timestamp="2026-09-02T12:00:00Z", tag="subprime_review"),
                dict(core1),  # completely unadorned
                dict(core2, extra_arbitrary_key="random_payload_value", number_of_dependents=3)
            ]
        }
        batch_res = client.post("/predict/batch", json=batch_payload)
        assert batch_res.status_code == 200
        preds = batch_res.json()["predictions"]

        # Assert exact bitwise match
        assert preds[0]["probability_of_default"] == single1
        assert preds[1]["probability_of_default"] == single2
        assert preds[2]["probability_of_default"] == single1
        assert preds[3]["probability_of_default"] == single2

        # Assert passthrough fidelity
        assert preds[0]["customer_id"] == "CID-1"
        assert preds[1]["customer_id"] == "CID-2"
        assert preds[2]["customer_id"] is None
        assert preds[3]["customer_id"] is None

    def test_alias_choices_field_invariance(self, client):
        """
        Tests that alias choices for FICO_score ('fico_score', 'fico') produce
        numerically identical probability to canonical 'FICO_score'.
        """
        canonical = {
            "FICO_score": 680.0,
            "income": 60000.0,
            "loan_amount": 14000.0,
            "dti": 0.25,
            "employment_length": 4.0
        }
        alias1 = dict(canonical)
        del alias1["FICO_score"]
        alias1["fico_score"] = 680.0

        alias2 = dict(canonical)
        del alias2["FICO_score"]
        alias2["fico"] = 680.0

        res_canon = client.post("/predict", json=canonical).json()["probability_of_default"]
        res_alias1 = client.post("/predict", json=alias1).json()["probability_of_default"]
        res_alias2 = client.post("/predict", json=alias2).json()["probability_of_default"]

        assert res_canon == res_alias1
        assert res_canon == res_alias2


class TestInferenceFidelityAndStatisticalCalibration:
    """
    Validates statistical fidelity of API inference outputs:
    - Probabilities strictly bounded in (0.0, 1.0)
    - Risk tiers strictly follow defined cutoffs (<0.20 Low, 0.20-0.45 Moderate, 0.45-0.70 High, >=0.70 Very High)
    - FICO bucket string correctly reflects the DP cutoffs
    - Decision threshold consistency (predicted_default == 1 iff PD >= 0.50)
    """

    def test_probability_range_extreme_cases(self, client):
        """
        Evaluates super-prime and distressed-subprime extremes to ensure
        probabilities remain strictly within (0.0, 1.0) without overflowing or saturating to 0/1.
        """
        super_prime = {
            "FICO_score": 850.0,
            "income": 1000000.0,
            "loan_amount": 1000.0,
            "dti": 0.001,
            "employment_length": 40.0
        }
        distressed = {
            "FICO_score": 300.0,
            "income": 5000.0,
            "loan_amount": 200000.0,
            "dti": 5.0,
            "employment_length": 0.0
        }

        res_sp = client.post("/predict", json=super_prime).json()
        res_ds = client.post("/predict", json=distressed).json()

        assert 0.0 < res_sp["probability_of_default"] < 0.10, "Super-prime PD should be strictly in (0, 0.10)"
        assert 0.90 < res_ds["probability_of_default"] < 1.0, "Distressed PD should be strictly in (0.90, 1.0)"

    def test_risk_tier_classification_fidelity(self, client):
        """
        Verifies risk tier labeling exactly matches probability thresholds across 50 applications.
        """
        np.random.seed(777)
        apps = [
            {
                "FICO_score": float(np.random.uniform(300.0, 850.0)),
                "income": float(np.random.uniform(20000.0, 250000.0)),
                "loan_amount": float(np.random.uniform(2000.0, 50000.0)),
                "dti": float(np.random.uniform(0.05, 0.8)),
                "employment_length": float(np.random.uniform(0.0, 20.0))
            }
            for _ in range(50)
        ]

        batch_res = client.post("/predict/batch", json={"loans": apps}).json()
        for pred in batch_res["predictions"]:
            pd_val = pred["probability_of_default"]
            tier = pred["risk_tier"]
            if pd_val < 0.20:
                assert tier == "Low Risk", f"PD={pd_val} but tier is {tier}"
            elif pd_val < 0.45:
                assert tier == "Moderate Risk", f"PD={pd_val} but tier is {tier}"
            elif pd_val < 0.70:
                assert tier == "High Risk", f"PD={pd_val} but tier is {tier}"
            else:
                assert tier == "Very High Risk", f"PD={pd_val} but tier is {tier}"

    def test_fico_bucket_labeling_fidelity(self, client):
        """
        Verifies that returned fico_bucket strings match DP cutoffs exactly.
        """
        test_points = [
            (300.0, "[-inf, 600.0)"),
            (599.9, "[-inf, 600.0)"),
            (600.0, "[600.0, 660.0)"),
            (659.9, "[600.0, 660.0)"),
            (660.0, "[660.0, 695.0)"),
            (694.9, "[660.0, 695.0)"),
            (695.0, "[695.0, 741.0)"),
            (740.9, "[695.0, 741.0)"),
            (741.0, "[741.0, inf)"),
            (850.0, "[741.0, inf)")
        ]

        base = {"income": 60000.0, "loan_amount": 10000.0, "dti": 0.2, "employment_length": 4.0}
        for fico_val, expected_bucket in test_points:
            res = client.post("/predict", json=dict(base, FICO_score=fico_val)).json()
            assert res["fico_bucket"] == expected_bucket, (
                f"FICO {fico_val} expected bucket {expected_bucket}, got {res['fico_bucket']}"
            )

    def test_decision_threshold_consistency(self, client):
        """
        Verifies that predicted_default is 1 if and only if probability_of_default >= 0.50,
        and predicted_class is always equal to predicted_default.
        """
        np.random.seed(888)
        apps = [
            {
                "FICO_score": float(np.random.uniform(300.0, 850.0)),
                "income": float(np.random.uniform(20000.0, 200000.0)),
                "loan_amount": float(np.random.uniform(2000.0, 60000.0)),
                "dti": float(np.random.uniform(0.05, 0.9)),
                "employment_length": float(np.random.uniform(0.0, 25.0))
            }
            for _ in range(60)
        ]

        batch_res = client.post("/predict/batch", json={"loans": apps}).json()
        for pred in batch_res["predictions"]:
            pd_val = pred["probability_of_default"]
            pred_def = pred["predicted_default"]
            pred_cls = pred["predicted_class"]

            assert pred_def == (1 if pd_val >= 0.50 else 0)
            assert pred_cls == pred_def


if __name__ == "__main__":
    pytest.main(["-v", __file__])
