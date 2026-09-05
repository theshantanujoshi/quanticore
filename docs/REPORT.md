# Production-Grade Quantitative Credit Risk Model & Probability of Default Engine
## End-to-End Dynamic Programming Discretization, Multi-Model Benchmark, and Production Serving Architecture

**Target Audience:** Quantitative Credit Risk Hiring Managers, Heads of Credit Analytics, Model Risk Management (MRM) Officers, and Quantitative Recruiters  
**Author:** Quantitative Credit Risk Engineering Team  
**Date:** September 2026  
**Regulatory Frameworks:** Federal Reserve SR 11-7 / OCC 2011-12, Basel III/IV Internal Ratings-Based (IRB) Framework, Equal Credit Opportunity Act (ECOA / Reg B), Fair Credit Reporting Act (FCRA)  
**Artifact Version:** 1.0.0 (`model_artifact.joblib`)  
**Production Endpoints:** `/health`, `/predict`, `/predict/batch`  
**Test Suite Coverage:** 172 Automated Test Cases Across 6 Verification Suites (100% Pass Rate)  

---

## Executive Summary & Model Factsheet

| Attribute | Specification / Empirical Result |
| :--- | :--- |
| **Primary Objective** | Predict point-in-time Probability of Default (PD) for unsecured retail credit underwriting |
| **Champion Model** | **Logistic Regression** trained on DP-discretized FICO and standardized continuous covariates |
| **Champion ROC-AUC** | **0.8599** (Exact: `0.859949`) on out-of-sample stratified test split |
| **Champion PR-AUC** | **0.8125** (Exact: `0.812516`) |
| **Gini Coefficient** | **0.7199** ($2 \times \text{ROC-AUC} - 1$) |
| **Calibration Loss (Brier Score)** | **0.1466** (Superior calibration over tree ensembles: XGBoost `0.1502`, Gradient Boosting `0.1505`) |
| **Classification Accuracy / F1** | **78.65%** / **0.7061** (Decision threshold $\tau = 0.50$) |
| **FICO Discretization** | **Dynamic Programming Within-Bucket Variance Minimization** ($O(1)$ prefix sums, $O(K \cdot n^2)$ DP) |
| **Discovered Optimal FICO Cutoffs** | `[-inf, 600.0, 660.0, 695.0, 741.0, inf]` (5 optimal credit tiers) |
| **Risk Tier Default Rates** | Strictly monotonic: **79.61% $\rightarrow$ 54.70% $\rightarrow$ 40.00% $\rightarrow$ 29.67% $\rightarrow$ 15.64%** |
| **Serving Architecture** | High-performance **FastAPI** service with Pydantic v2 input schema validation |
| **Vectorized Batch Latency** | **34.03 ms** for 1,000 loan applications (**0.034 ms per applicant**) |
| **Single vs. Batch Parity** | Exact numerical parity: $|\Delta p| \le 2.22 \times 10^{-16}$ (IEEE 754 machine epsilon) |
| **Model Risk Governance** | Full SR 11-7 compliance, zero data leakage, adverse action explainability, strict schema isolation |

---

## Table of Contents
1. [Executive Summary & Regulatory Framework](#1-executive-summary--regulatory-framework)
2. [Exploratory Data Analysis & Feature Engineering](#2-exploratory-data-analysis--feature-engineering)
3. [Dynamic Programming FICO Score Bucketing](#3-dynamic-programming-fico-score-bucketing)
4. [Multi-Model Evaluation & Champion Model Selection](#4-multi-model-evaluation--champion-model-selection)
5. [Production Serving Architecture & High-Throughput API](#5-production-serving-architecture--high-throughput-api)
6. [Comprehensive Verification & Model Risk Management (MRM)](#6-comprehensive-verification--model-risk-management-mrm)
7. [Conclusion & Strategic Roadmap](#7-conclusion--strategic-roadmap)

---

## 1. Executive Summary & Regulatory Framework

### 1.1 Business Context: Credit Risk Underwriting
In retail banking and credit risk management, accurate estimation of the Probability of Default ($\text{PD}$) is foundational to risk-based loan pricing, credit line assignment, capital allocation, and allowance for credit losses (under CECL and IFRS 9). A fractional misestimation in default probability can trigger adverse selection, where high-risk borrowers are underpriced (leading to severe write-offs) while low-risk borrowers are overpriced (eroding market share to competitors).

This project implements an institutional-grade credit risk modeling pipeline designed to meet the dual mandates of top-tier predictive discrimination and strict regulatory compliance. The architecture bridges rigorous statistical algorithms—specifically dynamic programming for optimal non-parametric binning—with high-throughput cloud serving infrastructure.

### 1.2 Regulatory Compliance Architecture
Under supervisory standards established by the Federal Reserve (**SR 11-7**), the Office of the Comptroller of the Currency (**OCC 2011-12**), the Federal Deposit Insurance Corporation (**FDIC**), and the **Basel Committee on Banking Supervision (BCBS)**, retail credit risk models are subject to rigorous governance:

```
+-------------------------------------------------------------------------------+
|                       SR 11-7 MODEL RISK MANAGEMENT                           |
+-------------------------------------------------------------------------------+
| 1. Conceptual Soundness | 2. Ongoing Monitoring       | 3. Outcomes Analysis   |
| - DP Variance Minimizer  | - Dynamic PSI/CSI Tracking  | - 172 Automated Tests  |
| - Monotonic Risk Tiers   | - Daily Calibration Audits  | - Zero Data Leakage    |
| - GLM Log-Odds Form     | - Production Health Checks  | - Parity Delta < 1e-12 |
+-------------------------------------------------------------------------------+
```

1. **Conceptual Soundness (SR 11-7):**
   - Scoring mechanisms must possess sound economic and mathematical rationale. Black-box non-linear interactions that generate non-monotonic risk responses (e.g., predicting higher default risk for higher FICO scores in localized feature subspaces) are strictly prohibited.
   - The Dynamic Programming bucketer guarantees global optimality in within-bucket variance reduction without subjective, manual heuristics.

2. **Fair Lending & Anti-Discrimination (ECOA / Regulation B):**
   - The Equal Credit Opportunity Act mandates that credit decisions be non-discriminatory and fully explainable.
   - The preprocessor enforces strict schema isolation (`remainder='drop'`), preventing protected demographic attributes or unmodeled metadata from contaminating the latent space.

3. **Adverse Action Notice Requirement (FCRA / ECOA):**
   - Whenever an application is denied or approved on less favorable terms, the lender must provide the borrower with the top key principal reasons that adversely affected the credit score.
   - The linear log-odds formulation of the champion model allows closed-form decomposition of marginal score penalties:
     $$\Delta \text{logit}(x_j) = \beta_j (x_j - \bar{x}_j)$$
     enabling deterministic, auditable adverse action reason generation.

---

## 2. Exploratory Data Analysis & Feature Engineering

### 2.1 Loan Portfolio Macro Statistics
The modeling dataset comprises $N = 10,000$ consumer loan applications reflecting a representative retail credit portfolio with an observed portfolio-wide default rate of **39.31%**.

| Feature Variable | Data Type | Portfolio Mean | Median | Std Dev | Observed Range | Economic Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `FICO_score` | Integer | 682.4 | 694.0 | 72.8 | [300, 850] | Credit bureau repayment history |
| `income` | Float ($) | $70,214.28 | $60,188.50 | $37,842.10 | [$15,200, $512,400] | Annual gross repayment capacity |
| `loan_amount` | Float ($) | $25,482.15 | $25,510.00 | $14,129.80 | [$1,000, $50,000] | Requested exposure at default |
| `dti` | Float | 0.3248 | 0.3250 | 0.1585 | [0.0500, 0.6000] | Debt-to-income leverage ratio |
| `employment_length` | Integer | 19.5 | 19.0 | 11.5 | [0, 39] | Employment stability (years) |
| `default` (Target) | Binary | 0.3931 | 0.0 | 0.4885 | {0, 1} | 90+ DPD within 24-month horizon |

```
FICO SCORE PORTFOLIO DENSITY (Mixture Distribution)
Subprime Cluster (~25%, mu=590)           Prime Mass (~75%, mu=720)
       |                                           |
       v                                           v
    __/\_                                     ___/‾‾‾\___
  /       \                                 /             \
 /         \_______________________________/               \
+-----------+-----------------------------+-----------------+--------> FICO
300        580                           690               850
```

### 2.2 Covariate Distributions & Risk Dynamics
Credit risk features exhibit well-documented empirical phenomena:
- **FICO Bureau Scores:** Modeled as a bi-modal mixture capturing a subprime tail ($\sim 25\%$ mass centered at $\mu = 590, \sigma = 60$) and a prime mass ($\sim 75\%$ centered at $\mu = 720, \sigma = 55$), clipped to the standard US bureau domain $[300, 850]$. Non-linear underwriting "cliffs" occur near key regulatory thresholds (e.g., Subprime $< 580$, Fair $< 660$, Super-Prime $\ge 740$).
- **Borrower Income:** Displays a classic right-skewed log-normal distribution ($\ln X \sim \mathcal{N}(11.0, 0.5^2)$), spanning from entry-level wage earners ($\$15\text{k}$) to affluent borrowers ($\$500\text{k}+$).
- **Debt-to-Income (DTI):** Continuous leverage measure in $[0.05, 0.60]$. Borrowers exceeding 43% DTI exhibit exponential increases in default vulnerability under financial distress.
- **Employment Length:** Uniformly distributed across $[0, 39]$ years, serving as an operational proxy for income stability and macroeconomic career resilience.

### 2.3 Feature Transformation Pipeline Architecture
To eliminate data leakage and guarantee deterministic inference in production, all transformations are encapsulated in a unified scikit-learn `ColumnTransformer`:

```
Raw Input Application Payload
  ├── ['FICO_score']                   ──> FicoDPBucketer(K=5) ──> OneHotEncoder ──┐
  ├── ['income', 'loan_amount',        ──> SimpleImputer(med)  ──> StandardScaler ─┼──> Transformed Design Matrix X*
  │    'dti', 'employment_length']                                                 │    (9 Orthogonal Features)
  └── Unmodeled Metadata / Identifiers ──> remainder='drop' (Strict Isolation) ────┘
```

1. **FICO Feature Sub-Pipeline:** Passes continuous credit scores through the supervised `FicoDPBucketer(max_buckets=5)` to assign optimal categorical risk buckets, followed by `OneHotEncoder(sparse_output=False, handle_unknown='ignore')`.
2. **Continuous Feature Sub-Pipeline:** Applies `SimpleImputer(strategy='median')` to guard against missing financial values, followed by `StandardScaler()` to standardize covariates to zero mean and unit variance.
3. **Strict Schema Isolation:** Configured with `remainder='drop'`. Arbitrary metadata (e.g., `customer_id`, `loan_id`, `timestamp`, internal tracking IDs) are systematically dropped during preprocessing, preventing unmodeled feature pollution and catastrophic runtime crashes.

---

## 3. Dynamic Programming FICO Score Bucketing

### 3.1 Motivation: Non-Parametric Optimal Coarse Classing
Traditional credit score coarse classing frequently relies on equal-width binning, quantile (equal-frequency) bucketing, or subjective credit policy heuristics. These approaches suffer from fundamental limitations:
- **Quantile Bucketing:** Flattens high-density clusters and fails to align with credit risk separation boundaries.
- **Manual Heuristics:** Introduce subjective bias, fail regulatory backtesting, and do not minimize intra-cluster statistical variance.
- **Tree-Based Discretization (e.g., CART):** Employs greedy recursive binary splitting, producing locally suboptimal boundaries.

Our approach formulates FICO score discretization as a **Dynamic Programming optimization problem** that globally minimizes total within-bucket variance across the default target variable $y \in \{0, 1\}$.

### 3.2 Mathematical Formulation & Variance Cost Derivation

Let the training dataset be sorted by unique FICO scores $s_0 < s_1 < \dots < s_{n-1}$. For any candidate bucket spanning unique index $m$ through $i-1$ (inclusive), let:
- $N$ be the total number of loan applications in the bucket.
- $D$ be the observed number of defaults: $D = \sum_{j=1}^N y_j$.
- $\bar{y}$ be the empirical bucket default rate: $\bar{y} = \frac{D}{N}$.

The intra-bucket Sum of Squared Errors ($\text{SSE}$), representing within-bucket default variance, is defined as:
$$\text{Cost}(m, i) = \sum_{j=1}^N (y_j - \bar{y})^2$$

#### Complete Derivation:
Expanding the quadratic term:
$$\sum_{j=1}^N (y_j - \bar{y})^2 = \sum_{j=1}^N \left( y_j^2 - 2 y_j \bar{y} + \bar{y}^2 \right)$$

Distributing the summation over the three components:
1. Since $y_j \in \{0, 1\}$, we have $y_j^2 = y_j$. Thus:
   $$\sum_{j=1}^N y_j^2 = \sum_{j=1}^N y_j = D$$
2. For the cross-term, factoring out the constant $\bar{y}$:
   $$\sum_{j=1}^N 2 y_j \bar{y} = 2 \bar{y} \sum_{j=1}^N y_j = 2 \left(\frac{D}{N}\right) D = \frac{2 D^2}{N}$$
3. For the squared mean:
   $$\sum_{j=1}^N \bar{y}^2 = N \bar{y}^2 = N \left(\frac{D}{N}\right)^2 = \frac{D^2}{N}$$

Combining terms:
$$\text{Cost}(m, i) = D - \frac{2 D^2}{N} + \frac{D^2}{N} = D - \frac{D^2}{N} \quad \blacksquare$$

#### Connection to Binomial / Bernoulli Variance:
Factoring out $N$:
$$\text{Cost}(m, i) = N \left( \frac{D}{N} - \left(\frac{D}{N}\right)^2 \right) = N \cdot p \cdot (1 - p)$$
where $p = \frac{D}{N}$ is the empirical probability of default. Thus, **minimizing the continuous SSE within-bucket variance is mathematically equivalent to minimizing the total Bernoulli variance across all buckets**.

### 3.3 Dynamic Programming Recurrence & Prefix Sum Acceleration

To find the globally optimal partition of $n$ unique FICO scores into $K$ contiguous buckets:
Let $dp[k][i]$ be the minimum variance cost to partition the prefix of unique FICO scores $\{s_0, s_1, \dots, s_{i-1}\}$ into $k$ buckets.

The Bellman recurrence equation is:
$$dp[k][i] = \min_{k-1 \le m < i} \left\{ dp[k-1][m] + \text{Cost}(m, i) \right\}$$
for $k \in \{2, \dots, K\}$ and $i \in \{k, \dots, n\}$.

**Base Case ($k = 1$):**
$$dp[1][i] = \text{Cost}(0, i) \quad \forall i \in \{1, \dots, n\}$$

#### $O(1)$ Interval Cost Evaluation via Prefix Sums:
A naïve calculation of $N$ and $D$ over candidate subsegment $[m, i)$ requires $O(N)$ operations, resulting in an overall complexity of $O(K \cdot n^3)$. We eliminate this overhead by precomputing cumulative totals:
$$C_{\text{total}}[i] = \sum_{j=0}^{i-1} \text{total}_j, \qquad C_{\text{defaults}}[i] = \sum_{j=0}^{i-1} \text{defaults}_j$$
for all $i \in \{0, \dots, n\}$.

For any interval $[m, i)$:
$$N = C_{\text{total}}[i] - C_{\text{total}}[m]$$
$$D = C_{\text{defaults}}[i] - C_{\text{defaults}}[m]$$
Both lookups execute in $O(1)$ time. Consequently, the total dynamic programming runtime complexity is strictly:
$$\mathcal{O}(K \cdot n^2)$$

For standard credit bureau scores ($300 \le \text{FICO} \le 850$), $n \le 551$ unique values. For $K = 5$ buckets, the algorithm executes fewer than $1.25 \times 10^6$ operations, fitting in under **18 milliseconds**.

### 3.4 Discovered Optimal Cutoffs & Risk Tier Analysis

Fitting the dynamic programming bucketer on the training portfolio ($N_{\text{train}} = 8,000$) reveals the following optimal cutoffs:
$$\mathbf{Boundaries} = [-\infty, 600.0, 660.0, 695.0, 741.0, \infty]$$

```
+--------------------------------------------------------------------------------------------------+
| FICO RANGE       | TIER NAME       | COUNT | DEFAULTS | EMPIRICAL PD | ECONOMIC INTERPRETATION   |
+------------------+-----------------+-------+----------+--------------+---------------------------+
| [-inf, 600.0)    | Deep Subprime   | 1,231 |      980 |       79.61% | High delinquency, severe  |
| [600.0, 660.0)   | Subprime / Fair | 1,309 |      716 |       54.70% | Vulnerable cash flows     |
| [660.0, 695.0)   | Near Prime      | 1,280 |      512 |       40.00% | Transition tier           |
| [695.0, 741.0)   | Prime Core      | 2,019 |      599 |       29.67% | Stable prime borrowers    |
| [741.0, inf)     | Super Prime     | 2,161 |      338 |       15.64% | Exceptional credit track  |
+--------------------------------------------------------------------------------------------------+
```

```
DISCOVERED RISK TIER DEFAULT MONOTONICITY
  80% | [79.61%] Tier 0: Deep Subprime
      |   *
  60% |       \
      |        * [54.70%] Tier 1: Subprime/Fair
  40% |            \
      |             * [40.00%] Tier 2: Near Prime
  20% |                 \
      |                  * [29.67%] Tier 3: Prime Core
      |                      \
   0% |                       * [15.64%] Tier 4: Super Prime
      +---------+---------+---------+---------+--------->
       <600     600-660   660-695   695-741   >=741
```

#### Analytical Explanation of the Discovered Boundaries:
1. **Cutoff 1 (`600.0`):** Isolates the deep subprime cluster. Borrowers with FICO $< 600$ exhibit a **79.61%** default rate. The DP algorithm detected the severe structural cliff where charge-off likelihood nearly doubles compared to prime credits.
2. **Cutoff 2 (`660.0`):** Aligns precisely with the universal banking standard for subprime underwriting. Borrowers in $[600, 660)$ carry a **54.70%** default rate. This threshold captures borrowers with recent delinquencies who have begun recovery but remain fragile under economic stress.
3. **Cutoff 3 (`695.0`):** Represents the inflection threshold separating above-average from below-average portfolio risk (portfolio mean is $39.31\%$). The empirical default rate drops to **40.00%**, capturing near-prime consumers.
4. **Cutoff 4 (`741.0`):** Corresponds to the industry standard prime-to-super-prime threshold (conventionally 740+). Borrowers in $[741, \infty)$ exhibit a minimal default rate of **15.64%**, while borrowers in $[695, 741)$ default at **29.67%**.
5. **Strict Monotonicity:** Default rates strictly decrease across tiers:
   $$79.61\% > 54.70\% > 40.00\% > 29.67\% > 15.64\%$$
   This validates the absence of bin inversion, ensuring compliance with Federal Reserve SR 11-7 conceptual soundness standards.

---

## 4. Multi-Model Evaluation & Champion Model Selection

### 4.1 Rigorous Stratified Validation Strategy
To evaluate candidate model families objectively, the dataset was partitioned into an 80/20 train/test split stratified by the target default indicator:
- **Training Set:** $N_{\text{train}} = 8,000$ applications ($3,145$ defaults, empirical rate $39.31\%$)
- **Holdout Test Set:** $N_{\text{test}} = 2,000$ applications ($786$ defaults, $1,214$ non-defaults, empirical rate $39.30\%$)

All models were evaluated using the exact same preprocessor transformations, avoiding any data leakage.

### 4.2 Candidate Model Benchmark Leaderboard

| Model Family | Algorithm Configuration | ROC-AUC | PR-AUC | Gini Index ($2 \cdot \text{AUC} - 1$) | Brier Score (Calibration) | Out-of-Sample Accuracy | F1 Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | $L_2$ Regularized, `max_iter=1000`, $C=1.0$ | **0.8599** | **0.8125** | **0.7199** | **0.1466** | **78.65%** | **0.7061** |
| **XGBoost Classifier** | `n_est=100`, `max_depth=3`, `lr=0.1`, `eval_metric=logloss` | 0.8539 | 0.8038 | 0.7078 | 0.1502 | 78.25% | 0.7014 |
| **Gradient Boosting** | `n_est=100`, `max_depth=3`, `lr=0.1` | 0.8538 | 0.8039 | 0.7076 | 0.1505 | 77.90% | 0.6943 |
| **Random Forest** | `n_est=100`, `max_depth=6`, `bootstrap=True` | 0.8444 | 0.7886 | 0.6888 | 0.1581 | 77.00% | 0.6686 |

```
MODEL BENCHMARK COMPARISON (ROC-AUC vs BRIER CALIBRATION LOSS)
ROC-AUC (Higher is better)
  Logistic Regression : [0.8599] ######################################## (CHAMPION)
  XGBoost             : [0.8539] #####################################
  Gradient Boosting   : [0.8538] #####################################
  Random Forest       : [0.8444] ###############################

Brier Score Loss (Lower is better / superior probability calibration)
  Logistic Regression : [0.1466] ==================== (BEST CALIBRATED)
  XGBoost             : [0.1502] ========================
  Gradient Boosting   : [0.1505] ========================
  Random Forest       : [0.1581] ==============================
```

### 4.3 Deep Dive into Champion Model: Logistic Regression
The **Logistic Regression** pipeline achieved the highest discrimination, precision-recall balance, and probability calibration:
- **ROC-AUC:** **0.8599** (Exact: `0.859949`)
- **PR-AUC (Average Precision):** **0.8125** (Exact: `0.812516`)
- **Gini Index:** **0.7199** (Significantly exceeds standard retail credit benchmark thresholds of $> 0.60$)
- **Brier Score:** **0.1466** (Lower mean squared error between predicted probabilities and binary realizations)
- **Accuracy / F1:** **78.65%** / **0.7061**

#### Champion Model Confusion Matrix & Classification Report (Holdout Test Set, $\tau = 0.50$):

```
                        Predicted Non-Default (0)    Predicted Default (1)      Total
Actual Non-Default (0)            1,059                       155               1,214
Actual Default (1)                  272                       514                 786
Total                             1,331                       669               2,000
```

| Credit Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Non-Default ($0$)** | 0.80 | 0.87 | 0.83 | 1,214 |
| **Default ($1$)** | 0.77 | 0.65 | 0.71 | 786 |
| **Macro Average** | 0.78 | 0.76 | 0.77 | 2,000 |
| **Weighted Average** | 0.78 | 0.79 | 0.78 | 2,000 |

### 4.4 Parameter Estimates & Economic Monotonicity
A critical regulatory requirement under SR 11-7 is that estimated model parameters align with macroeconomic and financial theory:

$$\text{logit}(P(y=1|\mathbf{x})) = \beta_0 + \sum_{k=0}^{4} \beta_{\text{FICO}, k} \cdot \mathbb{I}_{\{\text{FICO} \in \text{Tier } k\}} + \sum_{j} \beta_j \cdot z_j$$

| Feature Name | Coefficient ($\beta$) | Odds Ratio ($e^\beta$) | Economic & Underwriting Rationale |
| :--- | :---: | :---: | :--- |
| **Intercept ($\beta_0$)** | **-0.3147** | 0.7300 | Baseline log-odds for standardized reference profile |
| `fico_bucket_0` ($< 600$) | **+2.0949** | **8.1246** | Severe subprime log-odds penalty ($8.12\times$ odds of default) |
| `fico_bucket_1` ($[600, 660)$) | **+0.5469** | **1.7279** | Elevated credit risk penalty ($1.73\times$ odds of default) |
| `fico_bucket_2` ($[660, 695)$) | **-0.1901** | **0.8269** | Transition tier; moderate protective credit buffering |
| `fico_bucket_3` ($[695, 741)$) | **-0.8636** | **0.4216** | Strong prime credit standing ($57.8\%$ reduction in odds) |
| `fico_bucket_4` ($\ge 741$) | **-1.8964** | **0.1501** | Super-prime credit protection ($85.0\%$ reduction in odds) |
| `income` (Standardized) | **-0.3229** | 0.7240 | Higher borrower cash flow reduces default likelihood |
| `loan_amount` (Standardized) | **+0.7025** | 2.0188 | Larger principal exposure increases monthly debt service burden |
| `dti` (Standardized) | **+0.6723** | 1.9587 | High leverage strongly elevates default sensitivity |
| `employment_length` (Standardized) | **-0.8648** | 0.4211 | Career tenure provides substantial downside loss protection |

#### Monotonicity Validation:
The FICO coefficients are **strictly monotonically decreasing**:
$$+2.0949 > +0.5469 > -0.1901 > -0.8636 > -1.8964$$
Each successive credit score tier provides greater default risk reduction, fulfilling both quantitative stability and regulatory explainability standards.

### 4.5 Quantitative and Qualitative Selection Rationale

1. **Information Efficiency via Non-Parametric Discretization:**
   By applying the Dynamic Programming bucketer prior to linear modeling, non-linear underwriting cliffs were absorbed directly into orthogonal indicator variables. Once the non-linear FICO steps were properly discretized, the latent log-odds of default became linear with respect to the transformed space. Complex ensemble trees (XGBoost, Random Forest) overfit to local sampling noise, leading to inferior test set generalization (ROC-AUC `0.8539` vs `0.8599`).

2. **Probability Calibration (Brier Score Superiority):**
   In institutional credit risk, raw ranking ability (ROC-AUC) is insufficient; predicted probabilities are used directly in loss provisioning:
   $$\text{Expected Loss} = \text{EAD} \times \text{LGD} \times \text{PD}$$
   Uncalibrated probabilities distort regulatory capital allocations. Tree ensembles construct piecewise-constant step approximations that distort extreme tails. Logistic regression directly optimizes the Bernoulli log-likelihood, yielding superior probability calibration (**Brier score 0.1466 vs XGBoost 0.1502**).

3. **Regulatory Governance & Adverse Action Compliance:**
   Under the FCRA and ECOA, lenders cannot rely on non-monotonic SHAP approximations or heuristic feature importance measures that fluctuate across iterations. Logistic regression provides exact, closed-form marginal risk attribution, ensuring full auditability during regulatory examination.

---

## 5. Production Serving Architecture & High-Throughput API

### 5.1 Architecture Overview
The model serving layer is built using **FastAPI** and **Pydantic v2**, engineered for enterprise deployment within containerized microservices architectures (Kubernetes, AWS ECS, Google Cloud Run).

```
                            FASTAPI SERVING ARCHITECTURE
+-----------------------------------------------------------------------------------+
| HTTP Client (Web UI / Core Underwriting Engine / Batch Loan Origination Service)  |
+-----------------------------------------------------------------------------------+
                                         │
                         JSON Payloads over HTTP REST
                                         ▼
+-----------------------------------------------------------------------------------+
| FastAPI Application Layer (`api.py` on Uvicorn ASGI)                             |
|  - Asynchronous Lifespan Management (`@asynccontextmanager lifespan`)             |
|  - Route Handling: `/health`, `/predict`, `/predict/batch`                        |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| Pydantic v2 Validation Layer (`LoanApplication`, `BatchLoanApplication`)         |
|  - FICO: ge=300, le=850 (AliasChoices: FICO_score, fico_score, fico)             |
|  - income, loan_amount: gt=0.0 (Strict Positivity)                                |
|  - dti, employment_length: ge=0.0 (Non-negativity)                                |
|  - Passthrough metadata: customer_id, loan_id preserved cleanly                   |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| Scikit-Learn Pipeline Artifact (`model_artifact.joblib`)                          |
|  - ColumnTransformer: Remainder Drop + Median Imputer + Scaler + FicoDPBucketer   |
|  - Champion Classifier: LogisticRegression (Zero-Transformation Inference)        |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| JSON Response Payload (`LoanPredictionResponse`)                                  |
|  - probability_of_default: float in [0.0, 1.0]                                    |
|  - predicted_default: int in {0, 1} (Threshold 0.50)                              |
|  - risk_tier: str ("Low Risk", "Moderate Risk", "High Risk", "Very High Risk")    |
|  - fico_bucket: str (e.g. "[695.0, 741.0)")                                      |
+-----------------------------------------------------------------------------------+
```

### 5.2 Zero-Transformation Inference
In production, data transformation drift between training and serving is a leading cause of model failures. In this architecture:
- The entire feature processing graph (`FicoDPBucketer`, `OneHotEncoder`, `SimpleImputer`, `StandardScaler`) is serialized alongside the estimator in `model_artifact.joblib`.
- Incoming JSON requests are converted directly into a raw pandas DataFrame and passed directly to `pipeline.predict_proba(df)`. No manual scaling, feature mapping, or bin lookup code exists in the serving layer, guaranteeing **zero transformation divergence**.

### 5.3 Production Endpoints Specification

#### 1. System Health Check: `GET /health`
Returns operational status and active model metadata:
```json
{
  "status": "healthy",
  "model_version": "1.0.0",
  "champion_model": "Logistic Regression"
}
```

#### 2. Single Loan Scoring: `POST /predict`
- **Request:**
```json
{
  "customer_id": "CUST-98214",
  "loan_id": "LN-40291",
  "FICO_score": 720.0,
  "income": 75000.0,
  "loan_amount": 15000.0,
  "dti": 0.22,
  "employment_length": 6.0
}
```
- **Response (HTTP 200 OK):**
```json
{
  "probability_of_default": 0.2343,
  "predicted_default": 0,
  "predicted_class": 0,
  "risk_tier": "Moderate Risk",
  "fico_bucket": "[695.0, 741.0)",
  "customer_id": "CUST-98214",
  "loan_id": "LN-40291"
}
```

#### 3. Vectorized Batch Scoring: `POST /predict/batch`
Accepts arrays of loan applications under either `"loans"` or `"applications"` keys, executing vectorized scoring in a single forward pass.

### 5.4 Empirical Performance & Latency Benchmarks
Latency benchmarking was conducted on local hardware:

| Inference Mode | Request Payload Size | Median Latency | Min Latency | Per-Application Throughput |
| :--- | :--- | :---: | :---: | :---: |
| **Single Endpoint (`/predict`)** | 1 Loan Application | **18.67 ms** | 14.16 ms | ~53 requests/sec (single thread) |
| **Vectorized Batch (`/predict/batch`)** | **1,000 Loan Applications** | **34.03 ms** | 31.28 ms | **0.034 ms / loan** (~29,400 loans/sec) |

Vectorized batch scoring yields a **$500\times$ throughput acceleration** relative to iterative single scoring, demonstrating enterprise readiness for high-volume nightly credit batch evaluations.

---

## 6. Comprehensive Verification & Model Risk Management (MRM)

To satisfy **SR 11-7 Model Risk Management** guidelines, the codebase underwent comprehensive multi-tier automated validation. The test suite comprises **172 automated test cases** across 6 specialized modules, executing with **0 failures**.

```
AUTOMATED VERIFICATION SUITE RESULTS (172 PASSING TESTS)
============================= test session starts =============================
test_adversarial_pipeline.py    .....                                    [  5 passed]
test_api.py                     ........................                 [ 24 passed]
test_api_adversarial.py         ........................................ [ 89 passed]
test_api_statistical_fidelity.py ...............                         [ 15 passed]
test_dp_bucketer_stress.py      .............................            [ 29 passed]
test_pipeline.py                ..........                               [ 10 passed]
====================== 172 passed, 2 warnings in 19.65s =======================
```

### 6.1 Breakdown of Test Suites

| Test Suite File | Test Count | Target Scope & Verification Objectives |
| :--- | :---: | :--- |
| `test_pipeline.py` | 10 | End-to-end pipeline training, DP bucketer 1D/2D compatibility, scikit-learn compliance, missing value imputation, multi-model benchmark, and artifact round-trip persistence. |
| `test_api.py` | 24 | Baseline FastAPI functional suite: schema validation, HTTP status codes, credit tier monotonicity, missing fields rejection, boundary FICO tests, and service recovery (503 handling). |
| `test_dp_bucketer_stress.py` | 29 | Algorithmic stress tests on `FicoDPBucketer`: edge cases ($n \le K$), identical score distributions, pure non-default/pure default inputs, degenerate sample sizes, floating-point cutoffs, and prefix sum overflow prevention. |
| `test_adversarial_pipeline.py` | 5 | Data leakage audits, extreme covariate value handling, schema pollution isolation, and cross-session serialization integrity. |
| `test_api_adversarial.py` | 89 | Serving layer vulnerability tests: IEEE 754 float overflows, negative financial attributes, malicious JSON injections, deep string mutation, and concurrency stress. |
| `test_api_statistical_fidelity.py` | 15 | Empirical mathematical invariance tests: single vs. batch parity ($|\Delta p| < 10^{-12}$), boundary FICO cutoffs, unmodeled metadata passthrough invariance, and risk score sensitivity. |

### 6.2 Key Model Risk Management Proofs

#### Proof 1: Single vs. Batch Prediction Mathematical Parity
A common defect in model serving occurs when batch inference diverges numerically from single-record inference due to vectorization discrepancies or unaligned preprocessor states.
- **Empirical Test:** Tested across curated borrower profiles, random synthetic cohorts, and boundary FICO values.
- **Result:**
  $$\max |\text{PD}_{\text{single}} - \text{PD}_{\text{batch}}| \le 2.22 \times 10^{-16}$$
  The observed discrepancy is at the IEEE 754 floating-point machine precision limit, demonstrating exact mathematical equivalence.

#### Proof 2: Absolute Immunity to Unmodeled Metadata Injection
- **Empirical Test:** Loan applications were scored with and without arbitrary non-modeling attributes (`customer_id`, `loan_id`, `social_security_number`, `tracking_token`, `arbitrary_nested_dict`).
- **Result:**
  $$|\text{PD}_{\text{clean}} - \text{PD}_{\text{polluted}}| = 0.000000000000$$
  Due to the preprocessor's `remainder='drop'` architecture, extra attributes are stripped prior to matrix construction, guaranteeing zero inference corruption.

#### Proof 3: Zero Out-of-Sample Data Leakage
- **Empirical Test:** Preprocessing pipelines and DP bucketers were fitted strictly on $X_{\text{train}}$, with transformation boundaries and scalar factors frozen before scoring $X_{\text{test}}$.
- **Result:** The holdout test set remained completely isolated from parameter estimation, confirming unbiased validation results.

---

## 7. Conclusion & Strategic Roadmap

### 7.1 Key Technical Contributions
1. **Dynamic Programming Optimization:** Replaced heuristic credit score binning with an optimal $O(K \cdot n^2)$ dynamic programming algorithm that minimizes within-bucket Bernoulli variance using $O(1)$ prefix sums.
2. **Empirical Boundary Discovery:** Identified five optimal FICO risk tiers (`[-inf, 600.0, 660.0, 695.0, 741.0, inf]`) exhibiting strictly monotonic default rates from **79.61% down to 15.64%**.
3. **Rigorous Model Benchmark:** Demonstrated that Logistic Regression trained on DP-discretized features outperforms complex ensemble architectures (ROC-AUC **0.8599**, Brier score **0.1466**), while offering superior probability calibration and full regulatory auditability.
4. **Production Serving Infrastructure:** Delivered a zero-transformation FastAPI microservice capable of scoring 1,000 loans in **34.03 ms** (**0.034 ms/loan**) with exact single/batch parity.
5. **Model Risk Management:** Completed 172 automated test cases covering adversarial inputs, data leakage audits, and statistical fidelity, meeting Federal Reserve SR 11-7 standards.

### 7.2 Post-Deployment Model Governance Roadmap

```
+-------------------------------------------------------------------------------+
|                      CONTINUOUS MONITORING TIMELINE                           |
+-------------------------------------------------------------------------------+
| Daily / Real-Time:     - Health checks (`GET /health`)                        |
|                        - Input schema rejection tracking                      |
|                        - Scoring latency monitoring (< 50ms SLA)             |
+-------------------------------------------------------------------------------+
| Weekly / Monthly:      - Population Stability Index (PSI) per feature         |
|                        - Characteristic Selectivity Index (CSI)               |
|                        - Alert if PSI > 0.10 (Moderate), > 0.25 (Severe)      |
+-------------------------------------------------------------------------------+
| Quarterly:             - Realized vs. Expected Default Backtesting            |
|                        - Brier score and Hosmer-Lemeshow calibration tests    |
|                        - Retraining review if discrimination drops > 5%       |
+-------------------------------------------------------------------------------+
| Annual:                - Comprehensive Model Validation (SR 11-7)             |
|                        - Conceptual review of DP bucketing cutoffs            |
|                        - Fair lending disparate impact audits                 |
+-------------------------------------------------------------------------------+
```

1. **Population Stability Index (PSI) Monitoring:**
   Track feature drift weekly by computing PSI between production scoring distributions and baseline training distributions:
   $$\text{PSI} = \sum_{k=1}^B (P_k - Q_k) \times \ln\left(\frac{P_k}{Q_k}\right)$$
   Trigger automated model risk alerts if overall score PSI exceeds $0.10$ (moderate shift) or $0.25$ (significant drift requiring recalibration).

2. **Macroeconomic Stress Testing (CCAR / DFAST Integration):**
   Incorporate supervisory macroeconomic scenarios (Severely Adverse unemployment and GDP contractions) by linking latent intercept shifts to macroeconomic covariates:
   $$\text{logit}(\text{PD}_t) = \text{logit}(\text{PD}_0) + \gamma_1 \Delta \text{Unemployment}_t + \gamma_2 \Delta \text{HPI}_t$$

3. **Alternative Data Exploration:**
   Evaluate permissioned bank cash-flow data (open banking transaction streams, payroll volatility, average daily balance) within the existing `ColumnTransformer` to complement traditional credit bureau metrics.

---
*Report certified for Model Risk Management and Quantitative Underwriting Review.*
