"""
Mutation Testing Verification Harness for Challenger 1 (Refined)
Empirically assesses whether tests/ suite catches critical logic mutations
across preprocessing.py, model_training.py, train.py, incremental_train.py, and fico_bucketing.py.
"""

import sys
import subprocess
from pathlib import Path

PYTHON_EXE = sys.executable

MUTATIONS = [
    # --- Preprocessing Mutations ---
    {
        "id": "MUT_PREP_01_MEDIAN_TO_MEAN",
        "file": "preprocessing.py",
        "desc": "Continuous Imputer: change median to mean",
        "original": "('imputer', SimpleImputer(strategy='median'))",
        "mutant": "('imputer', SimpleImputer(strategy='mean'))",
        "test_target": "tests/test_preprocessing.py::TestBuildPreprocessorContracts::test_continuous_median_imputation"
    },
    {
        "id": "MUT_PREP_02_NO_SCALER",
        "file": "preprocessing.py",
        "desc": "Continuous Scaler: disable standard scaling (remove scaler)",
        "original": "            ('scaler', StandardScaler())\n",
        "mutant": "",
        "test_target": "tests/test_preprocessing.py::TestBuildPreprocessorContracts::test_continuous_standard_scaling"
    },
    {
        "id": "MUT_PREP_03_NO_DROP_BINARY",
        "file": "preprocessing.py",
        "desc": "Categorical OHE: remove drop='if_binary'",
        "original": "drop='if_binary'",
        "mutant": "drop=None",
        "test_target": "tests/test_preprocessing.py::TestBuildPreprocessorContracts::test_categorical_binary_drop_if_binary"
    },
    {
        "id": "MUT_PREP_04_PASSTHROUGH",
        "file": "preprocessing.py",
        "desc": "ColumnTransformer: remainder='drop' changed to remainder='passthrough'",
        "original": "        remainder='drop'\n    )",
        "mutant": "        remainder='passthrough'\n    )",
        "test_target": "tests/test_preprocessing.py::TestBuildPreprocessorContracts::test_remainder_drop_discards_unmodeled_columns"
    },
    {
        "id": "MUT_PREP_05_MODE_TO_CONST",
        "file": "preprocessing.py",
        "desc": "Categorical Imputer: change most_frequent to constant (missing_value)",
        "original": "strategy='most_frequent'",
        "mutant": "strategy='constant', fill_value='UNKNOWN_CAT'",
        "test_target": "tests/test_preprocessing.py::TestBuildPreprocessorContracts::test_categorical_most_frequent_imputation"
    },

    # --- Model Training Mutations ---
    {
        "id": "MUT_MODEL_01_GINI",
        "file": "model_training.py",
        "desc": "Gini formula corruption: gini = auc instead of 2 * auc - 1",
        "original": "gini = 2.0 * auc - 1.0",
        "mutant": "gini = auc",
        "test_target": "tests/test_model_training.py::TestEvaluateModelContract::test_gini_mathematical_identity"
    },
    {
        "id": "MUT_MODEL_02_BRIER_OMIT",
        "file": "model_training.py",
        "desc": "evaluate_model: omit brier_score from returned dictionary",
        "original": "        'brier_score': brier,\n",
        "mutant": "",
        "test_target": "tests/test_model_training.py::TestEvaluateModelContract::test_evaluate_model_keys_and_bounds"
    },
    {
        "id": "MUT_MODEL_03_CHAMPION_INVERT",
        "file": "model_training.py",
        "desc": "Champion selection inverted: select minimum ROC-AUC",
        "original": "key=lambda k: (benchmark_results[k]['roc_auc'], -benchmark_results[k]['brier_score'])",
        "mutant": "key=lambda k: (-benchmark_results[k]['roc_auc'], benchmark_results[k]['brier_score'])",
        "test_target": "tests/test_model_training.py"
    },
    {
        "id": "MUT_MODEL_04_CARDINALITY",
        "file": "model_training.py",
        "desc": "High cardinality filter disabled: threshold 100 instead of 20",
        "original": "if X[col].nunique() < 20:",
        "mutant": "if X[col].nunique() < 100:",
        "test_target": "tests/test_model_training.py::TestTrainAndEvaluateContracts::test_high_cardinality_filtering"
    },
    {
        "id": "MUT_MODEL_05_METADATA_KEY",
        "file": "model_training.py",
        "desc": "Metadata serialization: omit champion_model key",
        "original": "            'champion_model': champion_name,\n",
        "mutant": "",
        "test_target": "tests/test_model_training.py::TestTrainAndEvaluateContracts::test_dual_artifact_and_metadata_serialization"
    },

    # --- Train.py Mutations ---
    {
        "id": "MUT_TRAIN_01_SYNC_OFF",
        "file": "train.py",
        "desc": "Dual artifact sync: disable shutil.copyfile to model.joblib",
        "original": "        shutil.copyfile(output_artifact, alt_artifact)",
        "mutant": "        pass  # mutant disabled sync",
        "test_target": "tests/test_train.py::TestTrainPipelineExecution::test_dual_artifact_synchronization"
    },
    {
        "id": "MUT_TRAIN_02_FALLBACK_OFF",
        "file": "train.py",
        "desc": "Synthetic data fallback: comment out generation in train.py and model_training.py",
        "original": "        df = generate_loan_data(num_samples=10000, random_state=42)\n        df.to_csv(data_path, index=False)",
        "mutant": "        pass  # mutant disabled fallback in train.py",
        "test_target": "tests/test_train.py::TestTrainPipelineExecution::test_run_pipeline_synthetic_fallback"
    },
    {
        "id": "MUT_TRAIN_03_BOUNDS_INVERT",
        "file": "train.py",
        "desc": "Smoke test assertion: invert probability bounds check to assert default_prob > 1.0",
        "original": "assert 0.0 <= default_prob <= 1.0",
        "mutant": "assert default_prob > 1.0",
        "test_target": "tests/test_train.py::TestTrainPipelineExecution::test_smoke_test_scoring_and_bounds"
    },

    # --- Incremental Train Mutations ---
    {
        "id": "MUT_INC_01_LC_RENAME",
        "file": "incremental_train.py",
        "desc": "Lending Club loader: rename loan_amount to wrong name loan_capital",
        "original": "df = df.rename(columns={'loan_amnt': 'loan_amount', 'loan_status': 'target'})",
        "mutant": "df = df.rename(columns={'loan_amnt': 'loan_capital', 'loan_status': 'target'})",
        "test_target": "tests/test_incremental_train.py::TestHarmonizationLoaders::test_lending_club_harmonization_contract"
    },
    {
        "id": "MUT_INC_02_CR_DROP",
        "file": "incremental_train.py",
        "desc": "Credit risk loader: do not drop person_emp_length",
        "original": "    df = df.drop(columns=['person_emp_length'])",
        "mutant": "    pass  # mutant keeps person_emp_length",
        "test_target": "tests/test_incremental_train.py::TestHarmonizationLoaders::test_credit_risk_harmonization_contract"
    },
    {
        "id": "MUT_INC_03_GER_STR",
        "file": "incremental_train.py",
        "desc": "German credit loader: purpose not converted to string",
        "original": "    df['purpose'] = df['purpose'].astype(str)",
        "mutant": "    pass  # mutant leaves purpose as int",
        "test_target": "tests/test_incremental_train.py::TestHarmonizationLoaders::test_german_harmonization_contract"
    },
    {
        "id": "MUT_INC_04_TREE_COUNT",
        "file": "incremental_train.py",
        "desc": "Booster continuation: remove xgb_model continuation so tree count stays 100",
        "original": "    xgb_clf.fit(X2, y2, xgb_model=xgb_clf.get_booster())\n    \n    # 5. Update on Dataset 3 (German Credit)\n    print(\"\\n--- Training Step 3: Updating with German Credit Data ---\")\n    X3 = preprocessor.transform(df_ger.drop(columns=['target']))\n    y3 = df_ger['target']\n    xgb_clf.fit(X3, y3, xgb_model=xgb_clf.get_booster())",
        "mutant": "    xgb_clf.fit(X2, y2)\n    \n    # 5. Update on Dataset 3 (German Credit)\n    print(\"\\n--- Training Step 3: Updating with German Credit Data ---\")\n    X3 = preprocessor.transform(df_ger.drop(columns=['target']))\n    y3 = df_ger['target']\n    xgb_clf.fit(X3, y3)",
        "test_target": "tests/test_incremental_train.py::TestIncrementalMasterModelArtifact::test_run_incremental_training_execution_and_persistence"
    },

    # --- FICO DP Bucketer Mutations ---
    {
        "id": "MUT_FICO_01_BOUNDARIES",
        "file": "fico_bucketing.py",
        "desc": "FICO Bucketer: reverse boundaries so monotonicity is violated",
        "original": "        self.boundaries_ = boundaries",
        "mutant": "        self.boundaries_ = list(reversed(boundaries))",
        "test_target": "tests/test_preprocessing.py::TestFicoDPBucketerContracts::test_boundary_determination_and_sse_minimization"
    },
    {
        "id": "MUT_FICO_02_ALL_ZERO",
        "file": "fico_bucketing.py",
        "desc": "FICO Bucketer: transform returns all zeros regardless of bin",
        "original": "        return np.asarray(buckets).reshape(-1, 1)",
        "mutant": "        return np.zeros_like(np.asarray(buckets).reshape(-1, 1))",
        "test_target": "tests/test_preprocessing.py"
    }
]

def run_mutation_audit():
    print("=" * 90)
    print("EMPIRICAL MUTATION TESTING AUDIT (REFINED) — CHALLENGER 1")
    print(f"Python Interpreter: {PYTHON_EXE}")
    print(f"Total Mutations to Test: {len(MUTATIONS)}")
    print("=" * 90)

    target_files = set(m["file"] for m in MUTATIONS)
    backups = {}
    for fname in target_files:
        p = Path(fname)
        backups[fname] = p.read_text(encoding="utf-8")

    results = []

    try:
        for m in MUTATIONS:
            m_id = m["id"]
            fname = m["file"]
            desc = m["desc"]
            orig_snippet = m["original"]
            mut_snippet = m["mutant"]
            target_test = m["test_target"]

            file_path = Path(fname)
            current_content = file_path.read_text(encoding="utf-8")

            if orig_snippet not in current_content:
                print(f"[ERROR] {m_id}: Target snippet not found in {fname}! Skipping.")
                results.append({
                    "id": m_id,
                    "desc": desc,
                    "status": "ERROR_SNIPPET_NOT_FOUND",
                    "killed": False,
                    "output": "Snippet not found"
                })
                continue

            mutated_content = current_content.replace(orig_snippet, mut_snippet, 1)
            file_path.write_text(mutated_content, encoding="utf-8")

            cmd = [PYTHON_EXE, "-m", "pytest", target_test, "-q", "--tb=short"]
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                stdout = proc.stdout + proc.stderr
                if proc.returncode != 0:
                    status = "KILLED"
                    killed = True
                else:
                    status = "SURVIVED"
                    killed = False

                results.append({
                    "id": m_id,
                    "desc": desc,
                    "target_test": target_test,
                    "status": status,
                    "killed": killed,
                    "output": stdout.strip().splitlines()[-1] if stdout.strip() else "No output"
                })
                print(f"[{status:8s}] {m_id:28s} | {desc:55s}")

            except subprocess.TimeoutExpired:
                status = "KILLED (TIMEOUT)"
                results.append({
                    "id": m_id,
                    "desc": desc,
                    "target_test": target_test,
                    "status": status,
                    "killed": True,
                    "output": "Timeout > 60s"
                })
                print(f"[{status:8s}] {m_id:28s} | {desc:55s} | Timeout")

            finally:
                file_path.write_text(backups[fname], encoding="utf-8")

    finally:
        for fname, original_content in backups.items():
            Path(fname).write_text(original_content, encoding="utf-8")
        print("=" * 90)
        print("ALL SOURCE FILES RESTORED TO PRISTINE CONDITION.")
        print("=" * 90)

    total = len(results)
    killed_count = sum(1 for r in results if r["killed"])
    survived_count = total - killed_count
    kill_rate = (killed_count / total) * 100 if total > 0 else 0

    print("\n" + "=" * 90)
    print("MUTATION TESTING SUMMARY REPORT")
    print(f"Total Mutants Evaluated: {total}")
    print(f"Mutants Killed:          {killed_count} ({kill_rate:.1f}%)")
    print(f"Mutants Survived:        {survived_count}")
    print("=" * 90)

    for r in results:
        badge = "[KILLED]" if r["killed"] else "[SURVIVED]"
        print(f"  {badge:10s} {r['id']:28s} -> {r['desc']}")

    return results

if __name__ == "__main__":
    run_mutation_audit()
