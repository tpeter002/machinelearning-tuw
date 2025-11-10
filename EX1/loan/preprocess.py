from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def prepare_loan_train_test(
    df: pd.DataFrame,
    scale: bool = True,
    *,
    target_col: str = "grade",
    id_col: str = "ID",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Select features and target
    feature_cols = [c for c in df.columns if c not in [target_col, id_col]]
    X_all = df[feature_cols]
    y_all = df[target_col].astype(str)

    # Train/validation split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, stratify=y_all, random_state=random_state
    )
    if scale:
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    return X_train, X_test, y_train, y_test


def preprocess_loan_features(
    df: pd.DataFrame,
    *,
    exclude_cols: Optional[Tuple[str, ...]] = ("grade", "ID"),
) -> pd.DataFrame:
    out = df.copy()

    def _norm_str(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip().str.lower().replace({"nan": pd.NA})

    # term: "36 months" / "60 months" -> binary (60m -> 1, 36m -> 0)
    if "term" in out.columns:
        s = _norm_str(out["term"])
        months = pd.to_numeric(s.str.extract(r"(\d+)", expand=False), errors="coerce")
        out["term"] = (months >= 60).fillna(0).astype(int)

    # emp_length: "< 1 year" -> 0, "10+ years" -> 10, NaN stays NaN
    if "emp_length" in out.columns:
        s = _norm_str(out["emp_length"])
        years = pd.to_numeric(s.str.extract(r"(\d+)", expand=False), errors="coerce")
        lt_one_mask = s.str.contains(r"<\s*1", na=False)
        years = years.mask(lt_one_mask, 0)
        out["emp_length"] = years

    # home_ownership: map to integers (unseen -> -1)
    if "home_ownership" in out.columns:
        s = _norm_str(out["home_ownership"])
        mapping = {"mortgage": 0, "rent": 1, "own": 2}
        out["home_ownership"] = s.map(mapping).fillna(-1).astype(int)

    # verification_status: map to integers (unseen -> -1)
    if "verification_status" in out.columns:
        s = _norm_str(out["verification_status"])
        mapping = {"not verified": 0, "source verified": 1, "verified": 2}
        out["verification_status"] = s.map(mapping).fillna(-1).astype(int)

    # loan_status: map to integers (unseen -> -1)
    if "loan_status" in out.columns:
        s = _norm_str(out["loan_status"])
        mapping = {"current": 0, "fully paid": 1, "charged off": 2}
        out["loan_status"] = s.map(mapping).fillna(-1).astype(int)

    # pymnt_plan: "y"/"n" -> 1/0
    if "pymnt_plan" in out.columns:
        s = _norm_str(out["pymnt_plan"])
        mapping = {"y": 1, "n": 0}
        out["pymnt_plan"] = s.map(mapping).fillna(0).astype(int)

    # purpose: encode to stable category codes
    if "purpose" in out.columns:
        s = _norm_str(out["purpose"])
        cat = pd.Categorical(s)
        out["purpose"] = pd.Series(cat.codes, index=out.index).astype(int)

    # addr_state: encode to category codes
    if "addr_state" in out.columns:
        s = _norm_str(out["addr_state"])
        cat = pd.Categorical(s)
        out["addr_state"] = pd.Series(cat.codes, index=out.index).astype(int)

    # initial_list_status: "w"/"f" -> 1/0 (else -1)
    if "initial_list_status" in out.columns:
        s = _norm_str(out["initial_list_status"])
        mapping = {"w": 1, "f": 0}
        out["initial_list_status"] = s.map(mapping).fillna(-1).astype(int)

    # application_type: joint -> 1, individual -> 0 (else -1)
    if "application_type" in out.columns:
        s = _norm_str(out["application_type"])
        mapping = {"individual": 0, "joint type": 1, "joint app": 1, "joint": 1}
        out["application_type"] = s.map(mapping).fillna(-1).astype(int)

    # disbursement_method: directpay=1, cash=0 (else -1)
    if "disbursement_method" in out.columns:
        s = _norm_str(out["disbursement_method"])
        mapping = {"cash": 0, "directpay": 1}
        out["disbursement_method"] = s.map(mapping).fillna(-1).astype(int)

    # Convert common percentage-like columns if present
    def _coerce_percent(col: str):
        if col in out.columns and pd.api.types.is_object_dtype(out[col]):
            s = (
                out[col]
                .astype(str)
                .str.strip()
                .str.replace("%", "", regex=False)
                .str.replace(",", "", regex=False)
            )
            out[col] = pd.to_numeric(s, errors="coerce")

    for perc_col in ("int_rate", "revol_util"):
        _coerce_percent(perc_col)

    if "fico_range_low" in out.columns and "fico_range_high" in out.columns:
        out["fico_avg"] = (out["fico_range_low"] + out["fico_range_high"]) / 2.0
        out.drop(columns=["fico_range_low", "fico_range_high"], inplace=True)

    for col in out.select_dtypes(include=["object"]).columns.tolist():
        if exclude_cols and col in exclude_cols:
            continue
        s = (
            out[col]
            .astype(str)
            .str.strip()
            .str.replace("%", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        coerced = pd.to_numeric(s, errors="coerce")
        if coerced.notna().any() and coerced.isna().sum() < len(coerced):
            out[col] = coerced
        else:
            cat = pd.Categorical(_norm_str(out[col]))
            out[col] = pd.Series(cat.codes, index=out.index).astype(int)

    manual_drop = [
        "funded_amnt",
        "funded_amnt_inv",
        "out_prncp_inv",
        "total_pymnt_inv",
        "num_sats",
        "num_rev_tl_bal_gt_0",
        "collection_recovery_fee",
    ]
    keep = [c for c in manual_drop if c in out.columns]
    if keep:
        out.drop(columns=keep, inplace=True, errors="ignore")

    if "policy_code" in out.columns and out["policy_code"].nunique() <= 1:
        out.drop(columns=["policy_code"], inplace=True)

    num = out.drop(columns=list(exclude_cols), errors="ignore").select_dtypes(
        include=[np.number]
    )
    if num.shape[1] > 1:
        corr = num.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if (upper[c] >= 0.999).any()]
        if to_drop:
            out.drop(columns=to_drop, inplace=True, errors="ignore")

    return out
