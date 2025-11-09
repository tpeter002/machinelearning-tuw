from pathlib import Path
from typing import Optional, Union
import pandas as pd
from scipy.io import arff


def _to_text(value):
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")
    return value


def load_preprocessed_speeddating(
    data_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    base_dir = Path(__file__).parent
    arff_path = (
        Path(data_path) if data_path is not None else base_dir / "speeddating.arff"
    )

    data, _meta = arff.loadarff(str(arff_path))
    dating = pd.DataFrame(data)

    # Decode object (bytes) columns to str
    object_cols = dating.select_dtypes(include=["object"]).columns.tolist()
    for col in object_cols:
        dating[col] = dating[col].map(_to_text)

    # Drop columns
    for col in ["has_null", "wave"]:
        if col in dating.columns:
            dating.drop(columns=[col], inplace=True)

    # Type conversions and new feature
    if "samerace" in dating.columns:
        dating["samerace"] = dating["samerace"].astype(int)

    if "age" in dating.columns and "age_o" in dating.columns:
        dating["age_diff"] = dating["age"] - dating["age_o"]
        for col in ["age", "age_o", "d_age", "d_d_age"]:
            if col in dating.columns:
                dating.drop(columns=[col], inplace=True)

    # Drop all columns starting with 'd_'
    d_cols = [c for c in dating.columns if c.startswith("d_")]
    if d_cols:
        dating.drop(columns=d_cols, inplace=True)

    # Drop 'field'
    if "field" in dating.columns:
        dating.drop(columns=["field"], inplace=True)

    # Binarize 'met'
    if "met" in dating.columns:
        dating["met"] = dating["met"].replace({3.0: 0, 5.0: 0, 6.0: 0, 7.0: 0, 8.0: 0})

    # Drop decisions
    for col in ["decision_o", "decision"]:
        if col in dating.columns:
            dating.drop(columns=[col], inplace=True)

    # Missing values
    if "expected_num_interested_in_me" in dating.columns:
        dating.drop(columns=["expected_num_interested_in_me"], inplace=True)

    return dating
