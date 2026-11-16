from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

CLAIM_COLUMNS = (
    "ClaimID",
    "MemberID",
    "Diagnosis",
    "ICD10",
    "TreatmentCost_KES",
    "HospitalLevel",
    "LengthOfStay_Days",
    "BiometricVerified",
    "ClinicalNotes",
    "FraudType",
    "IsFraud",
)

DUPLICATE_NOTE = "Patient was seen and treated"


def validate_claims_dataset(data: pd.DataFrame) -> None:
    missing_columns = sorted(set(CLAIM_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    if data.empty:
        raise ValueError("The claims dataset is empty.")

    required_data = data.loc[:, list(CLAIM_COLUMNS)]
    null_columns = required_data.columns[required_data.isna().any()].tolist()
    if null_columns:
        raise ValueError(f"Required columns contain empty values: {', '.join(null_columns)}")

    for column in ("ClaimID", "MemberID", "Diagnosis", "ICD10", "FraudType"):
        text_values = data[column].astype("string")
        if text_values.str.strip().eq("" ).any():
            raise ValueError(f"{column} values cannot be empty.")
    if data["ClaimID"].astype(str).duplicated().any():
        raise ValueError("ClaimID values must be unique.")
    for column in (
        "TreatmentCost_KES",
        "HospitalLevel",
        "LengthOfStay_Days",
        "BiometricVerified",
        "IsFraud",
    ):
        numeric_values = pd.to_numeric(data[column], errors="coerce")
        values_array = numeric_values.to_numpy(dtype=float, na_value=np.nan)
        if pd.isna(values_array).any() or not np.isfinite(values_array).all():
            raise ValueError(f"{column} must contain finite numeric values.")

    if (pd.to_numeric(data["TreatmentCost_KES"], errors="coerce") <= 0).any():
        raise ValueError("TreatmentCost_KES must be greater than zero.")
    if (pd.to_numeric(data["LengthOfStay_Days"], errors="coerce") <= 0).any():
        raise ValueError("LengthOfStay_Days must be greater than zero.")
    hospital_levels = set(
        int(level)
        for level in pd.to_numeric(data["HospitalLevel"], errors="coerce").dropna().unique()
    )
    if not hospital_levels.issubset({2, 3, 4, 5, 6}):
        raise ValueError("HospitalLevel must contain levels 2, 3, 4, 5, or 6.")
    biometric_levels = set(
        int(level)
        for level in pd.to_numeric(data["BiometricVerified"], errors="coerce").dropna().unique()
    )
    if not biometric_levels.issubset({0, 1}):
        raise ValueError("BiometricVerified must contain only 0 or 1.")
    fraud_labels = pd.to_numeric(data["IsFraud"], errors="coerce").dropna()
    if not set(fraud_labels.unique()).issubset({0, 1}):
        raise ValueError("IsFraud must contain only 0 or 1.")
    if fraud_labels.nunique() != 2:
        raise ValueError("IsFraud must contain both classes (0 and 1) for training.")
    if not data["ClinicalNotes"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("ClinicalNotes must contain text.")


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Diagnosis",
        "TreatmentCost_KES",
        "HospitalLevel",
        "LengthOfStay_Days",
        "BiometricVerified",
        "ClinicalNotes",
    }
    missing_columns = sorted(required - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing feature columns: {', '.join(missing_columns)}")

    cost = pd.to_numeric(data["TreatmentCost_KES"], errors="raise")
    stay = pd.to_numeric(data["LengthOfStay_Days"], errors="raise")
    if (stay <= 0).any():
        raise ValueError("LengthOfStay_Days must be greater than zero.")

    features = pd.DataFrame(index=data.index)
    features["TreatmentCost_KES"] = cost
    features["HospitalLevel"] = pd.to_numeric(data["HospitalLevel"], errors="raise")
    features["LengthOfStay_Days"] = stay
    features["BiometricVerified"] = pd.to_numeric(data["BiometricVerified"], errors="raise")
    features["CostPerDay"] = cost / stay
    features["IsLevelMismatch"] = ((features["HospitalLevel"] <= 3) & (cost > 100_000)).astype(
        "int8"
    )
    notes = data["ClinicalNotes"].fillna("").astype(str)
    features["NotesLength"] = notes.str.len()
    features["IsDuplicateNote"] = notes.str.contains(DUPLICATE_NOTE, regex=False).astype("int8")
    diagnosis_features = pd.get_dummies(
        data["Diagnosis"].astype(str), prefix="Diagnosis", dtype="int8"
    )
    return pd.concat([features, diagnosis_features], axis=1)


def dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as dataset_file:
        for chunk in iter(lambda: dataset_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
