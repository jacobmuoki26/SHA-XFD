from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from claim_data import CLAIM_COLUMNS

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "dataset" / "SHA_Claims_10000_Dataset.csv"

DIAGNOSIS_CODES = {
    "Malaria": ("B54", 1_500),
    "Pneumonia": ("J18", 8_000),
    "Diabetes Checkup": ("E11", 2_000),
    "Maternity Delivery": ("O80", 15_000),
    "Appendectomy": ("K35", 35_000),
    "Hypertension": ("I10", 2_500),
}
GENUINE_NOTES = (
    "Fever and chills, malaria positive.",
    "Cough and chest pain, X-ray consolidation.",
    "Routine diabetic review, controlled.",
    "Mother delivered via SVD, baby healthy.",
)
FAKE_NOTES = (
    "Patient was seen and treated. Patient was seen and treated. Patient was seen and treated.",
    "Treatment given as per protocol. Treatment given as per protocol.",
)


def generate_dataset(rows: int = 10_000, seed: int = 42) -> pd.DataFrame:
    if rows < 10:
        raise ValueError("Generate at least 10 claims to allow a stratified train/test split.")
    if seed < 0:
        raise ValueError("Seed must be zero or greater.")

    rng = np.random.default_rng(seed)
    diagnoses = tuple(DIAGNOSIS_CODES)
    data: list[list[object]] = []
    for index in range(rows):
        diagnosis = str(rng.choice(diagnoses))
        icd10, base_cost = DIAGNOSIS_CODES[diagnosis]
        is_fraud = int(rng.choice([0, 1], p=[0.82, 0.18]))

        if not is_fraud:
            cost = rng.normal(base_cost, base_cost * 0.25)
            biometric = int(rng.choice([0, 1], p=[0.05, 0.95]))
            note = str(rng.choice(GENUINE_NOTES))
            fraud_type = "None"
        else:
            fraud_type = str(rng.choice(["Ghost Patient", "Upcoding"]))
            if fraud_type == "Ghost Patient":
                cost = rng.normal(base_cost * 1.8, base_cost * 0.5)
                biometric = 0
                note = str(rng.choice(FAKE_NOTES))
            else:
                cost = base_cost * rng.uniform(3.0, 5.5)
                biometric = int(rng.choice([0, 1], p=[0.3, 0.7]))
                note = str(rng.choice(GENUINE_NOTES))

        data.append(
            [
                f"SHA-ELD-{2_025_000_000 + index}",
                f"SHA-M-{rng.integers(1_000_000, 9_999_999)}",
                diagnosis,
                icd10,
                max(500, int(cost)),
                int(rng.choice([2, 3, 4, 5, 6])),
                int(rng.integers(1, 5)),
                biometric,
                note,
                fraud_type,
                is_fraud,
            ]
        )
    return pd.DataFrame(data, columns=CLAIM_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic SHA claims for development only."
    )
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    dataset = generate_dataset(args.rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(f"Wrote {len(dataset):,} synthetic claims to {args.output}")


if __name__ == "__main__":
    main()
