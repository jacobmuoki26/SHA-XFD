from __future__ import annotations

import base64
import csv
import hmac
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from claim_data import build_features, dataset_sha256, validate_claims_dataset
from train_model import DATASET_PATH, MODEL_PATH, train

ROOT = Path(__file__).resolve().parent
ACTION_LOG_PATH = ROOT / "data" / "claim_actions.csv"
STYLESHEET_PATH = ROOT / "assets" / "app.css"
MEDICAL_BACKGROUND_PATH = ROOT / "assets" / "medical-claims-bg.svg"
MEDICAL_BACKGROUND = base64.b64encode(MEDICAL_BACKGROUND_PATH.read_bytes()).decode("ascii")
ROLES = {"Admin", "System Administrator", "Model Trainer"}
REVIEW_THRESHOLD = 0.4

st.set_page_config(
    page_title="SHA-XFD System",
    page_icon="🇰🇪",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_app_styles() -> None:
    stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8").replace(
        "__MEDICAL_BACKGROUND__",
        f"data:image/svg+xml;base64,{MEDICAL_BACKGROUND}",
    )
    st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="brand-header">
            <div class="brand-mark">+</div>
            <div>
                <div class="brand-name">SHA-XFD</div>
                <div class="brand-subtitle">Claims review and fraud detection</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def configured_users() -> dict[str, dict[str, str]]:
    raw_users = os.environ.get("SHA_XFD_USERS")
    if not raw_users:
        return {}

    try:
        users = json.loads(raw_users)
    except json.JSONDecodeError as error:
        raise ValueError("SHA_XFD_USERS must contain valid JSON.") from error
    if not isinstance(users, dict):
        raise ValueError("SHA_XFD_USERS must be a JSON object keyed by username.")

    for username, user in users.items():
        if (
            not isinstance(username, str)
            or not username.strip()
            or not isinstance(user, dict)
            or not isinstance(user.get("password"), str)
            or not user["password"]
            or not isinstance(user.get("role"), str)
            or user["role"] not in ROLES
            or not isinstance(user.get("name"), str)
            or not user["name"].strip()
        ):
            raise ValueError(
                "Each SHA_XFD_USERS entry needs a non-empty password, name, and a supported role."
            )
    return users


def authenticate(users: dict[str, dict[str, str]], username: str, password: str) -> bool:
    user = users.get(username)
    return user is not None and hmac.compare_digest(
        user["password"].encode("utf-8"), password.encode("utf-8")
    )


def load_model_bundle() -> tuple[dict[str, Any] | None, str | None]:
    if not MODEL_PATH.exists():
        return None, "Model not trained. Generate the dataset and train the model first."
    if not DATASET_PATH.exists():
        return None, "Claims dataset is missing; the model's dataset version cannot be verified."

    try:
        with MODEL_PATH.open("rb") as model_file:
            bundle = pickle.load(model_file)
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ImportError) as error:
        return None, f"Could not load the local model artifact: {error}"

    required_keys = {"model", "features", "diagnoses", "metrics", "dataset_sha256"}
    if (
        not isinstance(bundle, dict)
        or not required_keys <= bundle.keys()
        or not isinstance(bundle["features"], list)
        or not isinstance(bundle["diagnoses"], list)
        or not isinstance(bundle["metrics"], dict)
    ):
        return None, "Model artifact is invalid. Retrain the model to rebuild it."
    try:
        current_hash = dataset_sha256(DATASET_PATH)
    except OSError as error:
        return None, f"Could not verify the claims dataset: {error}"
    if bundle["dataset_sha256"] != current_hash:
        return None, "Dataset changed since the last training; retrain before using the model."
    return bundle, None


def read_claims_dataset() -> pd.DataFrame:
    data = pd.read_csv(DATASET_PATH, keep_default_na=False)
    validate_claims_dataset(data)
    return data


def score_claims(bundle: dict[str, Any], claims: pd.DataFrame) -> list[float]:
    known_diagnoses = {str(value) for value in bundle["diagnoses"]}
    claim_diagnoses = {str(value).strip() for value in claims["Diagnosis"].dropna()}
    unknown_diagnoses = sorted(claim_diagnoses - known_diagnoses)
    if unknown_diagnoses:
        values = ", ".join(unknown_diagnoses)
        raise ValueError(f"Model has not been trained on these diagnoses: {values}")

    features = build_features(claims).reindex(columns=bundle["features"], fill_value=0)
    return bundle["model"].predict_proba(features)[:, 1].tolist()


def login_page(users: dict[str, dict[str, str]], configuration_error: str | None) -> None:
    render_app_styles()
    st.title("Sign in")
    st.subheader("Social Health Authority Fraud Detection System")
    st.warning(
        "Development prototype: its synthetic-data scores are not validated for "
        "real claims or payment decisions."
    )
    if configuration_error:
        st.error(configuration_error)
    elif not users:
        st.info(
            "No users are configured. Set the SHA_XFD_USERS environment variable "
            "as described in README.md."
        )

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button(
            "Login",
            type="primary",
            disabled=not users or bool(configuration_error),
        )
        if submitted:
            if authenticate(users, username, password):
                user = users[username]
                st.session_state.logged_in = True
                st.session_state.user = username
                st.session_state.role = user["role"]
                st.session_state.name = user["name"]
                st.rerun()
            else:
                st.error("Invalid username or password")
    st.divider()
    st.caption("JJ AI TECHNOLOGIES-@2026")


def logout() -> None:
    for key in ("logged_in", "user", "role", "name"):
        st.session_state.pop(key, None)
    st.rerun()


def save_action(claim_id: str, action: str, username: str, risk_score: float) -> None:
    ACTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not ACTION_LOG_PATH.exists() or ACTION_LOG_PATH.stat().st_size == 0
    with ACTION_LOG_PATH.open("a", newline="", encoding="utf-8") as action_file:
        writer = csv.DictWriter(
            action_file,
            fieldnames=("timestamp_utc", "claim_id", "action", "risk_score", "username"),
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "claim_id": claim_id,
                "action": action,
                "risk_score": f"{risk_score:.6f}",
                "username": username,
            }
        )


def main_app(
    users: dict[str, dict[str, str]], bundle: dict[str, Any] | None, model_error: str | None
) -> None:
    render_app_styles()
    st.sidebar.title(f"👤 {st.session_state['name']}")
    st.sidebar.write(f"Role: {st.session_state['role']}")
    st.sidebar.write(f"User: {st.session_state['user']}")
    if st.sidebar.button("Logout"):
        logout()
    st.sidebar.divider()

    menu = ["Dashboard", "Check Single Claim", "Flagged Claims"]
    if st.session_state["role"] in {"System Administrator", "Model Trainer"}:
        menu.append("Retrain Model")
    if st.session_state["role"] == "System Administrator":
        menu.extend(["User Management", "System Logs"])
    choice = st.sidebar.radio("Pages", menu)

    if model_error:
        st.warning(model_error)

    if choice == "Dashboard":
        st.title(f"Dashboard - {st.session_state['role']}")
        st.write("Review the development dataset and model evaluation.")
        if DATASET_PATH.exists():
            try:
                data = read_claims_dataset()
            except (pd.errors.ParserError, UnicodeDecodeError, OSError, ValueError) as error:
                st.error(f"Could not read the claims dataset: {error}")
                return
            st.warning(
                "This dashboard uses synthetic development labels. Counts and "
                "scores are not real-world fraud estimates."
            )
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Synthetic Claims", f"{len(data):,}")
            col2.metric("Synthetic Fraud Labels", f"{data['IsFraud'].sum():,}")
            col3.metric("Synthetic Label Rate", f"{data['IsFraud'].mean() * 100:.1f}%")
            col4.metric(
                "Synthetic Label Amount",
                f"Ksh {data.loc[data['IsFraud'] == 1, 'TreatmentCost_KES'].sum():,}",
            )
            st.subheader("Synthetic labels by type")
            st.bar_chart(data.groupby("FraudType")["TreatmentCost_KES"].sum())
            st.subheader("Claims sample")
            st.dataframe(data.head(100), use_container_width=True)
        else:
            st.error("Dataset not found. Run generate_dataset.py first.")

        if bundle:
            st.subheader("Holdout evaluation (synthetic data)")
            st.json(bundle["metrics"])

    elif choice == "Check Single Claim":
        st.title("Check a Claim")
        st.caption(
            "Enter claim details to view a development-model risk score. "
            "The score is not a fraud determination."
        )
        col1, col2 = st.columns(2)
        with col1:
            cost = st.number_input(
                "Treatment cost (KES)",
                500,
                500_000,
                25_000,
                help="Enter the claimed treatment cost in Kenyan shillings.",
            )
            level = st.selectbox("Hospital level", [2, 3, 4, 5, 6])
            stay = st.slider("Length of stay (days)", 1, 365, 2)
        with col2:
            biometric = st.selectbox(
                "Biometric verified?",
                [1, 0],
                format_func=lambda value: "Yes" if value == 1 else "No",
            )
            diagnoses = sorted(bundle["diagnoses"]) if bundle else []
            diagnosis = st.selectbox("Diagnosis", diagnoses) if diagnoses else ""
            notes = st.text_area("Clinical Notes", "Fever and chills, malaria positive.")

        if st.button("CHECK FRAUD SCORE", type="primary"):
            if bundle:
                claim = pd.DataFrame(
                    [
                        {
                            "TreatmentCost_KES": cost,
                            "HospitalLevel": level,
                            "LengthOfStay_Days": stay,
                            "BiometricVerified": biometric,
                            "Diagnosis": diagnosis,
                            "ClinicalNotes": notes,
                        }
                    ]
                )
                probability = score_claims(bundle, claim)[0]
                st.metric("Model Risk Score", f"{probability * 100:.1f}%")
                if probability >= 0.75:
                    st.error(f"High risk - manual review recommended ({probability * 100:.0f}%)")
                elif probability >= REVIEW_THRESHOLD:
                    st.warning(f"Needs manual review ({probability * 100:.0f}%)")
                else:
                    st.success(f"Lower model risk ({probability * 100:.0f}%)")
                st.caption("A model score is not a fraud finding or payment decision.")
            else:
                st.error(model_error or "Model is unavailable.")

    elif choice == "Flagged Claims":
        st.title("Claims for Review")
        if bundle and DATASET_PATH.exists():
            try:
                data = read_claims_dataset()
            except (pd.errors.ParserError, UnicodeDecodeError, OSError, ValueError) as error:
                st.error(f"Could not read the claims dataset: {error}")
                return
            try:
                data["ModelRiskScore"] = score_claims(bundle, data)
            except ValueError as error:
                st.error(f"Could not score claims: {error}")
                return
            flagged = data[data["ModelRiskScore"] >= REVIEW_THRESHOLD].copy()
            flagged = flagged.sort_values(by="ModelRiskScore", ascending=False)
            st.caption(
                "Threshold 40%; synthetic development data only. A flag is a "
                "request for review, not a fraud determination."
            )
            st.dataframe(
                flagged.drop(columns=["IsFraud", "FraudType"]),
            )

            if st.session_state["role"] in {"Admin", "System Administrator"} and not flagged.empty:
                claim_id = st.selectbox("Select ClaimID to Action", flagged["ClaimID"])
                action = st.selectbox(
                    "Action",
                    ["Approve Payment", "Reject - Fraud", "Send for Investigation"],
                )
                risk_score = float(
                    flagged.loc[flagged["ClaimID"] == claim_id, "ModelRiskScore"].iloc[0]
                )
                if st.button("Submit Action"):
                    save_action(
                        claim_id,
                        action,
                        st.session_state["user"],
                        risk_score,
                    )
                    st.success(f"Action recorded for {claim_id}: {action}")
            st.download_button(
                "Download Claims for Review",
                flagged.drop(columns=["IsFraud", "FraudType"]).to_csv(index=False),
                "claims_for_review.csv",
            )
        else:
            st.error(model_error or "Dataset or current model is unavailable.")

    elif choice == "Retrain Model":
        st.title("Retrain Model")
        st.warning(
            "Only use synthetic or approved, de-identified development data. "
            "Training metrics on synthetic data do not establish real-world accuracy."
        )
        if DATASET_PATH.exists():
            try:
                data = read_claims_dataset()
            except (pd.errors.ParserError, UnicodeDecodeError, OSError, ValueError) as error:
                st.error(f"Could not read the claims dataset: {error}")
                return
            st.success(f"Dataset found: {len(data):,} claims")
            if "FraudType" in data:
                st.dataframe(data["FraudType"].value_counts())
        else:
            st.error("Dataset not found. Generate one before training.")

        uploaded = st.file_uploader(
            "Upload replacement synthetic development claims CSV", type=["csv"]
        )
        if uploaded and st.button("Validate and replace dataset"):
            try:
                uploaded_data = pd.read_csv(uploaded, keep_default_na=False)
                validate_claims_dataset(uploaded_data)
                DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = DATASET_PATH.with_suffix(".upload.tmp")
                try:
                    uploaded_data.to_csv(temporary_path, index=False)
                    os.replace(temporary_path, DATASET_PATH)
                finally:
                    if temporary_path.exists():
                        temporary_path.unlink()
                st.success(
                    "Dataset validated and replaced. The existing model is now "
                    "marked stale; retrain it before scoring claims."
                )
            except (pd.errors.ParserError, UnicodeDecodeError, ValueError, OSError) as error:
                st.error(f"Dataset was not replaced: {error}")

        if st.button("RETRAIN MODEL NOW", type="primary"):
            with st.spinner("Training model..."):
                try:
                    metrics = train()
                except (ImportError, OSError, RuntimeError, ValueError) as error:
                    st.error(f"Model training failed: {error}")
                else:
                    st.success("Model retrained.")
                    st.json(metrics)
                    st.rerun()

    elif choice == "User Management":
        st.title("👥 User Management")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Username": username, "Role": user["role"], "Name": user["name"]}
                    for username, user in users.items()
                ]
            ),
            use_container_width=True,
        )
        st.info("Configure users with the SHA_XFD_USERS environment variable.")

    elif choice == "System Logs":
        st.title("📝 Claim Action Log")
        if ACTION_LOG_PATH.exists():
            st.dataframe(pd.read_csv(ACTION_LOG_PATH), use_container_width=True)
        else:
            st.info("No claim actions have been recorded yet.")


def main() -> None:
    for key, default in (("logged_in", False), ("user", None), ("role", None), ("name", None)):
        if key not in st.session_state:
            st.session_state[key] = default

    try:
        users = configured_users()
        configuration_error = None
    except ValueError as error:
        users = {}
        configuration_error = str(error)

    if not st.session_state["logged_in"]:
        login_page(users, configuration_error)
        return

    bundle, model_error = load_model_bundle()
    main_app(users, bundle, model_error)


if __name__ == "__main__":
    main()
