# SHA-XFD development prototype

SHA-XFD is a local Streamlit prototype for exploring a claims-data workflow. It
uses **synthetic data only**. Its labels, model scores, and holdout metrics are
not evidence of real-world fraud-detection accuracy and must not be used to
approve, deny, or investigate real claims.

## Windows development

Use Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Configure at least one development user in the current PowerShell session. Use
a unique local password and never commit it:

```powershell
$users = @{
  developer = @{
    password = "jacob@-2026"
    role = "System Administrator"
    name = "Local Developer"
  }
} | ConvertTo-Json -Compress
$env:SHA_XFD_USERS = $users
```

Generate the reproducible development dataset, train the model, then start the
app:

```powershell
python generate_dataset.py --rows 10000 --seed 42
python train_model.py
python -m streamlit run app.py
```

Then open `http://localhost:8501` in your browser and sign in with the
development user you configured above.

The dataset and model are written under the project directory, independent of
the shell's current working directory. Regenerate or replace the dataset and
retrain whenever its contents change; the app detects a model trained against
an older dataset and will not use it for scoring.

## Tests and lint

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

The CI workflow runs these checks on Windows with Python 3.12, including a
small model-training and prediction test.

## Docker

Docker Desktop can build and start the same demo stack. Set `SHA_XFD_USERS` in
PowerShell as above, then run:

```powershell
docker compose up --build
```

Open `http://localhost:8501`. The image builds a seeded synthetic dataset and
model. Compose stores the dataset, model artifact, and claim action log in
separate named volumes.

## Data and security limits

- Do not upload identifiable or otherwise sensitive health data to this demo.
- A real deployment needs documented data governance, least-privilege access,
  secure secret management, audit/retention controls, and security review.
- The login configuration is supplied with `SHA_XFD_USERS`; its passwords are
  plaintext environment secrets. Use a secrets manager and an appropriate
  identity provider for any shared or production deployment.
- Model artifacts use Python pickle. Only load artifacts generated and
  controlled by this project; never load an artifact from an untrusted source.
- The current random holdout split and synthetic feature patterns are suitable
  only for a workflow smoke test. A real model needs representative, approved
  labels, leakage-aware/time-based validation, subgroup and calibration checks,
  a human review process, and ongoing monitoring before any operational use.
