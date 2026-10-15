# InsightLab — Interactive Machine Learning Studio

> A polished, privacy-first Streamlit workspace for exploring CSV data, preparing a working copy, visualising patterns, training supervised machine-learning models, and testing individual predictions.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

## Project overview

InsightLab is a portfolio project built to demonstrate practical Python, data analysis, machine learning, UX, and deployment skills without depending on paid services, API keys, authentication, or external AI tools.

Upload a CSV to investigate its quality, make deliberate cleaning choices on a copy, build interactive Plotly charts, and train a scikit-learn classification or regression model. The app explains its data checks and model metrics using transparent, rule-based Python text rather than claiming AI-generated insight.

## Features

- **Professional product interface** — responsive dashboard, sidebar workflow, consistent cards, useful empty states, feedback, and an in-app light/dark workspace appearance selector.
- **CSV Dataset Explorer** — upload validation, row/column dimensions, types, sample rows, descriptive summaries, missing values, duplicate rows, memory estimate, and column-level quality details.
- **Automatic analysis** — deterministic insights for missing data, duplicates, constant columns, high-cardinality labels, and strongly correlated numerical variables.
- **Non-destructive cleaning** — choose columns, remove exact duplicates, drop incomplete rows, fill numeric values with median/mean and labels with mode, or use a custom replacement. Download the active copy as CSV.
- **Visualisation Studio** — interactive histogram, bar chart, scatter plot, Pearson correlation heatmap, and box plot. Charts cap display rendering at a reproducible 5,000-row sample for responsiveness.
- **Machine Learning Studio** — classification (Logistic Regression, Decision Tree, Random Forest) and regression (Linear Regression, Random Forest Regressor).
- **Leakage-aware preprocessing** — target-missing rows are dropped; feature imputation, scaling, and one-hot encoding live inside a scikit-learn pipeline fitted only after the train/test split.
- **Evaluation** — classification accuracy, weighted precision/recall/F1, classification report, and confusion matrix; regression MAE, RMSE, and R². Metrics are explained in plain English.
- **Prediction Playground** — validated numeric/categorical feature form, predictions, and class probabilities when available. It is disabled until a model trains successfully.
- **Privacy and safety** — CSV-only uploads, 25 MB app-side limit, in-memory data handling, no user-supplied code execution, no secrets, no paid services, and no external API calls.

## Technology stack

| Area | Tools |
| --- | --- |
| Web application | Python, Streamlit |
| Data handling | Pandas, NumPy |
| Visualisation | Plotly |
| Machine learning | scikit-learn |
| Testing | pytest |
| Deployment | GitHub + Streamlit Community Cloud |

## Screenshots

Run the app locally and capture your portfolio screenshots in `docs/screenshots/` before publishing. Recommended views:

1. Home dashboard with the included sample loaded.
2. Dataset Explorer showing the quality table and missing-value chart.
3. Visualisation Studio showing a scatter plot or correlation heatmap.
4. Machine Learning Studio showing metrics and a confusion matrix.

This repository intentionally does not include fabricated product screenshots. The `docs/screenshots/README.md` file lists suggested filenames for images you capture from the running application.

## Quick start on Windows 11

### Prerequisites

- Install **Python 3.10, 3.11, or 3.12** from [python.org](https://www.python.org/downloads/windows/). During setup, tick **Add python.exe to PATH**.
- Install [Visual Studio Code](https://code.visualstudio.com/) and its Python extension if you want an editor and integrated terminal.

### Create a virtual environment and install dependencies

Open PowerShell in the `InsightLab` folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell prevents activation for the current session, run this once and retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

If `py -3.12` is not installed, use the version you have, for example `py -3.11 -m venv .venv`.

### Run the app

```powershell
streamlit run app.py
```

Streamlit will print a local URL, normally `http://localhost:8501`. Open that URL in Microsoft Edge. Stop the app with `Ctrl+C` in the terminal.

## How to use it

1. Open **Use included sample** in the sidebar, or upload a `.csv` under 25 MB.
2. Visit **Dataset Explorer** to understand types, nulls, duplicates, summaries, and rule-based observations.
3. In **Data Cleaning**, choose columns and quality operations. This creates a new active copy; it never overwrites the original upload.
4. In **Visualisation Studio**, select chart types and variables appropriate to your data.
5. In **Machine Learning Studio**, choose a target, task, features, algorithm, and test-set size, then train.
6. Read the held-out metrics and inspect predicted versus actual values. Do not equate a good score with a reliable model.
7. Open **Prediction Playground** to submit one validated record to the latest trained model.
8. Use **Reset current analysis** to clear data and models from the current session.

## Testing

Run the automated tests from the project root after installing dependencies:

```powershell
pytest -q
```

The test suite covers CSV parsing helpers, profiling, rule-based insight signals, non-destructive cleaning, classification training, regression training, and manual prediction input validation.

For a quick manual smoke test, launch the app, load the included sample, create a cleaned copy, build every chart type, train a classifier with `Survived` as the target, and submit a prediction in the Prediction Playground.

## Architecture

```text
InsightLab/
├── app.py                  # Streamlit pages, navigation, and session state
├── src/
│   ├── analysis.py         # Dataset profiling and deterministic insights
│   ├── cleaning.py         # Copy-based cleaning operations
│   ├── config.py           # Paths and operational limits
│   ├── data_io.py          # Safe in-memory CSV input/output
│   ├── modeling.py         # Split-safe pipelines, training, metrics, prediction validation
│   └── ui.py               # Reusable visual presentation helpers
├── sample_data/            # Documented fictional practice dataset
├── tests/                  # Automated logic/workflow tests
├── .streamlit/config.toml  # Streamlit theme and upload settings
├── requirements.txt
└── .gitignore
```

### Model-training flow

```text
Active DataFrame
    → choose target + features
    → drop rows without target
    → train/test split
    → fit imputer + encoder/scaler on training rows only
    → fit selected estimator
    → evaluate on held-out test rows
    → retain successful pipeline for validated manual prediction
```

## Privacy and security

- User data stays in Streamlit's application/session memory unless the user explicitly downloads an export. InsightLab does not persist data to a database.
- CSV uploads are size-limited and parsed as data only. The project never executes file contents, formulas, or code from a dataset.
- The project has no API keys, passwords, payment flows, telemetry, user accounts, or external AI/ML services.
- A public deployment is still a public web app: do not upload confidential, regulated, personal, or sensitive data to a shared URL.

## Limitations and responsible use

- This is an educational analysis tool, not a substitute for domain review, privacy review, bias testing, or production MLOps.
- Evaluation uses one held-out split. Scores can be unstable on small or unrepresentative datasets and do not prove fairness, causal relationships, or future performance.
- Very large files are deliberately limited to 25 MB; charts sample at 5,000 rows for responsive rendering.
- The app supports standard tabular supervised learning only. It does not perform time-series validation, NLP, image modelling, hyperparameter search, persistence of trained models, authentication, or collaboration.
- Categorical prediction inputs are restricted to values observed during training to keep the form clear. The pipeline itself is configured to safely ignore unseen categories if used programmatically.

## Publish to GitHub

1. Sign in to GitHub and create a new empty repository, for example `insightlab-ml-studio`. Do **not** initialise it with a README because this project already has one.
2. In PowerShell at the `InsightLab` project folder, run the following. Replace `YOUR-USERNAME` with your GitHub username:

```powershell
git init
git add .
git commit -m "Initial commit: InsightLab ML Studio"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/insightlab-ml-studio.git
git push -u origin main
```

3. Refresh your GitHub repository page and add the screenshots described above if desired. Never commit `.env`, passwords, `secrets.toml`, personal datasets, or other sensitive files.

## Deploy for free with Streamlit Community Cloud

1. Ensure the repository is pushed to GitHub and `requirements.txt` is in the root of the repository.
2. Visit [share.streamlit.io](https://share.streamlit.io/), sign in with GitHub, and choose **Create app**.
3. Select the repository, the `main` branch, and set the main file path to `app.py`.
4. Choose a Python version supported by Community Cloud (Python 3.11 or 3.12 is a safe choice for the dependency ranges here), then deploy.
5. Community Cloud installs `requirements.txt`, builds the app, and presents a public `https://...streamlit.app` URL. Copy that URL into your GitHub repository description and portfolio.

No API key, payment method, database, or Streamlit secret is required for this application.

## License

Released under the [MIT License](LICENSE).

