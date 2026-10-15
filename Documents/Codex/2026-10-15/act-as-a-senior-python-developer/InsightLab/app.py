"""InsightLab — an in-browser, privacy-first machine learning studio."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analysis import build_profile, generate_data_insights, get_column_groups
from src.cleaning import CleaningOptions, apply_cleaning
from src.config import MAX_CHART_ROWS, MAX_PREVIEW_ROWS, SAMPLE_DATA_PATH
from src.data_io import DatasetLoadError, dataframe_to_csv_bytes, load_csv_bytes
from src.modeling import (
    ModelTrainingError,
    TrainingResult,
    available_algorithms,
    build_prediction_frame,
    infer_task,
    metric_explanations,
    train_model,
)
from src.ui import apply_theme, card, empty_state, hero, section


st.set_page_config(
    page_title="InsightLab | Interactive Machine Learning Studio",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "InsightLab processes CSV data in its current Streamlit session."},
)

STATE_DEFAULTS: dict[str, object] = {
    "original_data": None,
    "active_data": None,
    "source_name": None,
    "data_version": 0,
    "last_upload_signature": None,
    "model_result": None,
    "last_cleaning_report": None,
    "cleaning_flash": None,
    "last_prediction": None,
}


def initialise_state() -> None:
    """Create session keys once so ordinary widget reruns preserve work."""
    for key, value in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def active_data() -> pd.DataFrame | None:
    """Return the current working table only if it is a DataFrame."""
    data = st.session_state.active_data
    return data if isinstance(data, pd.DataFrame) else None


def clear_model_result() -> None:
    """A model becomes stale whenever the working table changes."""
    st.session_state.model_result = None
    st.session_state.last_prediction = None


def set_dataset(frame: pd.DataFrame, source_name: str, *, replace_original: bool) -> None:
    """Set a new active copy while protecting the original upload in memory."""
    if replace_original:
        st.session_state.original_data = frame.copy(deep=True)
    st.session_state.active_data = frame.copy(deep=True)
    st.session_state.source_name = source_name
    st.session_state.data_version += 1
    st.session_state.last_cleaning_report = None
    clear_model_result()


def reset_analysis() -> None:
    """Clear session-held data without touching any files on disk."""
    for key, value in STATE_DEFAULTS.items():
        st.session_state[key] = value
    st.session_state.pop("data_uploader", None)


def load_sample() -> None:
    """Load the bundled dataset into a fresh working session."""
    try:
        dataset = pd.read_csv(SAMPLE_DATA_PATH)
        set_dataset(dataset, "Included passenger outcomes sample", replace_original=True)
        st.session_state.last_upload_signature = "sample"
    except (OSError, pd.errors.ParserError) as error:
        st.error(f"Could not load the included sample: {error}")


def render_sidebar() -> str:
    """Draw data source controls and return the user-selected page."""
    with st.sidebar:
        st.markdown("## 🧪 InsightLab")
        st.caption("Interactive Machine Learning Studio")
        st.divider()
        apply_theme(
            st.selectbox(
                "Appearance",
                ["Light", "Dark"],
                help="This adjusts InsightLab's workspace cards. Streamlit's menu also provides theme settings.",
            )
        )
        st.markdown("##### DATA SOURCE")
        uploaded = st.file_uploader(
            "Upload a CSV",
            type=["csv"],
            key="data_uploader",
            help="CSV files only, up to 25 MB. Data is parsed only in this browser session.",
        )
        if uploaded is not None:
            signature = f"{uploaded.name}:{uploaded.size}"
            if signature != st.session_state.last_upload_signature:
                try:
                    dataset, encoding = load_csv_bytes(uploaded.getvalue(), uploaded.name)
                    set_dataset(dataset, uploaded.name, replace_original=True)
                    st.session_state.last_upload_signature = signature
                    st.success(f"Loaded {uploaded.name} ({encoding}).")
                except DatasetLoadError as error:
                    st.error(str(error))

        if st.button("Use included sample", use_container_width=True):
            load_sample()
            st.rerun()
        if active_data() is not None:
            data = active_data()
            assert data is not None
            st.caption(f"Active: **{len(data):,} rows × {len(data.columns):,} columns**")
        st.button(
            "Reset current analysis",
            use_container_width=True,
            help="Clears the data and model held in this session.",
            on_click=reset_analysis,
        )

        st.divider()
        page = st.radio(
            "Navigate",
            [
                "Home",
                "Dataset Explorer",
                "Data Cleaning",
                "Visualisation Studio",
                "Machine Learning Studio",
                "Prediction Playground",
                "About",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Privacy first · No API keys · In-memory analysis")
    return page


def show_home() -> None:
    hero("Make your dataset make sense.", "Explore, clean, visualise, and test machine-learning ideas from one calm, transparent workspace.")
    data = active_data()
    if data is None:
        section("Start with a dataset", "Upload a CSV from the sidebar or open the ready-to-explore sample.", "Welcome")
        for column, details in zip(
            st.columns(3),
            [
                ("01 · Explore", "Profile your data", "Types, missing values, duplicates, summaries"),
                ("02 · Build", "Train responsibly", "Leakage-safe preprocessing and held-out tests"),
                ("03 · Apply", "Try a prediction", "A validated form appears after training"),
            ],
        ):
            with column:
                card(*details)
        st.write("")
        if st.button("Explore the included sample dataset", type="primary"):
            load_sample()
            st.rerun()
        st.info("**Your data stays in this session.** InsightLab has no paid APIs, cloud AI calls, accounts, or external dataset uploads.")
        return

    profile = build_profile(data)
    section("Your active workspace", f"Working with {st.session_state.source_name}. Continue with the sidebar workflow.", "Ready")
    for column, details in zip(
        st.columns(4),
        [
            ("Rows", f"{profile.rows:,}", "Current working dataset"),
            ("Columns", str(profile.columns), f"{len(profile.numeric_columns)} numeric"),
            ("Missing cells", f"{profile.missing_cells:,}", "Review in Dataset Explorer"),
            ("Model", "Ready" if st.session_state.model_result else "Not trained", "Open Machine Learning Studio"),
        ],
    ):
        with column:
            card(*details)
    st.write("")
    st.subheader("Suggested workflow")
    stage = 3 if st.session_state.model_result else 2 if st.session_state.last_cleaning_report else 1
    st.progress(stage / 3, text=f"Step {stage} of 3")
    steps = st.columns(3)
    steps[0].markdown("**1. Understand**<br>Inspect quality checks and rule-based observations.", unsafe_allow_html=True)
    steps[1].markdown("**2. Prepare**<br>Create a cleaned working copy; the original stays preserved.", unsafe_allow_html=True)
    steps[2].markdown("**3. Evaluate**<br>Train on one split, test on another, then interpret carefully.", unsafe_allow_html=True)


def show_explorer() -> None:
    data = active_data()
    section("Dataset Explorer", "Inspect structure, quality, and transparent rule-based observations before modelling.", "Understand")
    if data is None:
        empty_state("No dataset yet", "Upload a CSV or use the included sample from the sidebar to begin.")
        return
    profile = build_profile(data)
    for column, details in zip(
        st.columns(4),
        [
            ("Rows", f"{profile.rows:,}", "Records in the working copy"),
            ("Columns", str(profile.columns), f"{len(profile.numeric_columns)} numeric · {len(profile.categorical_columns)} other"),
            ("Missing values", f"{profile.missing_cells:,}", "Across all cells"),
            ("Duplicate rows", f"{profile.duplicate_rows:,}", "Exact complete-row matches"),
        ],
    ):
        with column:
            card(*details)

    overview, quality, insights = st.tabs(["Overview", "Data quality", "Rule-based insights"])
    with overview:
        st.subheader("Sample rows")
        st.caption(f"Showing up to {MAX_PREVIEW_ROWS} rows. Values and column order are preserved.")
        st.dataframe(data.head(MAX_PREVIEW_ROWS), use_container_width=True, height=330)
        st.subheader("Summary statistics")
        st.caption("Numerical columns show distribution statistics; categorical columns show counts and frequent values when available.")
        st.dataframe(profile.summary, use_container_width=True, height=360)
    with quality:
        left, right = st.columns([1.35, 1])
        with left:
            st.subheader("Column-level quality")
            st.dataframe(profile.column_quality, use_container_width=True, height=420)
        with right:
            st.subheader("Missing values")
            missing = profile.column_quality.query("Missing > 0").sort_values("Missing", ascending=False)
            if missing.empty:
                st.success("No missing values detected in the working dataset.")
            else:
                chart = px.bar(missing.head(20), x="Missing", y="Column", orientation="h", color="Missing %", color_continuous_scale="Purples")
                chart.update_layout(title="Top columns with missing values", yaxis={"categoryorder": "total ascending"}, margin=dict(l=20, r=20, t=55, b=20))
                st.plotly_chart(chart, use_container_width=True)
            st.caption(f"Approximate in-memory size: {profile.memory_bytes / 1_048_576:.2f} MB")
    with insights:
        st.info("These observations use fixed Python checks and templates. They are not generated by an AI service.")
        for insight in generate_data_insights(data, profile):
            st.markdown(f"- {insight}")


def show_cleaning() -> None:
    data = active_data()
    section("Data Cleaning", "Create a new working copy with reversible choices. Your uploaded original remains untouched in session state.", "Prepare")
    if data is None:
        empty_state("No data to clean", "Load a CSV first, then choose columns and quality fixes.")
        return
    if st.session_state.cleaning_flash:
        st.success(str(st.session_state.cleaning_flash))
        st.session_state.cleaning_flash = None
    profile = build_profile(data)
    metrics = st.columns(3)
    metrics[0].metric("Current rows", f"{profile.rows:,}")
    metrics[1].metric("Missing cells", f"{profile.missing_cells:,}")
    metrics[2].metric("Exact duplicates", f"{profile.duplicate_rows:,}")

    with st.form("cleaning_form", clear_on_submit=False):
        chosen = st.multiselect("Columns to keep", data.columns.tolist(), default=data.columns.tolist(), help="This affects a new working copy only.")
        left, right = st.columns(2)
        with left:
            remove_duplicates = st.checkbox("Remove exact duplicate rows")
            strategy = st.selectbox(
                "Missing-value treatment",
                ["Keep missing values", "Drop rows with missing values", "Fill numeric with median; labels with mode", "Fill numeric with mean; labels with mode", "Fill all missing values with a custom value"],
                help="Median is less sensitive to outliers than mean. Mode fills labels with their most common value.",
            )
        with right:
            replacement = st.text_input("Custom replacement value", "Unknown", disabled=strategy != "Fill all missing values with a custom value")
            st.caption("Cleaning always starts from the current active copy. Reset restores the original uploaded source.")
        submitted = st.form_submit_button("Create cleaned working copy", type="primary")
    if submitted:
        try:
            cleaned, report = apply_cleaning(data, CleaningOptions(chosen, remove_duplicates, strategy, replacement))
            set_dataset(cleaned, f"Cleaned copy of {st.session_state.source_name}", replace_original=False)
            st.session_state.last_cleaning_report = report
            st.session_state.cleaning_flash = f"Created a cleaned working copy: {report['rows_after']:,} rows, {report['columns_kept']} columns, and {report['missing_after']:,} missing cells remaining."
            st.rerun()
        except ValueError as error:
            st.error(str(error))
    report = st.session_state.last_cleaning_report
    if isinstance(report, dict):
        st.subheader("Latest cleaning report")
        for column, (label, key) in zip(st.columns(4), [("Rows before", "rows_before"), ("Rows after", "rows_after"), ("Duplicates removed", "duplicates_removed"), ("Missing cells remaining", "missing_after")]):
            column.metric(label, f"{report[key]:,}")
    st.divider()
    st.subheader("Download the working copy")
    st.caption("The export is generated in memory and never overwrites the upload.")
    st.download_button("Download cleaned / working CSV", dataframe_to_csv_bytes(data), "insightlab_cleaned_data.csv", "text/csv")


def chart_data(data: pd.DataFrame) -> pd.DataFrame:
    """Limit visual rendering volume without modifying the active dataset."""
    if len(data) <= MAX_CHART_ROWS:
        return data
    st.warning(f"Charts use a reproducible sample of {MAX_CHART_ROWS:,} rows to stay responsive. The full data remains available for modelling.")
    return data.sample(MAX_CHART_ROWS, random_state=42)


def show_visualisations() -> None:
    data = active_data()
    section("Visualisation Studio", "Choose a chart and variables to inspect patterns interactively. Charts use the active working copy.", "Explore")
    if data is None:
        empty_state("No data to visualise", "Load a CSV first. Chart controls will adapt to its numeric and categorical columns.")
        return
    chart_frame = chart_data(data)
    numeric, categorical = get_column_groups(chart_frame)
    chart_type = st.selectbox("Chart type", ["Histogram", "Bar chart", "Scatter plot", "Correlation heatmap", "Box plot"])

    if chart_type == "Histogram":
        if not numeric:
            st.warning("A histogram needs at least one numeric column.")
            return
        column = st.selectbox("Numeric column", numeric, key="histogram_column")
        bins = st.slider("Number of bins", 5, 80, 30)
        figure = px.histogram(chart_frame, x=column, nbins=bins, marginal="box", color_discrete_sequence=["#5B5CE2"])
        figure.update_layout(title=f"Distribution of {column}", yaxis_title="Records", margin=dict(l=20, r=20, t=55, b=20))
        st.plotly_chart(figure, use_container_width=True)
        st.caption("Histograms group numeric values into bins, revealing their spread, concentration, and possible outliers.")

    elif chart_type == "Bar chart":
        category = st.selectbox("Category column", categorical or data.columns.tolist(), key="bar_category")
        measure = st.radio("Measure", ["Count records", "Average a numeric column"], horizontal=True)
        if measure == "Count records":
            plot_data = chart_frame[category].fillna("(missing)").astype(str).value_counts().head(30).reset_index()
            plot_data.columns = [category, "Count"]
            figure = px.bar(plot_data, x=category, y="Count", color_discrete_sequence=["#17B6A4"])
            title = f"Record count by {category}"
        else:
            if not numeric:
                st.warning("An average bar chart needs a numeric column.")
                return
            value = st.selectbox("Numeric value", numeric, key="bar_value")
            plot_data = (
                chart_frame.assign(**{category: chart_frame[category].fillna("(missing)").astype(str)})
                .groupby(category, dropna=False)[value]
                .mean()
                .sort_values(ascending=False)
                .head(30)
                .reset_index()
            )
            figure = px.bar(plot_data, x=category, y=value, color_discrete_sequence=["#17B6A4"])
            title = f"Average {value} by {category}"
        figure.update_layout(title=title, margin=dict(l=20, r=20, t=55, b=80))
        st.plotly_chart(figure, use_container_width=True)
        st.caption("The display is limited to 30 categories so labels remain readable.")

    elif chart_type == "Scatter plot":
        if len(numeric) < 2:
            st.warning("A scatter plot needs at least two numeric columns.")
            return
        left, right = st.columns(2)
        x_axis = left.selectbox("X axis", numeric)
        y_axis = right.selectbox("Y axis", [name for name in numeric if name != x_axis])
        colour = st.selectbox("Colour by (optional)", ["None"] + categorical[:30])
        figure = px.scatter(
            chart_frame,
            x=x_axis,
            y=y_axis,
            color=None if colour == "None" else colour,
            opacity=0.72,
            hover_data=chart_frame.columns[:8].tolist(),
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        figure.update_layout(title=f"{y_axis} vs {x_axis}", margin=dict(l=20, r=20, t=55, b=20))
        st.plotly_chart(figure, use_container_width=True)
        st.caption("Look for clusters, trends, and unusual points. A visible relationship does not establish cause and effect.")

    elif chart_type == "Correlation heatmap":
        if len(numeric) < 2:
            st.warning("A correlation heatmap needs at least two numeric columns.")
            return
        correlation = chart_frame[numeric].corr(numeric_only=True)
        figure = go.Figure(go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.index,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            colorbar_title="Pearson r",
            hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
        ))
        figure.update_layout(title="Numeric feature correlation", height=max(430, 42 * len(numeric)), margin=dict(l=20, r=20, t=55, b=70))
        st.plotly_chart(figure, use_container_width=True)
        st.caption("Pearson correlation ranges from -1 to 1 and measures linear association only. It does not prove causation.")

    else:  # Box plot
        if not numeric:
            st.warning("A box plot needs a numeric column.")
            return
        value = st.selectbox("Numeric value", numeric, key="box_value")
        group = st.selectbox("Group by (optional)", ["None"] + categorical[:30], key="box_group")
        figure = px.box(
            chart_frame,
            x=None if group == "None" else group,
            y=value,
            color=None if group == "None" else group,
            points="outliers",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        figure.update_layout(title=f"Spread of {value}" if group == "None" else f"Spread of {value} by {group}", margin=dict(l=20, r=20, t=55, b=50))
        st.plotly_chart(figure, use_container_width=True)
        st.caption("The box shows the middle half of values; points beyond the whiskers are potential outliers, not automatically mistakes.")


def show_model_results(result: TrainingResult) -> None:
    """Render held-out evaluation artefacts for one fitted pipeline."""
    st.divider()
    st.subheader("Held-out test evaluation")
    st.caption(
        f"{result.algorithm} trained on {result.training_rows:,} row(s) and was evaluated on {result.test_rows:,} unseen row(s). "
        "A score is evidence about this split, not a guarantee of real-world reliability."
    )
    for column, (name, value) in zip(st.columns(len(result.metrics)), result.metrics.items()):
        display = f"{value:.3f}" if result.task == "classification" or name == "R²" else f"{value:.4g}"
        column.metric(name, display)

    if result.task == "classification" and result.confusion is not None:
        left, right = st.columns([1.2, 1])
        with left:
            figure = go.Figure(go.Heatmap(
                z=result.confusion.values,
                x=result.confusion.columns.tolist(),
                y=result.confusion.index.tolist(),
                colorscale="Blues",
                hovertemplate="Actual %{y}; predicted %{x}: %{z}<extra></extra>",
            ))
            figure.update_layout(title="Confusion matrix", xaxis_title="Predicted class", yaxis_title="Actual class", height=400)
            st.plotly_chart(figure, use_container_width=True)
        with right:
            st.markdown("##### What this means")
            st.caption("Cells on the diagonal are correct predictions. Other cells show which classes the model confused.")
            if result.classification_report is not None:
                st.dataframe(result.classification_report.round(3), use_container_width=True, height=300)
    with st.expander("Metric guide"):
        for metric, explanation in metric_explanations(result.task).items():
            st.markdown(f"**{metric}** — {explanation}")
    st.subheader("Test-set predictions")
    st.caption("These are the held-out rows used for evaluation, not the rows that trained the model.")
    st.dataframe(result.test_results.head(MAX_PREVIEW_ROWS), use_container_width=True, height=290)
    st.download_button("Download test predictions as CSV", dataframe_to_csv_bytes(result.test_results), "insightlab_test_predictions.csv", "text/csv")


def show_machine_learning() -> None:
    data = active_data()
    section("Machine Learning Studio", "Train a supervised classification or regression model with preprocessing isolated inside the training pipeline.", "Model")
    if data is None:
        empty_state("No training data", "Load and inspect a CSV before selecting a target and features.")
        return
    if len(data.columns) < 2:
        st.warning("Add a dataset with at least two columns: one target and one feature.")
        return
    st.info("**Leakage guard:** imputation, scaling, and one-hot encoding are fitted only on training rows. Rows missing a target are excluded rather than assigned a guessed label.")
    target = st.selectbox("Target column", data.columns.tolist(), help="The value the model will learn to predict.")
    choice = st.radio("Prediction task", ["Automatically detect", "Classification", "Regression"], horizontal=True)
    try:
        task = infer_task(data, target) if choice == "Automatically detect" else choice.lower()
    except ModelTrainingError as error:
        st.error(str(error))
        return
    st.caption(f"Selected task: **{task.title()}**" + (" (inferred from target data type and cardinality)." if choice == "Automatically detect" else "."))
    feature_candidates = [column for column in data.columns if column != target]
    features = st.multiselect(
        "Feature columns",
        feature_candidates,
        default=feature_candidates,
        help="Exclude identifiers, post-outcome fields, and information unavailable at real prediction time.",
    )
    left, middle, right = st.columns([1, 1, 0.7])
    with left:
        algorithm = st.selectbox("Algorithm", available_algorithms(task))
    with middle:
        test_size = st.slider("Test-set share", 0.15, 0.40, 0.20, 0.05)
    with right:
        st.write("")
        st.write("")
        train_clicked = st.button("Train model", type="primary", use_container_width=True)
    if train_clicked:
        with st.spinner("Training model and evaluating held-out rows…"):
            try:
                result = train_model(data, target, features, task, algorithm, float(test_size))  # type: ignore[arg-type]
                st.session_state.model_result = result
                st.session_state.last_prediction = None
                st.success("Model trained successfully. Open Prediction Playground to try a new input.")
            except ModelTrainingError as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"Unexpected training error: {error}. Try a cleaner selection of columns or a larger dataset.")
    result = st.session_state.model_result
    if isinstance(result, TrainingResult):
        show_model_results(result)
    else:
        st.caption("No model has been trained in this session yet.")


def show_prediction_playground() -> None:
    result = st.session_state.model_result
    section("Prediction Playground", "Enter feature values for the latest trained model. Values are validated before a prediction is requested.", "Apply")
    if not isinstance(result, TrainingResult):
        empty_state("Train a model first", "The prediction form appears after a model trains successfully in Machine Learning Studio.")
        return
    st.info(f"Using **{result.algorithm}** to predict **{result.target}** from {len(result.features)} feature(s). Choose values that are realistic for the training data.")
    values: dict[str, object] = {}
    with st.form("prediction_form"):
        columns = st.columns(2)
        for index, (name, spec) in enumerate(result.feature_schema.items()):
            with columns[index % 2]:
                if spec["kind"] == "numeric":
                    values[name] = st.number_input(name, value=float(spec["default"]), step=0.1, format="%.4f", help="A median training value is shown as a starting point.")
                else:
                    values[name] = st.selectbox(name, spec["categories"], help="Only categories observed during training are offered.")
        submitted = st.form_submit_button("Generate prediction", type="primary")
    if submitted:
        try:
            request = build_prediction_frame(values, result.feature_schema)
            prediction = result.pipeline.predict(request)[0]
            display = f"{float(prediction):,.4g}" if result.task == "regression" else str(prediction)
            prediction_record: dict[str, object] = {"value": display, "inputs": request}
            if result.task == "classification" and hasattr(result.pipeline, "predict_proba"):
                probabilities = result.pipeline.predict_proba(request)[0]
                model = result.pipeline.named_steps["model"]
                prediction_record["probabilities"] = pd.DataFrame({"Class": model.classes_, "Probability": probabilities}).sort_values("Probability", ascending=False)
            st.session_state.last_prediction = prediction_record
        except ModelTrainingError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"The prediction could not be generated: {error}")
    latest = st.session_state.last_prediction
    if isinstance(latest, dict):
        st.success(f"Predicted **{result.target}**: **{latest['value']}**")
        if "probabilities" in latest:
            st.subheader("Class probabilities")
            st.dataframe(latest["probabilities"].style.format({"Probability": "{:.1%}"}), use_container_width=True)
        with st.expander("View submitted feature values"):
            st.dataframe(latest["inputs"], use_container_width=True)
        st.warning("A prediction is an output of this experimental model, not professional, medical, legal, financial, or safety advice. Validate it on relevant data before using it for a real decision.")


def show_about() -> None:
    section("About InsightLab", "A portfolio-ready, transparent learning environment for practical data analysis and machine learning.", "Product notes")
    st.markdown("""
### What it does
InsightLab turns a CSV into an in-session workflow for profiling, careful cleaning, interactive charts, supervised learning, and manual predictions. It explains checks and metrics with deterministic Python rules and plain English.

### Privacy and security
- CSV uploads are parsed in memory and are not written to an external database by this application.
- The app accepts only `.csv` files, applies a 25 MB limit, and never executes uploaded files or user-supplied code.
- There are no API keys, accounts, payments, external AI calls, or telemetry in this project.

### Responsible use
Model scores come from one held-out split and can vary with the data and random sampling. A high score is not proof of fairness, causality, generalisation, or production readiness. Review data provenance, bias, validation, and domain constraints before acting on results.

### Included stack
Python, Streamlit, Pandas, NumPy, Plotly, and scikit-learn. The README documents installation, tests, deployment, and the module architecture.
""")


def main() -> None:
    initialise_state()
    page = render_sidebar()
    pages = {
        "Home": show_home,
        "Dataset Explorer": show_explorer,
        "Data Cleaning": show_cleaning,
        "Visualisation Studio": show_visualisations,
        "Machine Learning Studio": show_machine_learning,
        "Prediction Playground": show_prediction_playground,
        "About": show_about,
    }
    pages[page]()


if __name__ == "__main__":
    main()

