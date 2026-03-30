from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _prepend_python_script_dir_to_path() -> None:
    """Ensure console entry points installed by the active Python are discoverable."""
    python_dir = Path(sys.executable).resolve().parent
    script_dirs = [python_dir / "Scripts", python_dir / "bin"]
    current_path = os.environ.get("PATH", "")
    for script_dir in script_dirs:
        if script_dir.exists() and str(script_dir) not in current_path.split(os.pathsep):
            os.environ["PATH"] = f"{script_dir}{os.pathsep}{current_path}"
            current_path = os.environ["PATH"]


def _load_installed_state(state_path: Path) -> dict[str, str]:
    if not state_path.exists():
        return {}
    try:
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            return {str(key): str(value) for key, value in loaded.items()}
    except json.JSONDecodeError:
        return {}
    return {}


def _save_installed_state(state_path: Path, state: dict[str, str]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")


def _ensure_runtime_dependencies(requirements: dict[str, str]) -> None:
    """Install missing dependencies once for the full SEA-AD patient association pipeline."""
    repo_root = Path(__file__).resolve().parents[1]
    state_path = repo_root / ".runtime" / "seaad_patient_dependency_state.json"
    installed_state = _load_installed_state(state_path)

    missing_packages: list[str] = []
    for module_name, package_name in requirements.items():
        module_ready = importlib.util.find_spec(module_name) is not None
        if module_ready:
            installed_state[module_name] = package_name
            continue
        missing_packages.append(package_name)

    missing_packages = sorted(set(missing_packages))
    if not missing_packages:
        _save_installed_state(state_path, installed_state)
        return

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-warn-script-location",
        *missing_packages,
    ]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        joined = " ".join(cmd)
        raise RuntimeError(
            f"Failed to install missing packages for interpreter {sys.executable}: {missing_packages}. Command: {joined}"
        ) from exc

    unresolved_modules: list[str] = []
    for module_name, package_name in requirements.items():
        if importlib.util.find_spec(module_name) is None:
            unresolved_modules.append(module_name)
        else:
            installed_state[module_name] = package_name
    if unresolved_modules:
        raise RuntimeError(
            "Dependency installation completed but modules are still missing "
            f"for this interpreter ({sys.executable}): {unresolved_modules}"
        )
    _save_installed_state(state_path, installed_state)


_prepend_python_script_dir_to_path()
_ensure_runtime_dependencies(
    {
        "matplotlib": "matplotlib",
        "numpy": "numpy",
        "pandas": "pandas",
        "seaborn": "seaborn",
        "scipy": "scipy",
        "sklearn": "scikit-learn",
        "openpyxl": "openpyxl",
        "networkx": "networkx",
    }
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import gaussian_kde, t
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree


OFFICIAL_SOURCE_PAGE = "https://brain-map.org/explore/seattle-alzheimers-disease/seattle-alzheimers-disease-brain-cell-atlas-download"
LOCAL_DEFAULT_SOURCE = Path("data/raw/donor_metadata/donor_metadata_Donor-metadata.xlsx")
ID_COLUMNS = {"Donor ID"}
MISSING_LABEL = "Missing"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a SEA-AD patient metadata project with robust summaries, one-hot encoding, tree diagnostics, and plots."
    )
    parser.add_argument(
        "--project-root",
        default=r"..\SEA-AD-patient",
        help="Target project directory to create under the AD folder.",
    )
    parser.add_argument(
        "--source-xlsx",
        default=str(LOCAL_DEFAULT_SOURCE),
        help="Official SEA-AD donor metadata xlsx path. Local official file is preferred when already downloaded.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tree-max-depth", type=int, default=3)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--numeric-threshold", type=float, default=0.8)
    return parser


def ensure_dirs(project_root: Path) -> dict[str, Path]:
    dirs = {
        "project": project_root,
        "scripts": project_root / "scripts",
        "output": project_root / "output",
        "output_raw": project_root / "output" / "raw",
        "output_processed": project_root / "output" / "processed",
        "results": project_root / "results",
        "plots": project_root / "results" / "plots",
        "plots_numeric": project_root / "results" / "plots" / "numeric",
        "plots_categorical": project_root / "results" / "plots" / "categorical",
        "plots_tree": project_root / "results" / "plots" / "categorical_tree",
        "plots_terminal": project_root / "results" / "plots" / "terminal_nodes",
        "plots_dendrogram": project_root / "results" / "plots" / "dendrogram",
        "csv": project_root / "results" / "csv",
        "tsv": project_root / "results" / "tsv",
        "check": project_root / "check",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("seaad_patient_metadata_analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def write_dual_table(df: pd.DataFrame, base_name: str, csv_dir: Path, tsv_dir: Path) -> tuple[Path, Path]:
    csv_path = csv_dir / f"{base_name}.csv"
    tsv_path = tsv_dir / f"{base_name}.tsv"
    df.to_csv(csv_path, index=False)
    df.to_csv(tsv_path, index=False, sep="\t")
    return csv_path, tsv_path


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_self_to_project(project_scripts_dir: Path) -> Path:
    source = Path(__file__).resolve()
    target = project_scripts_dir / source.name
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def load_metadata(source_xlsx: Path, output_raw_dir: Path, logger: logging.Logger) -> tuple[pd.DataFrame, Path]:
    if not source_xlsx.exists():
        raise FileNotFoundError(f"SEA-AD donor metadata xlsx was not found: {source_xlsx}")
    copied_source = output_raw_dir / "seaad_patient_metadata_original.xlsx"
    shutil.copy2(source_xlsx, copied_source)
    logger.info("Loaded official SEA-AD donor metadata from local file: %s", source_xlsx)
    df = pd.read_excel(copied_source, engine="openpyxl")
    return df, copied_source


def resolve_source_xlsx(source_xlsx_arg: str, project_root: Path) -> Path:
    requested = Path(source_xlsx_arg)
    repo_root = Path(__file__).resolve().parents[1]
    basename = requested.name

    candidates: list[Path] = []

    def add_candidate(path: Path) -> None:
        if path not in candidates:
            candidates.append(path)

    if requested.is_absolute():
        add_candidate(requested)
    else:
        # Priority 1: current working directory data folder (user-requested behavior)
        add_candidate(Path.cwd() / requested)
        add_candidate(Path.cwd() / "data" / basename)
        add_candidate(Path.cwd() / "data" / "raw" / "donor_metadata" / basename)
        # Priority 2: explicitly configured project root
        add_candidate(project_root / requested)
        add_candidate(project_root / "data" / basename)
        add_candidate(project_root / "data" / "raw" / "donor_metadata" / basename)
        # Priority 3: repository root
        add_candidate(repo_root / requested)
        add_candidate(repo_root / "data" / basename)
        add_candidate(repo_root / "data" / "raw" / "donor_metadata" / basename)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Fallback: recursive lookup by exact filename in common data roots
    for data_root in [Path.cwd() / "data", project_root / "data", repo_root / "data"]:
        if not data_root.exists():
            continue
        matches = sorted(data_root.rglob(basename))
        if matches:
            return matches[0].resolve()

    searched = "\n".join(f"- {str(path)}" for path in candidates)
    raise FileNotFoundError(
        "SEA-AD donor metadata xlsx was not found. "
        f"Searched for '{basename}' in these locations:\n{searched}"
    )


def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in cleaned.columns:
        if pd.api.types.is_object_dtype(cleaned[column]) or pd.api.types.is_string_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
    cleaned = cleaned.replace(
        {
            "": pd.NA,
            "NA": pd.NA,
            "N/A": pd.NA,
            "na": pd.NA,
            "n/a": pd.NA,
            "None": pd.NA,
            "none": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
        }
    )
    return cleaned


def infer_variable_types(df: pd.DataFrame, numeric_threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        role = "identifier" if column in ID_COLUMNS else "analysis"
        converted = pd.to_numeric(series, errors="coerce")
        numeric_ratio = float(converted.notna().mean()) if len(series) else 0.0
        n_unique = int(series.dropna().nunique())
        if column in ID_COLUMNS:
            inferred = "identifier"
        elif pd.api.types.is_numeric_dtype(series):
            inferred = "numeric"
        elif numeric_ratio >= numeric_threshold and n_unique >= 5:
            inferred = "numeric"
        else:
            inferred = "categorical"
        rows.append(
            {
                "variable": column,
                "role": role,
                "inferred_type": inferred,
                "n_non_missing": int(series.notna().sum()),
                "n_missing": int(series.isna().sum()),
                "n_unique_non_missing": n_unique,
                "numeric_parse_ratio": numeric_ratio,
            }
        )
    return pd.DataFrame(rows)


def safe_slug(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("=", "-")
        .replace(":", "_")
        .replace(",", "")
    )


def make_numeric_df(df: pd.DataFrame, numeric_vars: list[str]) -> pd.DataFrame:
    numeric_df = pd.DataFrame(index=df.index)
    for column in numeric_vars:
        numeric_df[column] = pd.to_numeric(df[column], errors="coerce")
    return numeric_df


def prepare_categorical_series(series: pd.Series) -> tuple[pd.Series, list[str], str]:
    string_series = series.astype("string").fillna(MISSING_LABEL)
    observed_levels = sorted({str(value) for value in string_series.tolist()})
    categories = observed_levels if observed_levels else [MISSING_LABEL]
    cat = pd.Series(pd.Categorical(string_series, categories=categories), index=series.index, name=series.name)
    reference = categories[0]
    return cat.astype("string"), categories, reference


def one_hot_encode(df: pd.DataFrame, categorical_vars: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    encoded_parts: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for column in categorical_vars:
        prepared, categories, reference = prepare_categorical_series(df[column])
        cat = pd.Categorical(prepared, categories=categories)
        dummies = pd.get_dummies(cat, prefix=safe_slug(column), prefix_sep="=", drop_first=True, dtype=int)
        dummies.index = df.index
        encoded_parts.append(dummies)
        for level in categories:
            dummy_name = f"{safe_slug(column)}={level}" if level != reference else ""
            manifest_rows.append(
                {
                    "variable": column,
                    "level": level,
                    "reference_level": reference,
                    "dummy_column": dummy_name,
                    "kept_as_dummy": level != reference,
                }
            )
    encoded = pd.concat(encoded_parts, axis=1) if encoded_parts else pd.DataFrame(index=df.index)
    manifest = pd.DataFrame(manifest_rows)
    return encoded, manifest


def build_predictor_matrix(
    df: pd.DataFrame,
    numeric_vars: list[str],
    categorical_vars: list[str],
    exclude: str,
) -> pd.DataFrame:
    predictor_numeric = [col for col in numeric_vars if col != exclude]
    predictor_categorical = [col for col in categorical_vars if col != exclude]
    numeric_part = make_numeric_df(df, predictor_numeric)
    if not numeric_part.empty:
        for column in numeric_part.columns:
            numeric_part[column] = numeric_part[column].fillna(numeric_part[column].median())
    categorical_part, _ = one_hot_encode(df[predictor_categorical], predictor_categorical) if predictor_categorical else (pd.DataFrame(index=df.index), pd.DataFrame())
    matrix = pd.concat([numeric_part, categorical_part], axis=1)
    return matrix


def numeric_summary(series: pd.Series, variable: str) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {
            "variable": variable,
            "n": 0,
            "missing": int(series.isna().sum()),
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
            "max": np.nan,
            "skew": np.nan,
            "kurtosis": np.nan,
            "t_df": np.nan,
            "t_loc": np.nan,
            "t_scale": np.nan,
        }
    t_df, t_loc, t_scale = t.fit(numeric.to_numpy())
    return {
        "variable": variable,
        "n": int(numeric.shape[0]),
        "missing": int(series.isna().sum()),
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=1)) if numeric.shape[0] > 1 else 0.0,
        "min": float(numeric.min()),
        "q25": float(numeric.quantile(0.25)),
        "median": float(numeric.median()),
        "q75": float(numeric.quantile(0.75)),
        "max": float(numeric.max()),
        "skew": float(numeric.skew()) if numeric.shape[0] > 2 else np.nan,
        "kurtosis": float(numeric.kurtosis()) if numeric.shape[0] > 3 else np.nan,
        "t_df": float(t_df),
        "t_loc": float(t_loc),
        "t_scale": float(t_scale),
    }


def plot_numeric_distribution(series: pd.Series, variable: str, out_path: Path) -> None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return
    t_df, t_loc, t_scale = t.fit(numeric.to_numpy())
    x_grid = np.linspace(float(numeric.min()), float(numeric.max()), 300)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(numeric, bins=min(20, max(5, numeric.shape[0] // 3)), stat="density", color="#9ecae1", edgecolor="white", ax=ax)
    if numeric.nunique() > 1:
        kde = gaussian_kde(numeric.to_numpy())
        ax.plot(x_grid, kde(x_grid), color="#2c7fb8", linewidth=2, label="KDE")
    ax.plot(x_grid, t.pdf(x_grid, df=t_df, loc=t_loc, scale=t_scale), color="#d95f0e", linewidth=2, label="Fitted t curve")
    ax.set_title(f"{variable} distribution")
    ax.set_xlabel(variable)
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def categorical_counts(series: pd.Series, variable: str) -> pd.DataFrame:
    prepared, _, reference = prepare_categorical_series(series)
    counts = prepared.value_counts(dropna=False).rename_axis("level").reset_index(name="count")
    counts.insert(0, "variable", variable)
    counts["reference_level"] = reference
    counts["fraction"] = counts["count"] / counts["count"].sum()
    return counts


def categorical_summary(series: pd.Series, variable: str) -> dict[str, Any]:
    counts = categorical_counts(series, variable)
    mode_row = counts.sort_values(["count", "level"], ascending=[False, True]).iloc[0]
    return {
        "variable": variable,
        "n": int(series.shape[0]),
        "missing": int(series.isna().sum()),
        "n_levels_including_missing": int(counts.shape[0]),
        "reference_level": str(counts["reference_level"].iloc[0]),
        "mode_level": str(mode_row["level"]),
        "mode_count": int(mode_row["count"]),
        "mode_fraction": float(mode_row["fraction"]),
    }


def plot_categorical_counts(counts: pd.DataFrame, variable: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * counts.shape[0] + 1)))
    sns.barplot(data=counts, x="count", y="level", orient="h", color="#74a9cf", ax=ax)
    ax.set_title(f"{variable} level counts")
    ax.set_xlabel("Count")
    ax.set_ylabel("Level")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_dendrogram_for_categorical(
    df: pd.DataFrame,
    target: str,
    predictors: pd.DataFrame,
    out_path: Path,
) -> str:
    prepared, _, _ = prepare_categorical_series(df[target])
    if prepared.nunique() < 2:
        return "not_enough_levels"
    centroids = predictors.assign(__target__=prepared).groupby("__target__", dropna=False).mean(numeric_only=True)
    if centroids.shape[0] < 2 or centroids.shape[1] == 0:
        return "not_enough_predictor_signal"
    linkage_matrix = linkage(centroids.to_numpy(), method="ward", optimal_ordering=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * centroids.shape[0] + 1)))
    dendrogram(linkage_matrix, labels=centroids.index.astype(str).tolist(), leaf_rotation=45, ax=ax)
    ax.set_title(f"{target} level dendrogram")
    ax.set_ylabel("Ward distance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return "ok"


def tree_classifier_diagnostics(
    df: pd.DataFrame,
    target: str,
    numeric_vars: list[str],
    categorical_vars: list[str],
    plots_tree_dir: Path,
    plots_terminal_dir: Path,
    csv_dir: Path,
    tsv_dir: Path,
    seed: int,
    max_depth: int,
    min_samples_leaf: int,
) -> dict[str, Any]:
    y = prepare_categorical_series(df[target])[0]
    if y.nunique() < 2:
        return {
            "variable": target,
            "model_type": "decision_tree_classifier",
            "n_samples": int(y.shape[0]),
            "n_classes": int(y.nunique()),
            "r_square_reported": np.nan,
            "misclassification_rate": np.nan,
            "accuracy": np.nan,
            "status": "skipped_single_class",
        }
    x = build_predictor_matrix(df, numeric_vars, categorical_vars, exclude=target)
    if x.empty:
        return {
            "variable": target,
            "model_type": "decision_tree_classifier",
            "n_samples": int(y.shape[0]),
            "n_classes": int(y.nunique()),
            "r_square_reported": np.nan,
            "misclassification_rate": np.nan,
            "accuracy": np.nan,
            "status": "skipped_no_predictors",
        }

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
    )
    model.fit(x, y)
    pred = pd.Series(model.predict(x), index=y.index)
    probas = model.predict_proba(x)
    classes = model.classes_
    class_priors = y.value_counts(normalize=True).reindex(classes).to_numpy()
    null_probas = np.tile(class_priors, (len(y), 1))
    model_log_loss = log_loss(y, probas, labels=list(classes))
    null_log_loss = log_loss(y, null_probas, labels=list(classes))
    pseudo_r2 = 1 - (model_log_loss / null_log_loss) if null_log_loss > 0 else np.nan
    accuracy = accuracy_score(y, pred)
    misclassification = 1 - accuracy
    leaf_ids = model.apply(x)
    leaf_df = pd.DataFrame({"leaf_id": leaf_ids, "target": y, "prediction": pred})
    leaf_summary = (
        leaf_df.groupby("leaf_id")
        .agg(
            n_samples=("leaf_id", "size"),
            majority_class=("target", lambda s: s.value_counts().index[0]),
            majority_fraction=("target", lambda s: s.value_counts(normalize=True).iloc[0]),
        )
        .reset_index()
        .sort_values("n_samples", ascending=False)
    )
    write_dual_table(leaf_summary, f"{safe_slug(target)}_terminal_nodes", csv_dir, tsv_dir)

    fig, ax = plt.subplots(figsize=(max(10, 0.7 * x.shape[1]), 6))
    plot_tree(
        model,
        feature_names=x.columns.tolist(),
        class_names=[str(item) for item in classes],
        filled=True,
        impurity=False,
        proportion=True,
        rounded=True,
        ax=ax,
    )
    ax.set_title(f"Decision tree for {target}")
    fig.tight_layout()
    fig.savefig(plots_tree_dir / f"{safe_slug(target)}_decision_tree.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * leaf_summary.shape[0] + 1)))
    sns.barplot(data=leaf_summary, x="n_samples", y=leaf_summary["leaf_id"].astype(str), hue="majority_class", dodge=False, ax=ax)
    ax.set_title(f"{target} terminal nodes")
    ax.set_xlabel("Samples in terminal node")
    ax.set_ylabel("Leaf ID")
    ax.legend(frameon=False, title="Majority class", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_terminal_dir / f"{safe_slug(target)}_terminal_nodes.png", dpi=180)
    plt.close(fig)

    return {
        "variable": target,
        "model_type": "decision_tree_classifier",
        "n_samples": int(y.shape[0]),
        "n_classes": int(y.nunique()),
        "r_square_reported": float(pseudo_r2),
        "misclassification_rate": float(misclassification),
        "accuracy": float(accuracy),
        "status": "ok",
    }


def tree_regressor_diagnostics(
    df: pd.DataFrame,
    target: str,
    numeric_vars: list[str],
    categorical_vars: list[str],
    seed: int,
    max_depth: int,
    min_samples_leaf: int,
) -> dict[str, Any]:
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.notna()
    y = y[valid]
    if y.shape[0] < 5 or y.nunique() < 2:
        return {
            "variable": target,
            "model_type": "decision_tree_regressor",
            "n_samples": int(y.shape[0]),
            "r_square_reported": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "status": "skipped_not_enough_data",
        }
    x = build_predictor_matrix(df.loc[valid], numeric_vars, categorical_vars, exclude=target)
    if x.empty:
        return {
            "variable": target,
            "model_type": "decision_tree_regressor",
            "n_samples": int(y.shape[0]),
            "r_square_reported": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "status": "skipped_no_predictors",
        }
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
    )
    model.fit(x, y)
    pred = model.predict(x)
    return {
        "variable": target,
        "model_type": "decision_tree_regressor",
        "n_samples": int(y.shape[0]),
        "r_square_reported": float(r2_score(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "status": "ok",
    }


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(args.project_root).resolve()
    source_xlsx = resolve_source_xlsx(args.source_xlsx, project_root)
    dirs = ensure_dirs(project_root)
    logger = setup_logging(dirs["check"] / "seaad_patient_metadata_analysis.log")
    logger.info("SEA-AD patient metadata project root: %s", project_root)
    logger.info("Runtime Python interpreter: %s", sys.executable)
    logger.info("Official source page reference: %s", OFFICIAL_SOURCE_PAGE)

    copied_script = copy_self_to_project(dirs["scripts"])
    logger.info("Copied analysis script into project scripts directory: %s", copied_script)

    raw_df, copied_source = load_metadata(source_xlsx, dirs["output_raw"], logger)
    cleaned_df = clean_strings(raw_df)

    source_manifest = pd.DataFrame(
        [
            {
                "official_source_page": OFFICIAL_SOURCE_PAGE,
                "local_source_path": str(source_xlsx),
                "copied_source_path": str(copied_source),
                "source_md5": file_md5(copied_source),
                "n_rows": int(cleaned_df.shape[0]),
                "n_columns": int(cleaned_df.shape[1]),
            }
        ]
    )
    write_dual_table(source_manifest, "seaad_patient_source_manifest", dirs["csv"], dirs["tsv"])

    raw_df.to_csv(dirs["output_processed"] / "seaad_patient_metadata_original.csv", index=False)
    raw_df.to_csv(dirs["output_processed"] / "seaad_patient_metadata_original.tsv", index=False, sep="\t")
    cleaned_df.to_csv(dirs["output_processed"] / "seaad_patient_metadata_clean.csv", index=False)
    cleaned_df.to_csv(dirs["output_processed"] / "seaad_patient_metadata_clean.tsv", index=False, sep="\t")

    catalog = infer_variable_types(cleaned_df, numeric_threshold=args.numeric_threshold)
    write_dual_table(catalog, "seaad_patient_variable_catalog", dirs["csv"], dirs["tsv"])

    numeric_vars = catalog.loc[catalog["inferred_type"] == "numeric", "variable"].tolist()
    categorical_vars = catalog.loc[catalog["inferred_type"] == "categorical", "variable"].tolist()
    analysis_vars = catalog.loc[catalog["role"] == "analysis", "variable"].tolist()
    logger.info(
        "Detected %s analysis variables: %s numeric and %s categorical",
        len(analysis_vars),
        len(numeric_vars),
        len(categorical_vars),
    )

    encoded, encoding_manifest = one_hot_encode(cleaned_df[categorical_vars], categorical_vars)
    encoded_with_id = pd.concat([cleaned_df[list(ID_COLUMNS)], encoded], axis=1)
    encoded_with_id.to_csv(dirs["output_processed"] / "seaad_patient_metadata_onehot_dropfirst.csv", index=False)
    encoded_with_id.to_csv(dirs["output_processed"] / "seaad_patient_metadata_onehot_dropfirst.tsv", index=False, sep="\t")
    write_dual_table(encoding_manifest, "seaad_patient_onehot_manifest", dirs["csv"], dirs["tsv"])

    numeric_summary_rows: list[dict[str, Any]] = []
    numeric_model_rows: list[dict[str, Any]] = []
    for variable in numeric_vars:
        numeric_summary_rows.append(numeric_summary(cleaned_df[variable], variable))
        plot_numeric_distribution(cleaned_df[variable], variable, dirs["plots_numeric"] / f"{safe_slug(variable)}_t_distribution.png")
        numeric_model_rows.append(
            tree_regressor_diagnostics(
                cleaned_df,
                variable,
                numeric_vars=numeric_vars,
                categorical_vars=categorical_vars,
                seed=args.seed,
                max_depth=args.tree_max_depth,
                min_samples_leaf=args.min_samples_leaf,
            )
        )

    categorical_summary_rows: list[dict[str, Any]] = []
    categorical_level_tables: list[pd.DataFrame] = []
    categorical_model_rows: list[dict[str, Any]] = []
    dendrogram_status_rows: list[dict[str, Any]] = []
    for variable in categorical_vars:
        counts = categorical_counts(cleaned_df[variable], variable)
        categorical_level_tables.append(counts)
        categorical_summary_rows.append(categorical_summary(cleaned_df[variable], variable))
        plot_categorical_counts(counts, variable, dirs["plots_categorical"] / f"{safe_slug(variable)}_levels.png")
        predictors = build_predictor_matrix(cleaned_df, numeric_vars, categorical_vars, exclude=variable)
        dendrogram_status = plot_dendrogram_for_categorical(
            cleaned_df,
            target=variable,
            predictors=predictors,
            out_path=dirs["plots_dendrogram"] / f"{safe_slug(variable)}_dendrogram.png",
        )
        dendrogram_status_rows.append({"variable": variable, "dendrogram_status": dendrogram_status})
        categorical_model_rows.append(
            tree_classifier_diagnostics(
                cleaned_df,
                target=variable,
                numeric_vars=numeric_vars,
                categorical_vars=categorical_vars,
                plots_tree_dir=dirs["plots_tree"],
                plots_terminal_dir=dirs["plots_terminal"],
                csv_dir=dirs["csv"],
                tsv_dir=dirs["tsv"],
                seed=args.seed,
                max_depth=args.tree_max_depth,
                min_samples_leaf=args.min_samples_leaf,
            )
        )

    numeric_summary_df = pd.DataFrame(numeric_summary_rows).sort_values("variable")
    numeric_model_df = pd.DataFrame(numeric_model_rows).sort_values("variable")
    categorical_summary_df = pd.DataFrame(categorical_summary_rows).sort_values("variable")
    categorical_levels_df = pd.concat(categorical_level_tables, ignore_index=True) if categorical_level_tables else pd.DataFrame()
    categorical_model_df = pd.DataFrame(categorical_model_rows).sort_values("variable")
    dendrogram_status_df = pd.DataFrame(dendrogram_status_rows).sort_values("variable")

    write_dual_table(numeric_summary_df, "seaad_patient_numeric_summary", dirs["csv"], dirs["tsv"])
    write_dual_table(numeric_model_df, "seaad_patient_numeric_tree_metrics", dirs["csv"], dirs["tsv"])
    write_dual_table(categorical_summary_df, "seaad_patient_categorical_summary", dirs["csv"], dirs["tsv"])
    write_dual_table(categorical_levels_df, "seaad_patient_categorical_level_counts", dirs["csv"], dirs["tsv"])
    write_dual_table(categorical_model_df, "seaad_patient_categorical_tree_metrics", dirs["csv"], dirs["tsv"])
    write_dual_table(dendrogram_status_df, "seaad_patient_dendrogram_status", dirs["csv"], dirs["tsv"])

    overall_summary = pd.concat(
        [
            numeric_summary_df[["variable", "n", "missing"]].assign(variable_type="numeric"),
            categorical_summary_df[["variable", "n", "missing"]].assign(variable_type="categorical"),
        ],
        ignore_index=True,
    ).sort_values(["variable_type", "variable"])
    write_dual_table(overall_summary, "seaad_patient_overall_variable_summary", dirs["csv"], dirs["tsv"])

    output_manifest_rows = []
    for base_dir in [dirs["output"], dirs["results"], dirs["check"]]:
        for path in sorted(base_dir.rglob("*")):
            if path.is_file():
                output_manifest_rows.append(
                    {
                        "file_path": str(path),
                        "size_bytes": int(path.stat().st_size),
                    }
                )
    output_manifest = pd.DataFrame(output_manifest_rows)
    write_dual_table(output_manifest, "seaad_patient_output_manifest", dirs["csv"], dirs["tsv"])
    logger.info("Wrote %s output files into %s", output_manifest.shape[0], project_root)


if __name__ == "__main__":
    main()
