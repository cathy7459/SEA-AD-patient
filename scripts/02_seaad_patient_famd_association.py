from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _prepend_python_script_dir_to_path() -> None:
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
    repo_root = Path(__file__).resolve().parents[1]
    state_path = repo_root / ".runtime" / "seaad_patient_dependency_state.json"
    installed_state = _load_installed_state(state_path)

    missing_packages: list[str] = []
    for module_name, package_name in requirements.items():
        if importlib.util.find_spec(module_name) is not None:
            installed_state[module_name] = package_name
        else:
            missing_packages.append(package_name)

    missing_packages = sorted(set(missing_packages))
    if missing_packages:
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
    }
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA


MISSING_LABEL = "Missing"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FAMD-style mixed association analysis on SEA-AD patient metadata."
    )
    parser.add_argument("--patient-root", required=True)
    parser.add_argument("--method-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def ensure_dirs(method_root: Path) -> dict[str, Path]:
    dirs = {
        "method_root": method_root,
        "scripts": method_root / "scripts",
        "output": method_root / "output",
        "results": method_root / "results",
        "plots": method_root / "results" / "plots",
        "csv": method_root / "results" / "csv",
        "tsv": method_root / "results" / "tsv",
        "check": method_root / "check",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"famd_{log_path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def write_tables(df: pd.DataFrame, stem: str, csv_dir: Path, tsv_dir: Path) -> None:
    df.to_csv(csv_dir / f"{stem}.csv", index=False)
    df.to_csv(tsv_dir / f"{stem}.tsv", index=False, sep="\t")


def copy_script(target_dir: Path) -> None:
    src = Path(__file__).resolve()
    (target_dir / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def load_inputs(patient_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_df = pd.read_csv(patient_root / "output" / "processed" / "seaad_patient_metadata_clean.tsv", sep="\t")
    catalog = pd.read_csv(patient_root / "results" / "tsv" / "seaad_patient_variable_catalog.tsv", sep="\t")
    return clean_df, catalog


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


def prepare_categorical(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(MISSING_LABEL)


def eta_squared(categories: pd.Series, values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna() & categories.notna()
    categories = categories[valid]
    numeric = numeric[valid]
    if numeric.shape[0] < 3 or categories.nunique() < 2:
        return np.nan
    grand_mean = numeric.mean()
    total_ss = float(((numeric - grand_mean) ** 2).sum())
    if total_ss == 0:
        return 0.0
    between_ss = 0.0
    for _, group in numeric.groupby(categories):
        between_ss += float(group.shape[0] * (group.mean() - grand_mean) ** 2)
    return between_ss / total_ss


def cramers_v_squared(a: pd.Series, b: pd.Series) -> float:
    a = prepare_categorical(a)
    b = prepare_categorical(b)
    table = pd.crosstab(a, b)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan
    chi2, _, _, _ = chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    if n == 0:
        return np.nan
    phi2 = chi2 / n
    denom = min(table.shape[0] - 1, table.shape[1] - 1)
    if denom <= 0:
        return np.nan
    return phi2 / denom


def relationship_value(
    df: pd.DataFrame,
    left: str,
    right: str,
    numeric_vars: set[str],
    categorical_vars: set[str],
) -> float:
    if left in numeric_vars and right in numeric_vars:
        x = pd.to_numeric(df[left], errors="coerce")
        y = pd.to_numeric(df[right], errors="coerce")
        valid = x.notna() & y.notna()
        return float(x[valid].corr(y[valid]) ** 2) if valid.sum() >= 3 else np.nan
    if left in categorical_vars and right in categorical_vars:
        return cramers_v_squared(df[left], df[right])
    cat = left if left in categorical_vars else right
    num = right if left in categorical_vars else left
    return eta_squared(prepare_categorical(df[cat]), df[num])


def pair_chunks(pairs: list[tuple[str, str]], batch_size: int) -> list[list[tuple[str, str]]]:
    size = max(1, min(batch_size, 8))
    return [pairs[i : i + size] for i in range(0, len(pairs), size)]


def compute_relationship_chunk(
    chunk: list[tuple[str, str]],
    df: pd.DataFrame,
    numeric_vars: set[str],
    categorical_vars: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in chunk:
        rows.append(
            {
                "variable_1": left,
                "variable_2": right,
                "association_score": relationship_value(df, left, right, numeric_vars, categorical_vars),
            }
        )
    return rows


def build_relationship_matrix(
    df: pd.DataFrame,
    numeric_vars: list[str],
    categorical_vars: list[str],
    max_workers: int,
    batch_size: int,
) -> pd.DataFrame:
    variables = numeric_vars + categorical_vars
    matrix = pd.DataFrame(np.eye(len(variables)), index=variables, columns=variables)
    pairs = [(left, right) for i, left in enumerate(variables) for right in variables[i + 1 :]]
    chunks = pair_chunks(pairs, batch_size)
    numeric_set = set(numeric_vars)
    categorical_set = set(categorical_vars)
    rows: list[dict[str, Any]] = []
    workers = max(1, min(max_workers, 8))
    if workers == 1:
        for chunk in chunks:
            rows.extend(compute_relationship_chunk(chunk, df, numeric_set, categorical_set))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(compute_relationship_chunk, chunk, df, numeric_set, categorical_set)
                for chunk in chunks
            ]
            for future in futures:
                rows.extend(future.result())
    for row in rows:
        matrix.loc[row["variable_1"], row["variable_2"]] = row["association_score"]
        matrix.loc[row["variable_2"], row["variable_1"]] = row["association_score"]
    return matrix


def build_famd_matrix(df: pd.DataFrame, numeric_vars: list[str], categorical_vars: list[str]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    blocks: list[pd.DataFrame] = []
    manifest: dict[str, list[str]] = {"numeric": [], "categorical": []}
    if numeric_vars:
        numeric = df[numeric_vars].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.fillna(numeric.median())
        numeric = (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, 1)
        numeric.columns = [f"num::{column}" for column in numeric.columns]
        manifest["numeric"] = numeric.columns.tolist()
        blocks.append(numeric)
    for column in categorical_vars:
        prepared = prepare_categorical(df[column])
        dummies = pd.get_dummies(prepared, prefix=f"cat::{column}", dtype=float)
        proportions = dummies.mean(axis=0)
        weighted = dummies.sub(proportions, axis=1).div(np.sqrt(proportions.replace(0, np.nan)), axis=1).fillna(0.0)
        weighted = weighted / np.sqrt(max(dummies.shape[1], 1))
        manifest["categorical"].extend(weighted.columns.tolist())
        blocks.append(weighted)
    matrix = pd.concat(blocks, axis=1)
    return matrix, manifest


def top_pairs_from_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cols = matrix.columns.tolist()
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            rows.append({"variable_1": left, "variable_2": right, "association_score": matrix.loc[left, right]})
    return pd.DataFrame(rows).sort_values("association_score", ascending=False, na_position="last")


def plot_heatmap(matrix: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(matrix, cmap="viridis", vmin=0, vmax=1, ax=ax)
    ax.set_title("FAMD-style mixed relationship matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_scatter(coords: pd.DataFrame, color_series: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_df = coords.copy()
    plot_df["color"] = prepare_categorical(color_series)
    sns.scatterplot(data=plot_df, x="Dim1", y="Dim2", hue="color", s=70, alpha=0.85, ax=ax)
    ax.set_title("FAMD sample map")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", title=color_series.name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_contributions(contrib: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    head = contrib.nlargest(top_n, "Dim1_abs_loading").copy()
    fig, ax = plt.subplots(figsize=(9, max(6, 0.35 * head.shape[0])))
    sns.barplot(data=head, x="Dim1_abs_loading", y="feature", color="#3182bd", ax=ax)
    ax.set_title("Top FAMD feature loadings on Dim1")
    ax.set_xlabel("Absolute loading")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    patient_root = Path(args.patient_root).resolve()
    method_root = Path(args.method_root).resolve()
    dirs = ensure_dirs(method_root)
    logger = setup_logger(dirs["check"] / "factor_analysis_of_mixed_data.log")
    copy_script(dirs["scripts"])
    logger.info("Runtime Python interpreter: %s", sys.executable)

    df, catalog = load_inputs(patient_root)
    numeric_vars = catalog.loc[(catalog["inferred_type"] == "numeric") & (catalog["role"] == "analysis"), "variable"].tolist()
    categorical_vars = catalog.loc[(catalog["inferred_type"] == "categorical") & (catalog["role"] == "analysis"), "variable"].tolist()
    workers = max(1, min(args.max_workers, 8))
    batch_size = max(1, min(args.batch_size, 8))
    logger.info(
        "Loaded patient metadata: %s rows, %s numeric variables, %s categorical variables | workers=%s | batch_size=%s",
        df.shape[0],
        len(numeric_vars),
        len(categorical_vars),
        workers,
        batch_size,
    )

    relationship = build_relationship_matrix(df, numeric_vars, categorical_vars, max_workers=workers, batch_size=batch_size)
    write_tables(relationship.reset_index(names="variable"), "famd_relationship_matrix", dirs["csv"], dirs["tsv"])

    top_pairs = top_pairs_from_matrix(relationship)
    write_tables(top_pairs, "famd_top_variable_pairs", dirs["csv"], dirs["tsv"])

    famd_matrix, manifest = build_famd_matrix(df, numeric_vars, categorical_vars)
    pca = PCA(n_components=min(args.n_components, famd_matrix.shape[1], famd_matrix.shape[0]), random_state=args.seed)
    coords = pca.fit_transform(famd_matrix.to_numpy())
    coord_df = pd.DataFrame(coords[:, :2], columns=["Dim1", "Dim2"])
    coord_df.insert(0, "Donor ID", df["Donor ID"])
    write_tables(coord_df, "famd_sample_coordinates", dirs["csv"], dirs["tsv"])

    explained = pd.DataFrame(
        {
            "component": [f"Dim{i+1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )
    write_tables(explained, "famd_explained_variance", dirs["csv"], dirs["tsv"])

    loadings = pd.DataFrame(
        pca.components_.T,
        index=famd_matrix.columns,
        columns=[f"Dim{i+1}" for i in range(pca.components_.shape[0])],
    ).reset_index(names="feature")
    loadings["Dim1_abs_loading"] = loadings["Dim1"].abs()
    write_tables(loadings, "famd_feature_loadings", dirs["csv"], dirs["tsv"])

    method_info = pd.DataFrame(
        [
            {
                "method_name": "Factor Analysis of Mixed Data",
                "primary_reference": "Pagès J. Analyse factorielle de données mixtes. Revue de Statistique Appliquée. 2004.",
                "reference_url": "https://eudml.org/doc/106558",
                "applied_example_reference": "Han L et al. Exploring the Clinical Characteristics of COVID-19 Clusters Identified Using FAMD-Based Cluster Analysis. Front Med. 2021.",
                "applied_example_url": "https://pubmed.ncbi.nlm.nih.gov/34336871/",
                "rationale": "FAMD is designed for tables containing both quantitative and qualitative patient variables, and its relationship matrix directly combines r^2, eta^2, and phi^2/Cramer's V^2 style associations.",
                "n_numeric_variables": len(numeric_vars),
                "n_categorical_variables": len(categorical_vars),
            }
        ]
    )
    write_tables(method_info, "famd_method_summary", dirs["csv"], dirs["tsv"])

    plot_heatmap(relationship, dirs["plots"] / "famd_relationship_matrix_heatmap.png")
    color_var = "Cognitive Status" if "Cognitive Status" in df.columns else categorical_vars[0]
    plot_scatter(coord_df[["Dim1", "Dim2"]], df[color_var], dirs["plots"] / "famd_sample_map.png")
    plot_contributions(loadings, dirs["plots"] / "famd_top_dim1_loadings.png")

    manifest_df = pd.DataFrame(
        [
            {"feature_group": "numeric", "n_features": len(manifest["numeric"])},
            {"feature_group": "categorical", "n_features": len(manifest["categorical"])},
        ]
    )
    write_tables(manifest_df, "famd_feature_manifest", dirs["csv"], dirs["tsv"])
    logger.info("Saved FAMD outputs to %s", method_root)


if __name__ == "__main__":
    main()
