from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import shutil
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
        "networkx": "networkx",
        "numpy": "numpy",
        "pandas": "pandas",
        "seaborn": "seaborn",
        "sklearn": "scikit-learn",
    }
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import mutual_info_score


MISSING_LABEL = "Missing"
REQUIRED_INPUT_RELATIVE_PATHS = [
    Path("output") / "processed" / "seaad_patient_metadata_clean.tsv",
    Path("results") / "tsv" / "seaad_patient_variable_catalog.tsv",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run normalized mutual information network analysis on SEA-AD patient metadata."
    )
    parser.add_argument(
        "--patient-root",
        default=None,
        help="SEA-AD patient project root. If omitted, the script auto-detects a root with required inputs.",
    )
    parser.add_argument(
        "--method-root",
        default=None,
        help="Output root for MI results. Default: <patient-root>/Mutual_Information_Network",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=250)
    parser.add_argument("--top-edges", type=int, default=25)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _candidate_patient_roots(patient_root_arg: str | None) -> list[Path]:
    script_repo_root = Path(__file__).resolve().parents[1]
    cwd = Path.cwd()
    candidates: list[Path] = []
    if patient_root_arg:
        candidates.append(Path(patient_root_arg))
    candidates.extend(
        [
            cwd,
            cwd / "SEA-AD-patient",
            script_repo_root,
            script_repo_root.parent / "SEA-AD-patient",
        ]
    )
    return _unique_paths(candidates)


def _missing_required_inputs(patient_root: Path) -> list[Path]:
    return [rel for rel in REQUIRED_INPUT_RELATIVE_PATHS if not (patient_root / rel).exists()]


def resolve_roots(patient_root_arg: str | None, method_root_arg: str | None) -> tuple[Path, Path]:
    existing_candidates = [path.resolve() for path in _candidate_patient_roots(patient_root_arg) if path.exists()]
    for candidate in existing_candidates:
        if not _missing_required_inputs(candidate):
            patient_root = candidate
            break
    else:
        searched = existing_candidates if existing_candidates else _candidate_patient_roots(patient_root_arg)
        details: list[str] = []
        for candidate in searched:
            missing = _missing_required_inputs(candidate) if candidate.exists() else REQUIRED_INPUT_RELATIVE_PATHS
            missing_text = ", ".join(str(rel) for rel in missing)
            details.append(f"- {candidate} | missing: {missing_text}")
        raise FileNotFoundError(
            "Could not resolve a valid patient-root for mutual information analysis. "
            "Required input files were not found.\n"
            + "\n".join(details)
        )

    if method_root_arg:
        method_root = Path(method_root_arg).resolve()
    else:
        method_root = (patient_root / "Mutual_Information_Network").resolve()
    return patient_root, method_root


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
    logger = logging.getLogger(f"mi_{log_path.parent.name}")
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
    shutil.copy2(src, target_dir / src.name)


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


def discretize_numeric(series: pd.Series, n_bins: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    filled = numeric.fillna(numeric.median())
    unique_vals = filled.nunique()
    bins = min(n_bins, max(2, unique_vals))
    if unique_vals < 2:
        return pd.Series(["constant"] * len(series), index=series.index, name=series.name)
    try:
        binned = pd.qcut(filled, q=bins, duplicates="drop")
    except ValueError:
        binned = pd.cut(filled, bins=bins, duplicates="drop")
    return binned.astype("string").fillna(MISSING_LABEL)


def entropy(labels: pd.Series) -> float:
    probs = labels.value_counts(normalize=True)
    return float(-(probs * np.log(probs)).sum())


def normalized_mutual_information(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    mi = mutual_info_score(x, y)
    hx = entropy(x)
    hy = entropy(y)
    denom = math.sqrt(max(hx * hy, 1e-12))
    nmi = float(mi / denom) if denom > 0 else np.nan
    return float(mi), nmi


def permutation_pvalue(x: pd.Series, y: pd.Series, observed_nmi: float, n_permutations: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    greater_or_equal = 1
    y_values = y.to_numpy(copy=True)
    for _ in range(n_permutations):
        permuted = pd.Series(rng.permutation(y_values), index=y.index)
        _, perm_nmi = normalized_mutual_information(x, permuted)
        if perm_nmi >= observed_nmi:
            greater_or_equal += 1
    return greater_or_equal / (n_permutations + 1)


def pair_chunks(pairs: list[tuple[str, str]], batch_size: int) -> list[list[tuple[str, str]]]:
    size = max(1, min(batch_size, 8))
    return [pairs[i : i + size] for i in range(0, len(pairs), size)]


def compute_mi_chunk(
    chunk: list[tuple[str, str]],
    encoded: pd.DataFrame,
    n_permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, (left, right) in enumerate(chunk):
        mi, nmi = normalized_mutual_information(encoded[left], encoded[right])
        p_value = permutation_pvalue(encoded[left], encoded[right], nmi, n_permutations, seed + offset)
        rows.append(
            {
                "variable_1": left,
                "variable_2": right,
                "mutual_information": mi,
                "nmi": nmi,
                "permutation_p_value": p_value,
            }
        )
    return rows


def load_inputs(patient_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_df = pd.read_csv(patient_root / "output" / "processed" / "seaad_patient_metadata_clean.tsv", sep="\t")
    catalog = pd.read_csv(patient_root / "results" / "tsv" / "seaad_patient_variable_catalog.tsv", sep="\t")
    return clean_df, catalog


def top_pairs_from_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = matrix.columns.tolist()
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            rows.append({"variable_1": left, "variable_2": right, "nmi": matrix.loc[left, right]})
    return pd.DataFrame(rows).sort_values("nmi", ascending=False, na_position="last")


def plot_heatmap(matrix: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(matrix, cmap="magma", vmin=0, vmax=1, ax=ax)
    ax.set_title("Normalized mutual information matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_network(edges: pd.DataFrame, out_path: Path, seed: int) -> None:
    if edges.empty:
        return
    graph = nx.Graph()
    for _, row in edges.iterrows():
        graph.add_edge(row["variable_1"], row["variable_2"], weight=row["nmi"])
    pos = nx.spring_layout(graph, seed=seed, weight="weight")
    fig, ax = plt.subplots(figsize=(12, 10))
    widths = [2 + 6 * graph[u][v]["weight"] for u, v in graph.edges()]
    nx.draw_networkx_edges(graph, pos, width=widths, alpha=0.5, edge_color="#6baed6", ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_color="#2171b5", node_size=900, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=8, font_color="white", ax=ax)
    ax.set_title("Top variable association network")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_top_pairs(edges: pd.DataFrame, out_path: Path) -> None:
    head = edges.head(20).copy()
    head["pair"] = head["variable_1"] + " <> " + head["variable_2"]
    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * head.shape[0])))
    sns.barplot(data=head, x="nmi", y="pair", color="#f16913", ax=ax)
    ax.set_title("Top normalized mutual information pairs")
    ax.set_xlabel("Normalized mutual information")
    ax.set_ylabel("Variable pair")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    patient_root, method_root = resolve_roots(args.patient_root, args.method_root)
    dirs = ensure_dirs(method_root)
    logger = setup_logger(dirs["check"] / "mutual_information_network.log")
    copy_script(dirs["scripts"])
    logger.info("Runtime Python interpreter: %s", sys.executable)
    logger.info("Resolved patient-root: %s", patient_root)
    logger.info("Resolved method-root: %s", method_root)

    df, catalog = load_inputs(patient_root)
    variables = catalog.loc[catalog["role"] == "analysis", "variable"].tolist()
    numeric_vars = set(catalog.loc[(catalog["inferred_type"] == "numeric") & (catalog["role"] == "analysis"), "variable"].tolist())
    workers = max(1, min(args.max_workers, 8))
    batch_size = max(1, min(args.batch_size, 8))
    logger.info(
        "Loaded patient metadata: %s rows and %s analysis variables | workers=%s | batch_size=%s | permutations=%s",
        df.shape[0],
        len(variables),
        workers,
        batch_size,
        args.n_permutations,
    )

    encoded = pd.DataFrame(index=df.index)
    encoding_rows: list[dict[str, Any]] = []
    for variable in variables:
        if variable in numeric_vars:
            transformed = discretize_numeric(df[variable], args.n_bins)
            enc_type = "numeric_discretized"
        else:
            transformed = prepare_categorical(df[variable])
            enc_type = "categorical"
        encoded[variable] = transformed.astype("string")
        encoding_rows.append(
            {
                "variable": variable,
                "encoding_type": enc_type,
                "n_levels_after_encoding": int(encoded[variable].nunique(dropna=False)),
            }
        )
    write_tables(pd.DataFrame(encoding_rows), "mi_encoding_manifest", dirs["csv"], dirs["tsv"])
    encoded.to_csv(dirs["output"] / "mi_encoded_metadata.tsv", sep="\t", index=False)

    matrix = pd.DataFrame(np.eye(len(variables)), index=variables, columns=variables)
    pairs = [(left, right) for i, left in enumerate(variables) for right in variables[i + 1 :]]
    chunks = pair_chunks(pairs, batch_size)
    pair_rows: list[dict[str, Any]] = []
    if workers == 1:
        for chunk_index, chunk in enumerate(chunks):
            pair_rows.extend(compute_mi_chunk(chunk, encoded, args.n_permutations, args.seed + 1000 * chunk_index))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    compute_mi_chunk,
                    chunk,
                    encoded,
                    args.n_permutations,
                    args.seed + 1000 * chunk_index,
                )
                for chunk_index, chunk in enumerate(chunks)
            ]
            for future in futures:
                pair_rows.extend(future.result())
    for row in pair_rows:
        matrix.loc[row["variable_1"], row["variable_2"]] = row["nmi"]
        matrix.loc[row["variable_2"], row["variable_1"]] = row["nmi"]
    pair_df = pd.DataFrame(pair_rows).sort_values(["nmi", "permutation_p_value"], ascending=[False, True])
    write_tables(matrix.reset_index(names="variable"), "mi_nmi_matrix", dirs["csv"], dirs["tsv"])
    write_tables(pair_df, "mi_pairwise_associations", dirs["csv"], dirs["tsv"])

    significant = pair_df.loc[pair_df["permutation_p_value"] <= 0.05].copy()
    write_tables(significant, "mi_significant_pairs", dirs["csv"], dirs["tsv"])

    top_edges = pair_df.head(args.top_edges).copy()
    write_tables(top_edges, "mi_top_edges", dirs["csv"], dirs["tsv"])

    plot_heatmap(matrix, dirs["plots"] / "mi_nmi_heatmap.png")
    plot_top_pairs(pair_df, dirs["plots"] / "mi_top_pairs.png")
    plot_network(top_edges, dirs["plots"] / "mi_top_network.png", args.seed)

    method_info = pd.DataFrame(
        [
            {
                "method_name": "Normalized Mutual Information Network",
                "primary_reference": "Steuer R et al. The mutual information: detecting and evaluating dependencies between variables. Bioinformatics. 2002.",
                "reference_url": "https://academic.oup.com/bioinformatics/article/18/suppl_2/S231/261423",
                "applied_example_reference": "de Matos Simoes R et al. Mining Mutational Processes in Cancer by Learning a Score-Based Mutational Interaction Network Model. NPJ Syst Biol Appl. 2022.",
                "applied_example_url": "https://www.nature.com/articles/s41540-022-00275-8",
                "rationale": "Normalized mutual information is robust to nonlinear and non-monotone dependencies and can be applied to mixed metadata once continuous variables are discretized in a reproducible way.",
                "n_variables": len(variables),
                "n_permutations": args.n_permutations,
            }
        ]
    )
    write_tables(method_info, "mi_method_summary", dirs["csv"], dirs["tsv"])
    logger.info("Saved mutual information outputs to %s", method_root)


if __name__ == "__main__":
    main()
