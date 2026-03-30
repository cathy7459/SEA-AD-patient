from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


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
_ensure_runtime_dependencies({"pandas": "pandas"})

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write the literature-based method selection summary for SEA-AD patient metadata association analysis.")
    parser.add_argument(
        "--patient-root",
        default=None,
        help="SEA-AD patient project root. If omitted, the script auto-detects a sensible local root.",
    )
    return parser


def resolve_patient_root(patient_root_arg: str | None) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    cwd = Path.cwd()
    candidates: list[Path] = []
    if patient_root_arg:
        candidates.append(Path(patient_root_arg))
    candidates.extend(
        [
            cwd,
            cwd / "SEA-AD-patient",
            repo_root,
            repo_root.parent / "SEA-AD-patient",
        ]
    )
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()
    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(f"Could not resolve patient-root. Searched:\n{searched}")


def main() -> None:
    args = build_parser().parse_args()
    patient_root = resolve_patient_root(args.patient_root)
    check_dir = patient_root / "check"
    check_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            {
                "rank": 1,
                "method_name": "Factor Analysis of Mixed Data",
                "primary_reference": "Pagès J. Analyse factorielle de données mixtes. Revue de Statistique Appliquée. 2004.",
                "reference_url": "https://eudml.org/doc/106558",
                "why_selected": "Designed for data tables that mix quantitative and qualitative variables, and directly yields interpretable mixed-variable relationship measures.",
                "strength_for_seaad_patient_metadata": "Good for global structure, cluster-like variable groupings, and interpretable variable-level association screening.",
                "main_output_focus": "Mixed relationship matrix, top variable pairs, sample map, feature loadings.",
            },
            {
                "rank": 2,
                "method_name": "Normalized Mutual Information Network",
                "primary_reference": "Steuer R et al. The mutual information: detecting and evaluating dependencies between variables. Bioinformatics. 2002.",
                "reference_url": "https://academic.oup.com/bioinformatics/article/18/suppl_2/S231/261423",
                "why_selected": "Captures nonlinear and non-monotone dependencies that ordinary correlation misses and works naturally on mixed metadata after deterministic discretization.",
                "strength_for_seaad_patient_metadata": "Good for edge ranking, association networks, and detecting distributional dependence beyond linear trends.",
                "main_output_focus": "NMI matrix, top edges, permutation-tested pairs, network plot.",
            },
        ]
    )
    df.to_csv(check_dir / "seaad_patient_association_method_selection.csv", index=False)
    df.to_csv(check_dir / "seaad_patient_association_method_selection.tsv", index=False, sep="\t")


if __name__ == "__main__":
    main()
