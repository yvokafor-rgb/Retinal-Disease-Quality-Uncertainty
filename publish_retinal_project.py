#!/usr/bin/env python3
"""Organize an RFMiD + EyeQ research submission, optionally publish using GitHub CLI.

Run from the VS Code project root:
  python publish_retinal_project.py
  python publish_retinal_project.py --publish --visibility public

Requires Git + GitHub CLI (gh) only for --publish; no Python packages needed.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

OWNER = "yvokafor-rgb"
DEFAULT_REPO = "Retinal-Disease-Quality-Uncertainty"
MAX_SIZE_MB = 20
MAX_BYTES = MAX_SIZE_MB * 1024 * 1024
RESULT_EXTENSIONS = {".csv", ".tsv", ".json", ".txt", ".md", ".png", ".jpg", ".jpeg", ".svg", ".pdf", ".html", ".xlsx"}
SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints",
    "raw_data", "raw_images", "images", "image", "datasets", "data",
    "training_set", "evaluation_set", "test_set", "cache", "cached",
    "checkpoints", "weights", "processed_datasets", "preprocessing_cache",
    "cropped_images", "padded_images", "original_images", "model_weights",
}
SENSITIVE_NAME = re.compile(r"(^\.(env|npmrc|pypirc)$|(^|[._-])(secret|credential|password|api[_-]?key|access[_-]?token|private[_-]?key)([._-]|$)|^id_rsa$)", re.I)
SENSITIVE_CONTENT = re.compile(
    r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{24,}|"
    r"sk-[A-Za-z0-9_-]{24,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"(?i:(?:api[_-]?key|access[_-]?token|secret[_-]?key|password))\s*=\s*['\"][^'\"\n]{8,}['\"]"
)
TEXT_EXTENSIONS = {".py", ".ipynb", ".json", ".csv", ".tsv", ".md", ".txt", ".html", ".yaml", ".yml", ".toml", ".log", ".ps1"}


class Packager:
    def __init__(self, root: Path, dest: Path):
        self.root, self.dest = root, dest
        self.files: list[str] = []
        self.skipped: list[str] = []

    def copy(self, source: Path, relative: Path) -> None:
        if not source.is_file():
            return
        label = relative.as_posix()
        if SENSITIVE_NAME.search(source.name):
            self.skipped.append(f"{label} (sensitive filename)")
            return
        if source.stat().st_size > MAX_BYTES:
            self.skipped.append(f"{label} (larger than {MAX_SIZE_MB} MiB)")
            return
        if source.suffix.lower() in TEXT_EXTENSIONS:
            try:
                text = source.read_bytes().decode("utf-8", errors="replace")
                if SENSITIVE_CONTENT.search(text):
                    self.skipped.append(f"{label} (possible embedded credential; review source)")
                    return
            except OSError as exc:
                self.skipped.append(f"{label} (unreadable: {exc})")
                return
        target = self.dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        self.files.append(label)

    def copy_results(self, source: Path, result_name: str) -> None:
        if not source.is_dir():
            return
        for folder, dirs, filenames in __import__("os").walk(source):
            dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS]
            base = Path(folder)
            for filename in filenames:
                file = base / filename
                if file.suffix.lower() not in RESULT_EXTENSIONS:
                    continue
                if any(s in file.name.lower() for s in ("manifest", "full_dataset", "raw_labels")):
                    continue
                self.copy(file, Path("results") / result_name / file.relative_to(source))


def content_readme(has_word: bool, has_final_test: bool) -> str:
    word_link = "- [Word research results summary](docs/Research_Results_Summary.docx)\n" if has_word else "- Word report: add `Research_Results_Summary.docx` to `docs/` if available.\n"
    final_line = (
        "**Official RFMiD test results:** Saved files were included under `results/m1_m2_m3_research/final_test_clean/` if present. Review those tables for the final independent evaluation."
        if has_final_test else
        "**Official RFMiD test results:** No completed official-test export was found at packaging time; the numbers below are **not test-set results**."
    )
    return f"""# Quality-Aware and Uncertainty-Guided Retinal Disease Recognition

**Research code and experiment results | RFMiD + EyeQ | Graduate computer vision project**

This repository documents a comparative retinal fundus-image study across three systems:

- **M1:** Baseline multi-label retinal disease recognition (selected Swin-T configuration).
- **M2:** Quality-aware disease prediction using an EyeQ image-quality classifier and a fusion head, compared against an **M2_no_quality** ablation.
- **M3:** Calibrated predictions with uncertainty-based referral / selective prediction.

## Start here (for the professor)

{word_link}- [Experiment outputs, tables, and figures](results/)
- [Python implementation](src/)
- [Data and checkpoint availability](data/README.md)
- [Repository file inventory](FILES_INCLUDED.md)

The numeric outputs shown in terminal windows are **not automatically embedded in Python scripts**. The `results/` folder contains the saved results available when this repository was assembled. Existing terminal logs are copied into `logs/` where available.

## Data split and experimental protocol

- **RFMiD:** 1,920 training images; 640 validation images (448 validation-selection, 192 validation-calibration); 640 official test images.
- **Selected disease labels:** DR, MH, ODC, TSLN, DN, MYA, ARMD.
- **EyeQ training manifest:** 10,035 train and 2,508 validation image records (quality labels: good, usable, reject).
- Model selection and comparisons on validation-selection data are **not independent test estimates**.

## Reported clean validation-selection results

| Model | Macro AP | Macro F1 | Macro specificity | Macro sensitivity / recall | Coverage |
|---|---:|---:|---:|---:|---:|
| M1 | 0.684373 | 0.648877 | 0.950753 | 0.677548 | 100% |
| M2_no_quality | 0.697922 | 0.653043 | 0.953102 | 0.687142 | 100% |
| M2 | 0.697363 | 0.649428 | 0.953102 | 0.681189 | 100% |
| M3 | 0.697557 | 0.639907 | 0.942979 | 0.679937 | 100% |
| M3_retained | 0.717114 | 0.651013 | 0.964176 | 0.688240 | 77.23% |

**Ablation:** M2 - M2_no_quality macro AP = -0.00056. M2 did not outperform the no-quality ablation on this metric; a benefit specifically caused by EyeQ quality features has not been established.

**Calibration:** ECE decreased from 0.04439 to 0.03639 **on the 192-image calibration subset** used to fit the calibration. Independent test confirmation is required.

**Referral:** M3 accepted about 77.23% of selection images. Retained-image scores are evaluated on a selected subset, so they should not be compared to full-coverage scores as though the test populations were identical.

**EyeQ quality classifier:** The reported validation macro F1 after epoch 8 was 0.8246, with balanced accuracy 0.8399. The checkpoint actually selected should be verified from the saved metadata.

## Reported robustness experiment: severity 2

Macro average precision on **validation-selection** images under synthetic image degradations:

| Degradation | M1 | M2_no_quality | M2 | M3 | M3_retained |
|---|---:|---:|---:|---:|---:|
| Blur | 0.6828 | 0.6956 | 0.6957 | 0.6957 | 0.7225 |
| Motion blur | 0.6825 | 0.6976 | 0.6979 | 0.6979 | 0.7262 |
| Brightness | 0.5410 | 0.5471 | 0.5474 | 0.5478 | 0.5532 |
| Contrast | 0.6380 | 0.6464 | 0.6462 | 0.6464 | 0.6598 |
| Glare | 0.6573 | 0.6657 | 0.6661 | 0.6667 | 0.6765 |
| Noise | 0.6799 | 0.6869 | 0.6870 | 0.6873 | 0.6890 |
| JPEG | 0.6607 | 0.6701 | 0.6700 | 0.6696 | 0.7124 |

Severe brightness changes produced the largest observed AP drop. Differences between M2 and M2_no_quality are small, and retained-image performance must be considered alongside coverage.

## Official held-out evaluation

{final_line}

## Repository structure

```text
src/          Python project scripts
notebooks/    Available Jupyter notebooks (if any)
docs/         Research summary files (if present)
results/      Saved clean, per-disease, calibration, and robustness outputs
logs/         Saved terminal logs (if present)
data/         Dataset and checkpoint access notes (no image data)
```

## Reproduction notes

1. Create a Python environment, install the dependencies used in the scripts, and obtain RFMiD and EyeQ from their authorized sources.
2. Check `src/preprocessing.py`, `src/model_training.py`, and `src/eyeq_quality.py` for preprocessing and training entry points.
3. See `src/model_deployment.py` for fitting and evaluating M1/M2/M3.
4. Original image datasets and large `.pt`/`.pth` checkpoints are intentionally **not uploaded**. To reproduce exact inference, obtain the frozen checkpoints separately.
5. The existing RFMiD final test is intended as a one-time held-out evaluation. Do not rerun it or tune against its results merely to produce a GitHub submission.

## Limitations

This repository packages **saved project artifacts**, not a new experiment. Metrics above are transcribed from the completed terminal runs and should be checked against the corresponding exported CSVs where present. Results on reused validation-selection data may be optimistic; significance, generalizability, and degradation-specific referral benefits require held-out evaluation.
"""


def run_checked(cmd: list[str], cwd: Path, capture: bool = False, allow_error: bool = False) -> subprocess.CompletedProcess[str]:
    p = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE if capture else None, stderr=subprocess.PIPE if capture else None)
    if p.returncode != 0 and not allow_error:
        details = (p.stderr or "").strip() if capture else "See command output above."
        raise RuntimeError(f"Command failed ({' '.join(cmd)}): {details}")
    return p


def publish(stage: Path, repo: str, visibility: str) -> None:
    for exe in ("git", "gh"):
        if not shutil.which(exe):
            raise RuntimeError(f"{exe} is not installed or not available in PATH. Install Git and GitHub CLI, then reopen VS Code.")
    identity = run_checked(["gh", "api", "user", "--jq", ".login"], cwd=stage, capture=True)
    login = identity.stdout.strip()
    if login.lower() != OWNER.lower():
        raise RuntimeError(f"GitHub CLI is authenticated as '{login}', not '{OWNER}'. Run: gh auth login")
    for key in ("user.name", "user.email"):
        value = run_checked(["git", "config", "--get", key], cwd=stage, capture=True, allow_error=True)
        if value.returncode != 0 or not value.stdout.strip():
            raise RuntimeError(f"Git {key} is not configured. Set it using: git config --global {key} \"YOUR_VALUE\"")

    full_name = f"{OWNER}/{repo}"
    expected_url = f"https://github.com/{full_name}"
    remote = run_checked(["git", "remote", "get-url", "origin"], cwd=stage, capture=True, allow_error=True)
    if remote.returncode == 0:
        remote_url = remote.stdout.strip().rstrip("/").removesuffix(".git")
        if not remote_url.lower().endswith(("/" + full_name).lower()) and not remote_url.lower().endswith((":" + full_name).lower()):
            raise RuntimeError(f"Existing 'origin' points to another repository: {remote_url}. Not pushing.")
    exists = run_checked(["gh", "repo", "view", full_name, "--json", "name"], cwd=stage, capture=True, allow_error=True).returncode == 0
    if exists and remote.returncode != 0:
        raise RuntimeError(f"{expected_url} already exists, but this package is not linked to it. Choose a new --repo name or connect your existing repository manually.")
    if visibility == "public":
        print("\nWARNING: A PUBLIC repository can be viewed by anyone, not only your professor.")
        check = input(f"Type PUBLIC to publish {expected_url} publicly: ").strip()
        if check != "PUBLIC":
            print("Publication cancelled; local files were prepared successfully.")
            return
    else:
        check = input(f"Type PUBLISH to create/update your PRIVATE repository {expected_url}: ").strip()
        if check != "PUBLISH":
            print("Publication cancelled; local files were prepared successfully.")
            return

    if not (stage / ".git").exists():
        run_checked(["git", "init", "-b", "main"], cwd=stage)
    run_checked(["git", "add", "-A"], cwd=stage)
    has_changes = bool(run_checked(["git", "status", "--porcelain"], cwd=stage, capture=True).stdout.strip())
    if has_changes:
        run_checked(["git", "commit", "-m", "Add retinal disease research code and experimental results"], cwd=stage)
    run_checked(["gh", "auth", "setup-git"], cwd=stage)
    if not exists:
        run_checked(["gh", "repo", "create", full_name, f"--{visibility}", "--source", ".", "--remote", "origin", "--push", "--description", "RFMiD + EyeQ retinal disease classification, quality ablation, uncertainty, and robustness research"], cwd=stage)
    else:
        run_checked(["git", "push", "-u", "origin", "main"], cwd=stage)
    print(f"\nSUCCESS: {expected_url}")
    if visibility == "private":
        print("Private repository: invite your professor in GitHub Settings > Collaborators before sending the link.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Package an RFMiD research project for GitHub and optionally publish it")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Folder containing model_deployment.py")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Repository name under the yvokafor-rgb account")
    parser.add_argument("--visibility", choices=["private", "public"], default="private")
    parser.add_argument("--publish", action="store_true", help="Authorize GitHub CLI creation/push (interactive confirmation follows)")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.repo):
        parser.error("--repo may contain only letters, numbers, hyphens, periods and underscores")
    root = args.project_root.expanduser().resolve()
    if not (root / "model_deployment.py").is_file():
        parser.error(f"Could not find model_deployment.py in {root}. Run this script from the VS Code project folder.")
    dest = root.parent / "RFMiD_GitHub_Ready"
    dest.mkdir(parents=True, exist_ok=True)
    pack = Packager(root, dest)

    # All top-level Python modules and optional notebooks: no raw datasets.
    for file in sorted(root.glob("*.py")):
        if file.name == Path(__file__).name:
            pack.copy(file, Path("tools") / file.name)
        else:
            pack.copy(file, Path("src") / file.name)
    # Preserve any project modules organized in source subfolders as well.
    for folder_name in ("src", "utils", "models", "scripts"):
        source_folder = root / folder_name
        if source_folder.is_dir():
            for file in source_folder.rglob("*"):
                if file.is_file() and file.suffix.lower() in {".py", ".json", ".yaml", ".yml", ".toml"}:
                    if not any(part.lower() in SKIP_DIRS for part in file.relative_to(source_folder).parts):
                        pack.copy(file, Path("src") / folder_name / file.relative_to(source_folder))
    for file in sorted(root.glob("*.ipynb")):
        pack.copy(file, Path("notebooks") / file.name)
    notebooks_dir = root / "notebooks"
    if notebooks_dir.is_dir():
        for file in notebooks_dir.rglob("*.ipynb"):
            pack.copy(file, Path("notebooks") / file.relative_to(notebooks_dir))

    # Configs, if any. Do not guess dependencies if no lockfile is present.
    for name in ("requirements.txt", "pyproject.toml", "environment.yml", "environment.yaml", "setup.py"):
        pack.copy(root / name, Path(name))

    experiments = root / "RFMiD" / "revised_supervisor_experiments_v2"
    for name, label in (
        ("m1_m2_m3_research", "m1_m2_m3_research"),
        ("eyeq_12epoch_trial", "eyeq_12epoch_trial"),
        ("sklearn_grid_search", "m1_grid_search"),
    ):
        pack.copy_results(experiments / name, label)
    # Also include result exports from a standalone results directory, if created.
    pack.copy_results(root / "results", "additional_results")
    # The Folio submission folder might contain extra preserved result files.
    pack.copy_results(root / "Folio_RFMiD_Submission" / "results", "folio_saved_results")

    for candidate in (
        root / "Folio_RFMiD_Submission" / "Research_Results_Summary.docx",
        root / "Research_Results_Summary.docx",
    ):
        if candidate.is_file():
            pack.copy(candidate, Path("docs") / "Research_Results_Summary.docx")
            break
    for candidate in (
        root / "Folio_RFMiD_Submission" / "Research_Results_Summary.pdf",
        root / "Research_Results_Summary.pdf",
    ):
        if candidate.is_file():
            pack.copy(candidate, Path("docs") / "Research_Results_Summary.pdf")
            break

    for file in list(root.glob("*.log")) + [root / "training_output.txt"]:
        pack.copy(file, Path("logs") / file.name)
    log_dir = root / "Folio_RFMiD_Submission" / "training_logs"
    if log_dir.is_dir():
        for file in log_dir.rglob("*"):
            if file.is_file() and file.suffix.lower() in {".txt", ".log"}:
                pack.copy(file, Path("logs") / file.relative_to(log_dir))

    (dest / ".gitignore").write_text("""# Raw images, datasets, and large trained model files
RFMiD/
EyeQ/
*.pt
*.pth
*.ckpt
*.safetensors
*.onnx
*.pkl
*.pickle
*.zip
# Python / VS Code / secrets
.venv/
venv/
__pycache__/
.ipynb_checkpoints/
.pytest_cache/
.env
.env.*
*.pem
id_rsa*
.vscode/
.DS_Store
""", encoding="utf-8")
    (dest / "results").mkdir(exist_ok=True)
    if not any((dest / "results").rglob("*.csv")) and not any((dest / "results").rglob("*.png")):
        (dest / "results" / "README.md").write_text(
            "# Results directory\n\nNo saved CSV/figure exports were detected at packaging time. "
            "Copy existing exported result files from your experiment folder, then rerun the packaging script.\n",
            encoding="utf-8",
        )
    (dest / "data").mkdir(exist_ok=True)
    (dest / "data" / "README.md").write_text(
        "# Dataset and checkpoint access\n\nThis repository does **not** contain original RFMiD/EyeQ photographs, "
        "any potentially restricted patient data, local file manifests, or trained `.pt` weights. "
        "Datasets must be obtained from their authorized sources and used under applicable licenses.\n\n"
        "Model checkpoints can be supplied to the professor separately if needed (for example, a school-approved "
        "storage link) or tracked with Git LFS after checking size and sharing permissions. "
        "The files under `results/` are saved experiment outputs, not raw images.\n",
        encoding="utf-8",
    )
    has_word = (dest / "docs" / "Research_Results_Summary.docx").is_file()
    has_final_test = any((dest / "results" / "m1_m2_m3_research" / "final_test_clean").glob("*.csv")) if (dest / "results" / "m1_m2_m3_research" / "final_test_clean").exists() else False
    (dest / "README.md").write_text(content_readme(has_word, has_final_test), encoding="utf-8")
    inventory = "# Files copied for the GitHub submission\n\n" + "\n".join(f"- `{f}`" for f in sorted(set(pack.files)))
    if pack.skipped:
        inventory += "\n\n## Files intentionally skipped / requiring review\n\n" + "\n".join(f"- `{s}`" for s in pack.skipped)
    (dest / "FILES_INCLUDED.md").write_text(inventory + "\n", encoding="utf-8")

    print("\nGitHub-ready project:", dest)
    print(f"Copied {len(pack.files)} files. Skipped {len(pack.skipped)} files.")
    if not has_word:
        print("NOTE: Word report not found locally. Download or copy Research_Results_Summary.docx into Folio_RFMiD_Submission/ and rerun to include it.")
    if not has_final_test:
        print("NOTE: No completed official final-test CSV found. README labels all reported numerical results as validation-selection.")
    if pack.skipped:
        print("REVIEW: FILES_INCLUDED.md lists files skipped due to file size or possible sensitive content.")
    print("Review README.md, results and FILES_INCLUDED.md before publishing.")
    if args.publish:
        publish(dest, args.repo, args.visibility)
    else:
        print("\nTo publish after reviewing: python publish_retinal_project.py --publish --visibility public")
        print("Or choose private: python publish_retinal_project.py --publish --visibility private")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
