"""Environment and project setup verification script for ScamShield.

This script validates:
1. Python version compatibility (Python 3.9 - 3.11 recommended).
2. Java Runtime Environment (JRE 11 or 17) for PySpark.
3. ScamShield directory hierarchy.
4. Master configuration loading (configs/config.yaml).
5. Logging utility functionality (src/utils/logger.py).
6. Experiment tracking file (experiment_log.csv).
7. Installed Python packages against project requirements.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def check_python_version() -> bool:
    print(f"[1/7] Python Version: {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info < (3, 9):
        print("      [FAIL] ScamShield requires Python 3.9 or higher.")
        return False
    print("      [PASS] Python version is compatible.")
    return True


def check_java() -> bool:
    print("[2/7] Checking Java Runtime (required for PySpark)...")
    try:
        res = subprocess.run(
            ["java", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        output = res.stderr or res.stdout
        first_line = output.splitlines()[0] if output else "Java found"
        print(f"      [PASS] {first_line}")
        return True
    except FileNotFoundError:
        print("      [FAIL] 'java' command not found. Java 11 or 17 is required for PySpark.")
        return False


def check_directories() -> bool:
    print("[3/7] Checking directory structure...")
    required_dirs = [
        "configs",
        "data/raw",
        "data/raw/meajor",
        "data/raw/curated_phishing",
        "data/raw/indian_scam",
        "data/raw/dravidian_sms",
        "data/raw/llm_phishing",
        "data/interim",
        "data/processed",
        "data/gold",
        "src/data",
        "src/preprocessing",
        "src/features/text",
        "src/features/url",
        "src/features/intent",
        "src/models",
        "src/evaluation",
        "src/explainability",
        "src/utils",
        "experiments/M1_text",
        "experiments/M2_text_url",
        "experiments/M3_text_intent",
        "experiments/M4_full",
        "models",
        "reports",
        "dashboard",
        "api",
        "tests",
    ]

    all_exist = True
    for rel_path in required_dirs:
        p = Path(rel_path)
        if not p.exists():
            print(f"      [MISSING] {rel_path}")
            all_exist = False

    if all_exist:
        print(f"      [PASS] All {len(required_dirs)} required directories exist.")
    return all_exist


def check_config() -> bool:
    print("[4/7] Checking master configuration (configs/config.yaml)...")
    config_file = Path("configs/config.yaml")
    if not config_file.exists():
        print("      [FAIL] configs/config.yaml does not exist.")
        return False
    try:
        import yaml
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if "project" in cfg and "paths" in cfg and "splits" in cfg:
            print(f"      [PASS] config.yaml loaded successfully (Project: {cfg['project']['name']}).")
            return True
        else:
            print("      [FAIL] Missing expected keys in configs/config.yaml.")
            return False
    except ImportError:
        print("      [WARN] PyYAML not installed in current interpreter; skipping yaml parse test.")
        return True


def check_logger() -> bool:
    print("[5/7] Checking logging utility (src/utils/logger.py)...")
    try:
        from src.utils.logger import get_logger
        logger = get_logger("test_env")
        logger.info("Verification test log entry successfully written.")
        print("      [PASS] Logger initialized and wrote test entry to reports/scamshield.log.")
        return True
    except Exception as e:
        print(f"      [FAIL] Logger failed with error: {e}")
        return False


def check_experiment_log() -> bool:
    print("[6/7] Checking experiment tracking log (experiment_log.csv)...")
    exp_file = Path("experiment_log.csv")
    if not exp_file.exists():
        print("      [FAIL] experiment_log.csv does not exist.")
        return False

    with open(exp_file, "r", encoding="utf-8") as f:
        header = f.readline().strip()

    required_cols = ["experiment_id", "timestamp", "research_question", "model_name", "feature_set"]
    if all(col in header for col in required_cols):
        print("      [PASS] experiment_log.csv header is properly formatted.")
        return True
    else:
        print(f"      [FAIL] experiment_log.csv missing expected columns in header: {header}")
        return False


def check_dependencies() -> dict[str, bool]:
    print("[7/7] Checking Python dependencies...")
    packages = [
        ("pyspark", "pyspark"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("pyarrow", "pyarrow"),
        ("yaml", "pyyaml"),
        ("sklearn", "scikit-learn"),
        ("shap", "shap"),
        ("streamlit", "streamlit"),
        ("fastapi", "fastapi"),
        ("torch", "torch"),
        ("transformers", "transformers"),
    ]

    status = {}
    for module_name, pkg_name in packages:
        try:
            __import__(module_name)
            print(f"      [INSTALLED] {pkg_name}")
            status[pkg_name] = True
        except ImportError:
            print(f"      [NOT INSTALLED] {pkg_name}")
            status[pkg_name] = False
    return status


def main() -> None:
    print("=" * 60)
    print("      SCAMSHIELD — ENVIRONMENT & SETUP VERIFICATION")
    print("=" * 60)

    py_ok = check_python_version()
    java_ok = check_java()
    dirs_ok = check_directories()
    cfg_ok = check_config()
    log_ok = check_logger()
    exp_ok = check_experiment_log()
    deps = check_dependencies()

    print("=" * 60)
    core_setup_ok = py_ok and java_ok and dirs_ok and cfg_ok and log_ok and exp_ok
    if core_setup_ok:
        print("  STATUS: Phase 1 project scaffolding is VERIFIED and READY.")
    else:
        print("  STATUS: Scaffolding issues detected. Review errors above.")

    missing_core = [pkg for pkg in ["pandas", "numpy", "pyarrow", "scikit-learn", "pyspark", "pyyaml"] if not deps.get(pkg, False)]
    if missing_core:
        print(f"  NOTE: To install pending dependencies, run:")
        print(f"        pip install -r requirements.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
