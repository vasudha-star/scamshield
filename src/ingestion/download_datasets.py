"""Download raw datasets and append their metadata to data/manifest.csv.

Usage:
    python src/ingestion/download_datasets.py --dataset all
    python src/ingestion/download_datasets.py --dataset meajor

Design (blueprint Section 10, steps 18-19):
  - Record URL, DOI, version, license and retrieval date for every dataset.
  - Store raw files unchanged, under data/raw/<dataset_key>/.

Zenodo records are fetched via the Zenodo REST API. The Hugging Face dataset
is fetched via `huggingface_hub`. Records that are DOI-only (no direct
download API, e.g. IEEE DataPort) are flagged for manual download — this
script only writes a stub note in that case and never fabricates a URL.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "datasets.yaml"
RAW_DIR = ROOT / "data" / "raw"
MANIFEST_PATH = ROOT / "data" / "manifest.csv"

MANIFEST_FIELDS = [
    "dataset_key",
    "name",
    "url",
    "doi",
    "approx_scale",
    "languages",
    "retrieved_at",
    "license",
    "record_count",
    "status",
    "notes",
]

ZENODO_RECORD_RE = re.compile(r"zenodo\.org/records?/(\d+)")


def load_registry() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["datasets"]


def append_manifest_row(row: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not MANIFEST_PATH.exists()
    with open(MANIFEST_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def download_zenodo(key: str, meta: dict) -> dict:
    """Download all files for a Zenodo record via its REST API."""
    match = ZENODO_RECORD_RE.search(meta["url"] or "")
    if not match:
        return {"status": "error", "notes": "could not parse zenodo record id"}
    record_id = match.group(1)
    api_url = f"https://zenodo.org/api/records/{record_id}"

    dest = RAW_DIR / key
    dest.mkdir(parents=True, exist_ok=True)

    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    record = resp.json()
    license_id = (record.get("metadata", {}).get("license") or {}).get("id", "unknown")

    files = record.get("files", [])
    for file_entry in files:
        file_url = file_entry["links"]["self"]
        file_name = file_entry["key"]
        out_path = dest / file_name
        if out_path.exists():
            continue
        with requests.get(file_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(out_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)

    return {
        "status": "downloaded" if files else "no_files_found",
        "license": license_id,
        "record_count": "",  # fill in after inspecting the file(s)
        "notes": f"{len(files)} file(s) -> {dest}",
    }


def download_huggingface(key: str, meta: dict) -> dict:
    from huggingface_hub import snapshot_download

    dest = RAW_DIR / key
    dest.mkdir(parents=True, exist_ok=True)
    repo_id = meta["url"].split("huggingface.co/datasets/")[-1].strip("/")

    local_dir = snapshot_download(
        repo_id=repo_id, repo_type="dataset", local_dir=str(dest)
    )
    return {
        "status": "downloaded",
        "license": "see dataset card",
        "record_count": "",
        "notes": f"snapshot -> {local_dir}",
    }


def download_manual(key: str, meta: dict) -> dict:
    dest = RAW_DIR / key
    dest.mkdir(parents=True, exist_ok=True)
    readme = dest / "MANUAL_DOWNLOAD_REQUIRED.txt"
    readme.write_text(
        f"No programmatic download available for '{meta['name']}'.\n"
        f"DOI: {meta.get('doi')}\n"
        f"Download manually and place raw files in this folder, then\n"
        f"re-run this script with --skip-download to log the manifest entry.\n"
    )
    return {"status": "manual_required", "license": "unknown", "record_count": "", "notes": "see MANUAL_DOWNLOAD_REQUIRED.txt"}


def process_dataset(key: str, meta: dict, skip_download: bool) -> None:
    print(f"[{key}] {meta['name']}")
    if skip_download:
        result = {"status": "skipped", "license": "", "record_count": "", "notes": "skip_download flag"}
    elif meta["url"] and "zenodo.org" in meta["url"]:
        result = download_zenodo(key, meta)
    elif meta["url"] and "huggingface.co/datasets/" in meta["url"]:
        result = download_huggingface(key, meta)
    else:
        result = download_manual(key, meta)

    append_manifest_row(
        {
            "dataset_key": key,
            "name": meta["name"],
            "url": meta["url"],
            "doi": meta.get("doi"),
            "approx_scale": meta.get("approx_scale"),
            "languages": ",".join(meta.get("languages", [])),
            "retrieved_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            **result,
        }
    )
    print(f"  -> {result['status']}: {result.get('notes', '')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", help="dataset key from config/datasets.yaml, or 'all'")
    parser.add_argument("--skip-download", action="store_true", help="only log manifest entry (files already placed manually)")
    args = parser.parse_args()

    registry = load_registry()
    keys = list(registry.keys()) if args.dataset == "all" else [args.dataset]

    for key in keys:
        if key not in registry:
            print(f"Unknown dataset key: {key}. Available: {list(registry.keys())}")
            continue
        process_dataset(key, registry[key], args.skip_download)

    print(f"\nManifest updated: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
