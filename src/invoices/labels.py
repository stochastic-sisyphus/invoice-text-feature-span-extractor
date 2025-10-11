"""Label Studio integration for label pulling, import, and alignment."""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from . import ingest, paths, tokenize, utils


def pull_labels() -> dict[str, Any]:
    """
    Pull labels from Label Studio HTTP API.
    Safe no-op when credentials missing.

    Returns:
        Summary of pulled labels or error info
    """
    # Check for credentials
    api_url = os.environ.get("LABEL_STUDIO_URL")
    api_token = os.environ.get("LABEL_STUDIO_TOKEN")

    if not api_url or not api_token:
        print("Label Studio credentials missing (LABEL_STUDIO_URL, LABEL_STUDIO_TOKEN)")
        print("Skipping label pull (safe no-op)")
        return {"status": "skipped", "reason": "missing_credentials"}

    try:
        # Construct API endpoint for tasks
        tasks_url = f"{api_url.rstrip('/')}/api/tasks/"

        # Create request with auth header
        request = urllib.request.Request(tasks_url)
        request.add_header("Authorization", f"Token {api_token}")
        request.add_header("Content-Type", "application/json")

        # Make request
        with urllib.request.urlopen(request) as response:
            tasks_data = json.loads(response.read().decode())

        # Save raw labels
        labels_raw_dir = paths.get_repo_root() / "data" / "labels" / "raw"
        labels_raw_dir.mkdir(parents=True, exist_ok=True)

        timestamp = utils.get_current_utc_iso().replace(":", "-").replace(".", "-")
        raw_file = labels_raw_dir / f"labels_{timestamp}.json"

        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(tasks_data, f, indent=2)

        task_count = len(tasks_data) if isinstance(tasks_data, list) else 0

        print(f"Pulled {task_count} tasks from Label Studio")
        print(f"Saved raw labels to: {raw_file}")

        return {
            "status": "success",
            "task_count": task_count,
            "raw_file": str(raw_file),
        }

    except urllib.error.HTTPError as e:
        error_msg = f"HTTP error {e.code}: {e.reason}"
        print(f"Failed to pull labels: {error_msg}")
        return {"status": "error", "error": error_msg}

    except Exception as e:
        error_msg = str(e)
        print(f"Failed to pull labels: {error_msg}")
        return {"status": "error", "error": error_msg}


def import_labels(path: str) -> dict[str, Any]:
    """
    Import labels from local Label Studio export file.

    Args:
        path: Path to Label Studio export JSON file

    Returns:
        Summary of imported labels
    """
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")

    # Load labels file
    with open(path_obj, encoding="utf-8") as f:
        labels_data = json.load(f)

    # Save to raw directory
    labels_raw_dir = paths.get_repo_root() / "data" / "labels" / "raw"
    labels_raw_dir.mkdir(parents=True, exist_ok=True)

    timestamp = utils.get_current_utc_iso().replace(":", "-").replace(".", "-")
    import_filename = f"import_{path_obj.stem}_{timestamp}.json"
    raw_file = labels_raw_dir / import_filename

    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(labels_data, f, indent=2)

    task_count = len(labels_data) if isinstance(labels_data, list) else 0

    print(f"Imported {task_count} tasks from {path}")
    print(f"Saved to: {raw_file}")

    return {
        "status": "success",
        "task_count": task_count,
        "raw_file": str(raw_file),
        "source_file": str(path),
    }


def compute_iou(bbox1: list[float], bbox2: list[float]) -> float:
    """
    Compute Intersection over Union (IoU) for two bounding boxes.

    Args:
        bbox1: [x0, y0, x1, y1] normalized coordinates
        bbox2: [x0, y0, x1, y1] normalized coordinates

    Returns:
        IoU score between 0.0 and 1.0
    """
    # Extract coordinates
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Compute intersection
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)

    if x_max <= x_min or y_max <= y_min:
        return 0.0

    intersection = (x_max - x_min) * (y_max - y_min)

    # Compute union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def align_labels(iou_threshold: float = 0.3, all_files: bool = False) -> dict[str, Any]:
    """
    Align labels with document candidates using IoU-based matching.

    Args:
        iou_threshold: Minimum IoU for candidate-label alignment
        all_files: Process all raw label files if True

    Returns:
        Summary of alignment results
    """
    labels_raw_dir = paths.get_repo_root() / "data" / "labels" / "raw"
    aligned_dir = paths.get_repo_root() / "data" / "labels" / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    if not labels_raw_dir.exists():
        print("No raw labels directory found")
        return {"status": "no_labels", "aligned_count": 0}

    label_files = list(labels_raw_dir.glob("*.json"))

    if not label_files:
        print("No label files found in raw directory")
        return {"status": "no_files", "aligned_count": 0}

    if not all_files:
        # Use most recent file
        label_files = [max(label_files, key=lambda p: p.stat().st_mtime)]

    total_aligned = 0
    alignment_results = []

    for label_file in label_files:
        print(f"Processing labels from: {label_file.name}")

        # Load labels
        with open(label_file, encoding="utf-8") as f:
            labels_data = json.load(f)

        if not isinstance(labels_data, list):
            print(f"Skipping {label_file.name}: not a list of tasks")
            continue

        # Process each labeled task
        aligned_rows = []

        for task in tqdm(labels_data, desc="Aligning labels"):
            # Extract document info
            task_data = task.get("data", {})
            doc_id = task_data.get("doc_id")

            if not doc_id:
                continue

            # Find document SHA256
            indexed_docs = ingest.get_indexed_documents()
            doc_row = indexed_docs[indexed_docs["doc_id"] == doc_id]

            if doc_row.empty:
                continue

            sha256 = doc_row.iloc[0]["sha256"]

            # Load document tokens and candidates
            try:
                tokenize.get_document_tokens(sha256)
                candidates_path = paths.get_candidates_path(sha256)

                if not candidates_path.exists():
                    continue

                candidates_df = pd.read_parquet(candidates_path)
            except Exception:
                continue

            # Process annotations
            annotations = task.get("annotations", [])

            for annotation in annotations:
                results = annotation.get("result", [])

                for result in results:
                    # Extract label info
                    label_field = result.get("from_name")
                    label_value = result.get("value", {}).get("text", [""])[0]

                    if not label_field or not label_value:
                        continue

                    # Extract bounding box (normalized)
                    bbox_info = result.get("value", {})
                    if "x" not in bbox_info or "y" not in bbox_info:
                        continue

                    # Convert percentage to normalized coordinates
                    x_norm = bbox_info["x"] / 100.0
                    y_norm = bbox_info["y"] / 100.0
                    w_norm = bbox_info.get("width", 0) / 100.0
                    h_norm = bbox_info.get("height", 0) / 100.0

                    label_bbox = [x_norm, y_norm, x_norm + w_norm, y_norm + h_norm]

                    # Find best matching candidate
                    best_iou = 0.0
                    best_candidate_idx = None

                    for idx, candidate in candidates_df.iterrows():
                        candidate_bbox = [
                            candidate["bbox_norm_x0"],
                            candidate["bbox_norm_y0"],
                            candidate["bbox_norm_x1"],
                            candidate["bbox_norm_y1"],
                        ]

                        iou = compute_iou(label_bbox, candidate_bbox)

                        if iou > best_iou and iou >= iou_threshold:
                            best_iou = iou
                            best_candidate_idx = idx

                    # Create aligned row if match found
                    if best_candidate_idx is not None:
                        aligned_rows.append(
                            {
                                "doc_id": doc_id,
                                "sha256": sha256,
                                "field": label_field,
                                "label_value": label_value,
                                "candidate_idx": best_candidate_idx,
                                "iou": best_iou,
                                "label_bbox_x0": label_bbox[0],
                                "label_bbox_y0": label_bbox[1],
                                "label_bbox_x1": label_bbox[2],
                                "label_bbox_y1": label_bbox[3],
                            }
                        )

        # Save aligned data if any
        if aligned_rows:
            aligned_df = pd.DataFrame(aligned_rows)
            timestamp = utils.get_current_utc_iso().replace(":", "-").replace(".", "-")
            aligned_file = (
                aligned_dir / f"aligned_{label_file.stem}_{timestamp}.parquet"
            )
            aligned_df.to_parquet(aligned_file, index=False)

            file_aligned_count = len(aligned_rows)
            total_aligned += file_aligned_count

            print(f"Aligned {file_aligned_count} labels from {label_file.name}")
            print(f"Saved to: {aligned_file}")

            alignment_results.append(
                {
                    "source_file": label_file.name,
                    "aligned_file": aligned_file.name,
                    "aligned_count": file_aligned_count,
                }
            )

    return {
        "status": "success",
        "total_aligned": total_aligned,
        "iou_threshold": iou_threshold,
        "files_processed": len(label_files),
        "alignment_results": alignment_results,
    }


def get_aligned_label_files() -> list[Path]:
    """Get list of all aligned label files."""
    aligned_dir = paths.get_repo_root() / "data" / "labels" / "aligned"

    if not aligned_dir.exists():
        return []

    return list(aligned_dir.glob("*.parquet"))


def load_aligned_labels() -> pd.DataFrame:
    """Load all aligned labels into a single DataFrame."""
    aligned_files = get_aligned_label_files()

    if not aligned_files:
        return pd.DataFrame()

    dfs = []
    for file_path in aligned_files:
        try:
            df = pd.read_parquet(file_path)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)
