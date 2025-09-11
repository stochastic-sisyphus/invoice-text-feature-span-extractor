"""Utility functions for the invoice extraction system."""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Version stamps - fixed for today as specified
CONTRACT_VERSION = "v1"
FEATURE_VERSION = "v1"
DECODER_VERSION = "v1"
MODEL_VERSION = "unscored-baseline"
CALIBRATION_VERSION = "none"


def get_version_stamps() -> Dict[str, str]:
    """Get all version stamps for consistent labeling."""
    return {
        "contract_version": CONTRACT_VERSION,
        "feature_version": FEATURE_VERSION,
        "decoder_version": DECODER_VERSION,
        "model_version": MODEL_VERSION,
        "calibration_version": CALIBRATION_VERSION,
    }


def compute_sha256(data: bytes) -> str:
    """Compute SHA256 hash of data."""
    return hashlib.sha256(data).hexdigest()


def compute_stable_token_id(doc_id: str, page_idx: int, token_idx: int, text: str, bbox_norm: tuple) -> str:
    """Compute stable token ID using SHA1 hash as specified."""
    # Convert bbox_norm to string for consistent hashing
    bbox_str = f"{bbox_norm[0]:.6f},{bbox_norm[1]:.6f},{bbox_norm[2]:.6f},{bbox_norm[3]:.6f}"
    hash_input = f"{doc_id}|{page_idx}|{token_idx}|{text}|{bbox_str}"
    return hashlib.sha1(hash_input.encode('utf-8')).hexdigest()


def get_current_utc_iso() -> str:
    """Get current UTC timestamp in ISO8601 format."""
    return datetime.now(timezone.utc).isoformat()


def safe_filename(filename: str) -> str:
    """Convert filename to safe format for storage."""
    # Keep only alphanumeric, dots, hyphens, underscores
    safe_chars = []
    for char in filename:
        if char.isalnum() or char in '.-_':
            safe_chars.append(char)
        else:
            safe_chars.append('_')
    return ''.join(safe_chars)


def write_json_with_backup(filepath: Path, data: Dict[str, Any]) -> None:
    """Write JSON with atomic operation and backup."""
    temp_path = filepath.with_suffix('.tmp')
    
    # Write to temporary file first
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Atomic move
    temp_path.replace(filepath)


def log_timing(operation: str, duration_seconds: float, doc_count: int = 1) -> Dict[str, Any]:
    """Log timing information."""
    return {
        "operation": operation,
        "duration_seconds": round(duration_seconds, 4),
        "doc_count": doc_count,
        "docs_per_second": round(doc_count / duration_seconds, 2) if duration_seconds > 0 else 0,
        "timestamp": get_current_utc_iso(),
        **get_version_stamps()
    }


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time: Optional[float] = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            print(f"{self.operation_name}: {duration:.3f}s")
            
    def elapsed(self) -> float:
        """Get elapsed time since start."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
