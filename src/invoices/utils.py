"""Utility functions for the invoice extraction system."""

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Version stamps - updated for v2 contract
CONTRACT_VERSION = "v2"
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
        **get_version_info()
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


def stable_feature_vector_v1(features_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract and flatten features_v1 into a stable, sorted feature vector.
    
    Args:
        features_dict: Dictionary containing features_v1 structure
        
    Returns:
        Dictionary with sorted feature keys and float values
    """
    import numpy as np
    
    flattened = {}
    
    # Handle nested feature structures safely
    for key, value in features_dict.items():
        if isinstance(value, dict):
            # Flatten nested dictionaries with dot notation
            for subkey, subvalue in value.items():
                flat_key = f"{key}.{subkey}"
                flattened[flat_key] = float(subvalue) if subvalue is not None else 0.0
        elif isinstance(value, (list, tuple)):
            # Flatten arrays/lists with index notation
            for i, item in enumerate(value):
                flat_key = f"{key}.{i}"
                flattened[flat_key] = float(item) if item is not None else 0.0
        else:
            # Direct scalar values
            flattened[key] = float(value) if value is not None else 0.0
    
    # Return sorted by key for stability
    return dict(sorted(flattened.items()))


def json_dump_sorted(obj: Any) -> str:
    """
    Serialize object to JSON with sorted keys for deterministic output.
    
    Args:
        obj: Object to serialize
        
    Returns:
        Deterministic JSON string
    """
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def stable_bytes_hash(data: bytes) -> str:
    """
    Compute stable SHA256 hash of bytes.
    
    Args:
        data: Bytes to hash
        
    Returns:
        SHA256 hash as hex string
    """
    return hashlib.sha256(data).hexdigest()


def load_contract_schema() -> Dict[str, Any]:
    """
    Load and canonicalize the contract schema from schema/contract.invoice.json.
    
    Returns:
        Canonical schema dictionary with sorted keys
    """
    from . import paths
    
    schema_path = paths.get_repo_root() / "schema" / "contract.invoice.json"
    
    if not schema_path.exists():
        raise FileNotFoundError(f"Contract schema not found: {schema_path}")
    
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    
    # Canonicalize by sorting keys and ensuring consistent structure
    canonical_schema = {
        "name": schema.get("name", ""),
        "semver": schema.get("semver", ""),
        "fields": sorted(schema.get("fields", [])),
        "line_item_fields": sorted(schema.get("line_item_fields", [])),
        "line_items": schema.get("line_items", [])
    }
    
    return canonical_schema


def contract_fingerprint(schema_obj: Dict[str, Any]) -> str:
    """
    Compute fingerprint (short hash) of canonical schema content.
    
    Args:
        schema_obj: Canonical schema dictionary
        
    Returns:
        12-character hex fingerprint
    """
    canonical_json = json_dump_sorted(schema_obj)
    hash_bytes = canonical_json.encode('utf-8')
    return stable_bytes_hash(hash_bytes)[:12]


def compute_contract_version(schema_obj: Dict[str, Any]) -> str:
    """
    Compute contract version from semver + content fingerprint.
    
    Args:
        schema_obj: Canonical schema dictionary
        
    Returns:
        Version string in format "semver+fingerprint"
    """
    semver = schema_obj.get("semver", "0.0.0")
    fingerprint = contract_fingerprint(schema_obj)
    return f"{semver}+{fingerprint}"


def get_version_info() -> Dict[str, str]:
    """
    Get complete version information with environment variable overrides.
    
    Returns:
        Dictionary with all five version stamps
    """
    # Load schema to compute contract version
    try:
        schema = load_contract_schema()
        contract_version = compute_contract_version(schema)
    except Exception:
        # Fallback to hardcoded version if schema loading fails
        contract_version = CONTRACT_VERSION
    
    # Check for environment variable overrides
    feature_version = os.environ.get('FEATURE_VERSION', FEATURE_VERSION)
    decoder_version = os.environ.get('DECODER_VERSION', DECODER_VERSION)
    calibration_version = os.environ.get('CALIBRATION_VERSION', CALIBRATION_VERSION)
    
    # Model version from environment or model ID file
    model_version = os.environ.get('MODEL_VERSION')
    if not model_version:
        model_id = os.environ.get('MODEL_ID')
        if model_id:
            model_version = model_id
        else:
            # Check for model ID file
            try:
                from . import paths
                model_id_path = paths.get_models_dir() / "current" / "model_id.txt"
                if model_id_path.exists():
                    with open(model_id_path, 'r', encoding='utf-8') as f:
                        model_version = f.read().strip()
                else:
                    model_version = MODEL_VERSION
            except Exception:
                model_version = MODEL_VERSION
    
    return {
        "contract_version": contract_version,
        "feature_version": feature_version,
        "decoder_version": decoder_version,
        "model_version": model_version,
        "calibration_version": calibration_version,
    }
