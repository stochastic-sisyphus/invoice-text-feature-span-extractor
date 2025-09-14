"""Decoder module for Hungarian assignment with NONE option per field."""

import json
from typing import Dict, Any, Optional, Tuple

import numpy as np

from . import paths, utils, ingest, candidates


# Field sets now loaded dynamically from schema/contract.invoice.json
# Legacy constants removed - all field sets come from canonical schema

# Default NONE bias (MUCH higher to encourage abstaining when uncertain)
DEFAULT_NONE_BIAS = 15.0


def try_import_scipy_hungarian():
    """Try to import scipy's Hungarian algorithm implementation."""
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment
    except ImportError:
        return None


def simple_hungarian_fallback(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple fallback Hungarian algorithm implementation.
    Not optimal but works for small matrices.
    """
    n_rows, n_cols = cost_matrix.shape
    
    # Greedy assignment - assign each row to its minimum cost column
    row_indices = []
    col_indices = []
    used_cols = set()
    
    # Sort rows by their minimum cost to prioritize easier assignments
    row_min_costs = [(i, np.min(cost_matrix[i])) for i in range(n_rows)]
    row_min_costs.sort(key=lambda x: x[1])
    
    for row_idx, _ in row_min_costs:
        # Find best available column for this row
        best_col = None
        best_cost = float('inf')
        
        for col_idx in range(n_cols):
            if col_idx not in used_cols and cost_matrix[row_idx, col_idx] < best_cost:
                best_col = col_idx
                best_cost = cost_matrix[row_idx, col_idx]
        
        if best_col is not None:
            row_indices.append(row_idx)
            col_indices.append(best_col)
            used_cols.add(best_col)
    
    return np.array(row_indices), np.array(col_indices)


def maybe_load_model_v1() -> Optional[Dict[str, Any]]:
    """
    Load trained XGBoost model bundle if present.
    
    Returns:
        Model bundle dict if successful, None if not available or incompatible
    """
    model_bundle_path = paths.get_model_bundle_path()
    model_manifest_path = paths.get_model_manifest_path()
    
    if not model_bundle_path.exists() or not model_manifest_path.exists():
        return None
    
    try:
        # Load manifest first to check compatibility
        with open(model_manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        # Verify version compatibility
        if manifest.get('feature_version') != utils.FEATURE_VERSION:
            print(f"Warning: Feature version mismatch. Expected {utils.FEATURE_VERSION}, got {manifest.get('feature_version')}. Falling back to baseline.")
            return None
        
        if manifest.get('normalize_version') != "doctext_nfc_newline_v1":  # From normalize.py
            print(f"Warning: Normalize version mismatch. Expected doctext_nfc_newline_v1, got {manifest.get('normalize_version')}. Falling back to baseline.")
            return None
        
        # Load model bundle
        import numpy as np
        model_data = np.load(model_bundle_path, allow_pickle=True)
        
        # Rebuild XGBoost boosters from saved JSON
        import xgboost as xgb
        
        boosters = {}
        labels = model_data['labels'].tolist()
        feature_keys = model_data['feature_keys'].tolist()
        booster_jsons = model_data['booster_jsons'].tolist()
        
        for label, booster_json in zip(labels, booster_jsons):
            booster = xgb.Booster()
            booster.load_model(bytearray(booster_json, 'utf-8'))
            boosters[label] = booster
        
        model_bundle = {
            'boosters': boosters,
            'labels': labels,
            'feature_keys': feature_keys,
            'manifest': manifest
        }
        
        print(f"Loaded model bundle with {len(labels)} labels, {len(feature_keys)} features")
        return model_bundle
        
    except ImportError:
        print("Warning: XGBoost not available. Falling back to baseline decoder.")
        return None
    except Exception as e:
        print(f"Warning: Failed to load model bundle: {e}. Falling back to baseline decoder.")
        return None


def compute_weak_prior_cost(field: str, candidate: Dict[str, Any]) -> float:
    """
    Compute weak prior cost for field-candidate assignment with precision focus.
    Lower cost = better match. Heavily penalize long spans for precision.
    """
    bucket = candidate.get('bucket', '')
    center_x = candidate.get('center_x', 0.5)
    center_y = candidate.get('center_y', 0.5)
    raw_text = candidate.get('raw_text', '')
    text_length = len(raw_text.strip())
    
    base_cost = 1.0  # Neutral cost
    
    # CRITICAL: Heavy length penalties for precision
    # Field-specific length expectations (even stricter)
    field_length_limits = {
        'invoice_number': 25,
        'account_number': 25,
        'customer_account': 25,
        'invoice_date': 15,
        'due_date': 15,
        'issue_date': 15,
        'total_amount': 15,
        'subtotal': 15,
        'tax_amount': 15,
        'discount': 15,
        'currency': 8,
        'contact_phone': 20,
        'contact_email': 40,
        'routing_number': 15,
        'swift_code': 12,
        'tax_id': 20,
    }
    
    # CRITICAL: Immediate rejection for garbled text
    garbled_indicators = ['eht', 'sah', 'ekat', 'ruoy', 'morf', 'yap']
    if any(indicator in raw_text.lower() for indicator in garbled_indicators):
        return 10.0  # Maximum penalty for garbled text
    
    expected_length = field_length_limits.get(field, 100)  # Default for address/name fields
    
    # Heavy penalty for exceeding expected length
    if text_length > expected_length:
        length_penalty = (text_length - expected_length) / expected_length
        base_cost += min(2.0, length_penalty * 2.0)  # Cap at +2.0
    
    # Bonus for appropriate length
    if text_length <= expected_length:
        base_cost -= 0.1
    
    # CRITICAL: Strict type validation for field appropriateness
    if field in ['total_amount', 'subtotal', 'tax_amount', 'discount']:
        # Must look like an amount
        if bucket != 'amount_like' or not any(c.isdigit() for c in raw_text):
            base_cost += 3.0  # Heavy penalty for non-amounts
        else:
            base_cost -= 0.6  # Strong preference for valid amounts
        
        # Prefer top-right or summary box area
        if center_x > 0.6 and center_y < 0.4:  # Top-right
            base_cost -= 0.3
        elif center_y > 0.6:  # Bottom area (summary)
            base_cost -= 0.2
    
    elif field in ['invoice_date', 'due_date', 'issue_date']:
        # Must look like a date
        if bucket != 'date_like':
            base_cost += 3.0  # Heavy penalty for non-dates
        else:
            # Additional validation: check for date-like content
            digits = sum(1 for c in raw_text if c.isdigit())
            if digits < 2:  # Dates need substantial digits
                base_cost += 2.0
            else:
                base_cost -= 0.6  # Strong preference for valid dates
        
        # Prefer header area
        if center_y < 0.3:  # Top 30%
            base_cost -= 0.3
    
    elif field in ['invoice_number', 'account_number', 'customer_account', 'routing_number', 'swift_code', 'tax_id']:
        # Must look like an ID
        if bucket != 'id_like':
            base_cost += 2.0  # Heavy penalty for non-IDs
        else:
            # Additional validation: must have alphanumeric content
            if not any(c.isalnum() for c in raw_text):
                base_cost += 2.0
            else:
                base_cost -= 0.6  # Strong preference for valid IDs
        
        # Prefer header area
        if center_y < 0.3:  # Top 30%
            base_cost -= 0.3
    
    elif field == 'currency':
        # Currency should be very short
        if text_length <= 5:
            base_cost -= 0.4
        if any(sym in raw_text for sym in ['$', '€', '£', 'USD', 'EUR', 'GBP']):
            base_cost -= 0.5
    
    # Contact field preferences
    elif field == 'contact_email':
        if '@' in raw_text and '.' in raw_text:
            base_cost -= 0.4
    
    elif field == 'contact_phone':
        digit_count = sum(1 for c in raw_text if c.isdigit())
        if digit_count >= 7:  # Phone numbers have many digits
            base_cost -= 0.3
    
    # Keyword proximity bonus (enhanced)
    if bucket == 'keyword_proximal':
        base_cost -= 0.3  # Increased bonus
    
    # Penalize random negatives and very long spans
    if bucket == 'random_negative':
        base_cost += 0.8
    
    # Global penalty for very long text (any field)
    if text_length > 100:
        base_cost += 1.0  # Strong penalty
    
    return max(0.0, base_cost)  # Ensure non-negative


def decode_document(sha256: str, none_bias: float = DEFAULT_NONE_BIAS) -> Dict[str, Any]:
    """
    Decode a single document using Hungarian assignment.
    
    Args:
        sha256: Document SHA256 hash
        none_bias: Cost for NONE assignment (higher = more likely to abstain)
        
    Returns:
        Assignment results for each field
    """
    # Load schema to get current field set
    schema = utils.load_contract_schema()
    field_set = schema['fields']
    # Get candidates
    candidates_df = candidates.get_document_candidates(sha256)
    
    doc_info = ingest.get_document_info(sha256)
    if not doc_info:
        raise ValueError(f"Document not found: {sha256}")
    
    # If no candidates, return all NONE assignments
    if candidates_df.empty:
        assignments = {}
        for field in field_set:
            assignments[field] = {
                'assignment_type': 'NONE',
                'candidate_index': None,
                'cost': none_bias,
                'field': field,
            }
        return assignments
    
    candidates_list = candidates_df.to_dict('records')
    n_candidates = len(candidates_list)
    n_fields = len(field_set)
    
    # Try to load trained model
    model_bundle = maybe_load_model_v1()
    
    # Build cost matrix: fields × (candidates + NONE)
    # Each field can be assigned to any candidate or to NONE
    cost_matrix = np.full((n_fields, n_candidates + 1), 2.0)  # Base cost
    
    if model_bundle is not None:
        # Use trained model for scoring
        try:
            # Build feature matrix for all candidates
            feature_vectors = []
            for candidate in candidates_list:
                # Extract features_v1 structure from candidate
                features_v1 = {}
                for key, value in candidate.items():
                    if not key.startswith(('candidate_id', 'doc_id', 'sha256', 'bucket', 'raw_text', 'normalized_text', 'bbox_norm')):
                        features_v1[key] = value
                
                # Flatten features with same key order as training
                feature_vector = utils.stable_feature_vector_v1(features_v1)
                feature_vectors.append(feature_vector)
            
            # Check feature key compatibility
            if feature_vectors:
                candidate_feature_keys = sorted(feature_vectors[0].keys())
                model_feature_keys = model_bundle['feature_keys']
                
                if candidate_feature_keys != model_feature_keys:
                    print(f"Warning: Feature keys mismatch. Model has {len(model_feature_keys)} features, candidates have {len(candidate_feature_keys)}. Falling back to baseline.")
                    model_bundle = None
                else:
                    # Build prediction matrix
                    X = np.array([[fv[key] for key in model_feature_keys] for fv in feature_vectors], dtype=np.float32)
                    
                    # Predict probabilities for each label
                    for field_idx, field in enumerate(field_set):
                        if field in model_bundle['boosters']:
                            booster = model_bundle['boosters'][field]
                            # Convert to DMatrix for XGBoost
                            import xgboost as xgb
                            dmatrix = xgb.DMatrix(X)
                            probs = booster.predict(dmatrix)
                            
                            # Convert probabilities to costs (1 - p)
                            for cand_idx, prob in enumerate(probs):
                                cost_matrix[field_idx, cand_idx] = 1.0 - float(prob)
                        else:
                            # Field not in trained model, use weak prior
                            for cand_idx, candidate in enumerate(candidates_list):
                                cost = compute_weak_prior_cost(field, candidate)
                                cost_matrix[field_idx, cand_idx] = cost
        except Exception as e:
            print(f"Warning: Model scoring failed: {e}. Falling back to baseline.")
            model_bundle = None
    
    # Fallback to weak priors if no model or model failed
    if model_bundle is None:
        for field_idx, field in enumerate(field_set):
            for cand_idx, candidate in enumerate(candidates_list):
                cost = compute_weak_prior_cost(field, candidate)
                cost_matrix[field_idx, cand_idx] = cost
    
    # Set NONE cost (last column) - always preserved
    for field_idx in range(n_fields):
        cost_matrix[field_idx, n_candidates] = none_bias
    
    # Apply Hungarian algorithm
    hungarian_fn = try_import_scipy_hungarian()
    
    if hungarian_fn is not None:
        try:
            row_indices, col_indices = hungarian_fn(cost_matrix)
        except Exception as e:
            print(f"Scipy Hungarian failed, using fallback: {e}")
            row_indices, col_indices = simple_hungarian_fallback(cost_matrix)
    else:
        print("Scipy not available, using simple Hungarian fallback")
        row_indices, col_indices = simple_hungarian_fallback(cost_matrix)
    
    # Build assignments
    assignments = {}
    
    for field_idx, field in enumerate(field_set):
        # Find assignment for this field
        field_assignment = None
        for i, (row_idx, col_idx) in enumerate(zip(row_indices, col_indices)):
            if row_idx == field_idx:
                field_assignment = (col_idx, cost_matrix[row_idx, col_idx])
                break
        
        if field_assignment is None:
            # No assignment found, default to NONE
            assignments[field] = {
                'assignment_type': 'NONE',
                'candidate_index': None,
                'cost': none_bias,
                'field': field,
            }
        else:
            col_idx, cost = field_assignment
            
            if col_idx == n_candidates:  # NONE assignment
                assignments[field] = {
                    'assignment_type': 'NONE',
                    'candidate_index': None,
                    'cost': cost,
                    'field': field,
                }
            else:  # Candidate assignment
                assignments[field] = {
                    'assignment_type': 'CANDIDATE',
                    'candidate_index': col_idx,
                    'cost': cost,
                    'field': field,
                    'candidate': candidates_list[col_idx],
                }
    
    return assignments


def decode_all_documents(none_bias: float = DEFAULT_NONE_BIAS) -> Dict[str, Dict[str, Any]]:
    """Decode all documents in the index."""
    indexed_docs = ingest.get_indexed_documents()
    
    # Load schema to get current field set
    schema = utils.load_contract_schema()
    field_set = schema['fields']
    
    if indexed_docs.empty:
        print("No documents found in index")
        return {}
    
    results = {}
    
    print(f"Decoding {len(indexed_docs)} documents with NONE bias {none_bias}")
    
    for _, doc_info in indexed_docs.iterrows():
        sha256 = doc_info['sha256']
        
        try:
            assignments = decode_document(sha256, none_bias)
            results[sha256] = assignments
            
            # Count assignments
            candidate_count = sum(1 for a in assignments.values() if a['assignment_type'] == 'CANDIDATE')
            none_count = sum(1 for a in assignments.values() if a['assignment_type'] == 'NONE')
            
            print(f"Decoded {sha256[:16]}: {candidate_count} predictions, {none_count} abstains")
            
        except Exception as e:
            print(f"Failed to decode {sha256[:16]}: {e}")
            # Create default NONE assignments
            assignments = {}
            for field in field_set:
                assignments[field] = {
                    'assignment_type': 'NONE',
                    'candidate_index': None,
                    'cost': none_bias,
                    'field': field,
                }
            results[sha256] = assignments
    
    return results
