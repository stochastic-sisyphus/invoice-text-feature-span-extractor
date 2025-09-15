"""Decoder module for Hungarian assignment with NONE option per field."""

from typing import Dict, Any, Optional, Tuple

import numpy as np

from . import utils, ingest, candidates, train


# Header field set as specified
HEADER_FIELDS = [
    'invoice_number',
    'invoice_date', 
    'due_date',
    'total_amount_due',
    'previous_balance',
    'payments_and_credits',
    'account_number',
    'carrier_name',
    'document_type',
    'currency_code',
]

# Default NONE bias (high cost to encourage abstaining)
DEFAULT_NONE_BIAS = 10.0

# Global cache for loaded models
_LOADED_MODELS = None


def maybe_load_model_v1() -> Optional[Dict[str, Any]]:
    """
    Load trained XGBoost models if available.
    Graceful fallback when models not present.
    
    Returns:
        Loaded models dict or None
    """
    global _LOADED_MODELS
    
    if _LOADED_MODELS is not None:
        return _LOADED_MODELS
    
    try:
        loaded_models = train.load_trained_models()
        _LOADED_MODELS = loaded_models
        return loaded_models
    except Exception as e:
        print(f"Could not load trained models: {e}")
        print("Falling back to baseline weak-prior decoder")
        _LOADED_MODELS = None
        return None


def compute_ml_cost(field: str, candidate: Dict[str, Any], loaded_models: Dict[str, Any]) -> float:
    """
    Compute ML-based cost using trained XGBoost model.
    
    Args:
        field: Field name
        candidate: Candidate dictionary
        loaded_models: Loaded models from maybe_load_model_v1()
        
    Returns:
        ML cost (lower = better match)
    """
    models_dict = loaded_models.get('models', {})
    
    if field not in models_dict:
        # No model for this field, fall back to weak prior
        return compute_weak_prior_cost(field, candidate)
    
    model_info = models_dict[field]
    model = model_info['model']
    feature_names = model_info['feature_names']
    
    # Create features for this candidate
    try:
        # Basic geometric features
        center_x = (candidate['bbox_norm_x0'] + candidate['bbox_norm_x1']) / 2
        center_y = (candidate['bbox_norm_y0'] + candidate['bbox_norm_y1']) / 2
        width = candidate['bbox_norm_x1'] - candidate['bbox_norm_x0']
        height = candidate['bbox_norm_y1'] - candidate['bbox_norm_y0']
        area = width * height
        
        # Text features
        text = str(candidate.get('text', ''))
        char_count = len(text)
        word_count = len(text.split())
        digit_count = sum(1 for c in text if c.isdigit())
        alpha_count = sum(1 for c in text if c.isalpha())
        
        # Bucket features (one-hot)
        bucket = candidate.get('bucket', 'other')
        bucket_features = {
            'bucket_amount_like': int(bucket == 'amount_like'),
            'bucket_date_like': int(bucket == 'date_like'),
            'bucket_id_like': int(bucket == 'id_like'),
            'bucket_keyword_proximal': int(bucket == 'keyword_proximal'),
            'bucket_random_negative': int(bucket == 'random_negative'),
            'bucket_other': int(bucket not in ['amount_like', 'date_like', 'id_like', 'keyword_proximal', 'random_negative'])
        }
        
        # Page features
        page_idx = candidate.get('page_idx', 0)
        
        features = {
            'center_x': center_x,
            'center_y': center_y,
            'width': width,
            'height': height,
            'area': area,
            'char_count': char_count,
            'word_count': word_count,
            'digit_count': digit_count,
            'alpha_count': alpha_count,
            'page_idx': page_idx,
            **bucket_features
        }
        
        # Create feature vector in correct order
        feature_vector = [features.get(name, 0.0) for name in feature_names]
        
        # Get prediction probability
        prob_positive = model.predict_proba([feature_vector])[0][1]
        
        # Convert to cost (lower probability = higher cost)
        ml_cost = 1.0 - prob_positive
        
        return max(0.0, ml_cost)
        
    except Exception as e:
        print(f"ML cost computation failed for {field}: {e}")
        return compute_weak_prior_cost(field, candidate)


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


def compute_weak_prior_cost(field: str, candidate: Dict[str, Any]) -> float:
    """
    Compute weak prior cost for field-candidate assignment.
    Lower cost = better match.
    """
    bucket = candidate.get('bucket', '')
    center_x = candidate.get('center_x', 0.5)
    center_y = candidate.get('center_y', 0.5)
    
    base_cost = 1.0  # Neutral cost
    
    # Field-specific preferences
    if field == 'total_amount_due':
        if bucket == 'amount_like':
            base_cost -= 0.5  # Strong preference for amount-like
        
        # Prefer top-right or summary box area
        if center_x > 0.6 and center_y < 0.4:  # Top-right
            base_cost -= 0.3
        elif center_y > 0.6:  # Bottom area (summary)
            base_cost -= 0.2
    
    elif field in ['invoice_date', 'due_date']:
        if bucket == 'date_like':
            base_cost -= 0.5  # Strong preference for date-like
        
        # Prefer header area
        if center_y < 0.3:  # Top 30%
            base_cost -= 0.3
    
    elif field in ['invoice_number', 'account_number']:
        if bucket == 'id_like':
            base_cost -= 0.5  # Strong preference for id-like
        
        # Prefer header area
        if center_y < 0.3:  # Top 30%
            base_cost -= 0.3
    
    elif field == 'previous_balance':
        if bucket == 'amount_like':
            base_cost -= 0.3  # Moderate preference for amount-like
    
    elif field == 'payments_and_credits':
        if bucket == 'amount_like':
            base_cost -= 0.3  # Moderate preference for amount-like
    
    # Keyword proximity bonus
    if bucket == 'keyword_proximal':
        base_cost -= 0.2
    
    # Penalize random negatives
    if bucket == 'random_negative':
        base_cost += 0.5
    
    return max(0.0, base_cost)  # Ensure non-negative


def decode_document(sha256: str, none_bias: float = DEFAULT_NONE_BIAS) -> Dict[str, Any]:
    """
    Decode a single document using Hungarian assignment with ML models when available.
    
    Args:
        sha256: Document SHA256 hash
        none_bias: Cost for NONE assignment (higher = more likely to abstain)
        
    Returns:
        Assignment results for each field
    """
    # Try to load trained models
    loaded_models = maybe_load_model_v1()
    
    # Get schema fields instead of hardcoded HEADER_FIELDS
    schema_obj = utils.load_contract_schema()
    schema_fields = schema_obj.get('fields', [])
    
    if not schema_fields:
        raise ValueError("Schema contains no fields")
    
    # Get candidates
    candidates_df = candidates.get_document_candidates(sha256)
    
    doc_info = ingest.get_document_info(sha256)
    if not doc_info:
        raise ValueError(f"Document not found: {sha256}")
    
    # If no candidates, return all NONE assignments
    if candidates_df.empty:
        assignments = {}
        for field in schema_fields:
            assignments[field] = {
                'assignment_type': 'NONE',
                'candidate_index': None,
                'cost': none_bias,
                'field': field,
            }
        return assignments
    
    candidates_list = candidates_df.to_dict('records')
    n_candidates = len(candidates_list)
    n_fields = len(schema_fields)
    
    # Build cost matrix: fields × (candidates + NONE)
    # Each field can be assigned to any candidate or to NONE
    cost_matrix = np.full((n_fields, n_candidates + 1), 2.0)  # Base cost
    
    # Fill candidate costs
    for field_idx, field in enumerate(schema_fields):
        for cand_idx, candidate in enumerate(candidates_list):
            if loaded_models:
                # Use ML cost when models available
                cost = compute_ml_cost(field, candidate, loaded_models)
            else:
                # Fall back to weak prior cost
                cost = compute_weak_prior_cost(field, candidate)
            cost_matrix[field_idx, cand_idx] = cost
        
        # Set NONE cost (last column)
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
    
    for field_idx, field in enumerate(schema_fields):
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
            # Create default NONE assignments using schema fields
            try:
                schema_obj = utils.load_contract_schema()
                schema_fields = schema_obj.get('fields', [])
                assignments = {}
                for field in schema_fields:
                    assignments[field] = {
                        'assignment_type': 'NONE',
                        'candidate_index': None,
                        'cost': none_bias,
                        'field': field,
                    }
                results[sha256] = assignments
            except Exception as schema_error:
                print(f"Could not create fallback assignments: {schema_error}")
                results[sha256] = {}
    
    return results
