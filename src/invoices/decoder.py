"""Decoder module for Hungarian assignment with NONE option per field."""

import warnings
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

from . import paths, utils, ingest, candidates


# Header field set as specified (v1 contract)
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

# Expanded field set for v2 contract
HEADER_FIELDS_V2 = [
    'invoice_number',
    'invoice_date',
    'due_date',
    'issue_date',
    'total_amount',
    'subtotal',
    'tax_amount',
    'discount',
    'currency',
    'remittance_address',
    'bill_to_address',
    'ship_to_address',
    'vendor_name',
    'vendor_address',
    'customer_name',
    'customer_account',
    'purchase_order',
    'invoice_reference',
    'payment_terms',
    'bank_account',
    'routing_number',
    'swift_code',
    'notes',
    'tax_id',
    'contact_name',
    'contact_email',
    'contact_phone',
]

# Default NONE bias (high cost to encourage abstaining)
DEFAULT_NONE_BIAS = 10.0


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


def decode_document(sha256: str, none_bias: float = DEFAULT_NONE_BIAS, contract_version: str = "v1") -> Dict[str, Any]:
    """
    Decode a single document using Hungarian assignment.
    
    Args:
        sha256: Document SHA256 hash
        none_bias: Cost for NONE assignment (higher = more likely to abstain)
        contract_version: Contract version ("v1" or "v2")
        
    Returns:
        Assignment results for each field
    """
    # Select field set based on contract version
    field_set = HEADER_FIELDS_V2 if contract_version == "v2" else HEADER_FIELDS
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
    
    # Build cost matrix: fields × (candidates + NONE)
    # Each field can be assigned to any candidate or to NONE
    cost_matrix = np.full((n_fields, n_candidates + 1), 2.0)  # Base cost
    
    # Fill candidate costs
    for field_idx, field in enumerate(field_set):
        for cand_idx, candidate in enumerate(candidates_list):
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


def decode_all_documents(none_bias: float = DEFAULT_NONE_BIAS, contract_version: str = "v1") -> Dict[str, Dict[str, Any]]:
    """Decode all documents in the index."""
    indexed_docs = ingest.get_indexed_documents()
    
    # Select field set for default NONE assignments
    field_set = HEADER_FIELDS_V2 if contract_version == "v2" else HEADER_FIELDS
    
    if indexed_docs.empty:
        print("No documents found in index")
        return {}
    
    results = {}
    
    print(f"Decoding {len(indexed_docs)} documents with NONE bias {none_bias}")
    
    for _, doc_info in indexed_docs.iterrows():
        sha256 = doc_info['sha256']
        
        try:
            assignments = decode_document(sha256, none_bias, contract_version)
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
