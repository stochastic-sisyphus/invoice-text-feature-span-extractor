"""Normalization module for cleaning predicted field values."""

import re
from decimal import Decimal, InvalidOperation
from typing import Optional, Tuple, Dict, Any

from dateutil import parser as date_parser


def normalize_date(raw_text: str) -> Tuple[Optional[str], str]:
    """
    Normalize date text to ISO8601 format.
    
    Args:
        raw_text: Original date text from PDF
        
    Returns:
        Tuple of (normalized_value, original_raw_text)
        normalized_value is None if parsing fails
    """
    if not raw_text or not raw_text.strip():
        return None, raw_text
    
    clean_text = raw_text.strip()
    
    try:
        # Use dateutil parser for flexible date parsing
        parsed_date = date_parser.parse(clean_text, fuzzy=True)
        
        # Convert to ISO8601 date format (YYYY-MM-DD)
        iso_date = parsed_date.strftime('%Y-%m-%d')
        
        return iso_date, clean_text
        
    except (ValueError, TypeError, date_parser.ParserError) as e:
        # If parsing fails, return None but keep original text
        return None, clean_text


def normalize_amount(raw_text: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    Normalize amount text to decimal with currency code.
    
    Args:
        raw_text: Original amount text from PDF
        
    Returns:
        Tuple of (normalized_value, currency_code, original_raw_text)
        normalized_value is None if parsing fails
    """
    if not raw_text or not raw_text.strip():
        return None, None, raw_text
    
    clean_text = raw_text.strip()
    
    # Currency symbol mapping
    currency_map = {
        '$': 'USD',
        '€': 'EUR', 
        '£': 'GBP',
        '¥': 'JPY',
        '₹': 'INR',
        '₽': 'RUB',
    }
    
    # Extract currency code
    currency_code = None
    
    # Check for currency symbols
    for symbol, code in currency_map.items():
        if symbol in clean_text:
            currency_code = code
            break
    
    # Check for currency codes in text
    currency_codes = ['USD', 'EUR', 'GBP', 'JPY', 'INR', 'RUB', 'CAD', 'AUD']
    for code in currency_codes:
        if code.upper() in clean_text.upper():
            currency_code = code
            break
    
    # Extract numeric value
    # Remove currency symbols and letters, keep digits, dots, commas
    numeric_text = re.sub(r'[^\d\.\,\-]', '', clean_text)
    
    if not numeric_text:
        return None, currency_code, clean_text
    
    # Handle comma as thousands separator
    if ',' in numeric_text and '.' in numeric_text:
        # Assume comma is thousands separator if it comes before dot
        if numeric_text.rindex(',') < numeric_text.rindex('.'):
            numeric_text = numeric_text.replace(',', '')
    elif ',' in numeric_text:
        # Check if comma might be decimal separator (European style)
        parts = numeric_text.split(',')
        if len(parts) == 2 and len(parts[1]) == 2:
            # Likely decimal separator
            numeric_text = numeric_text.replace(',', '.')
        else:
            # Likely thousands separator
            numeric_text = numeric_text.replace(',', '')
    
    # Handle negative amounts
    is_negative = '-' in numeric_text or '(' in clean_text
    numeric_text = numeric_text.replace('-', '')
    
    try:
        # Parse as decimal
        amount = Decimal(numeric_text)
        
        if is_negative:
            amount = -amount
        
        # Format to two decimal places
        normalized_value = f"{amount:.2f}"
        
        return normalized_value, currency_code, clean_text
        
    except (InvalidOperation, ValueError):
        return None, currency_code, clean_text


def normalize_id(raw_text: str) -> Tuple[Optional[str], str]:
    """
    Normalize ID text by stripping zero-width and control characters.
    
    Args:
        raw_text: Original ID text from PDF
        
    Returns:
        Tuple of (normalized_value, original_raw_text)
    """
    if not raw_text or not raw_text.strip():
        return None, raw_text
    
    clean_text = raw_text.strip()
    
    # Remove zero-width and control characters but keep hyphens
    normalized = ''
    for char in clean_text:
        # Keep alphanumeric, hyphens, underscores, and basic punctuation
        if char.isalnum() or char in '-_#.':
            normalized += char
        elif char == ' ':
            normalized += char
    
    # Clean up multiple spaces
    normalized = ' '.join(normalized.split())
    
    if not normalized:
        return None, clean_text
    
    return normalized, clean_text


def normalize_text(raw_text: str) -> Tuple[Optional[str], str]:
    """
    Normalize general text fields (carrier name, document type, etc.).
    
    Args:
        raw_text: Original text from PDF
        
    Returns:
        Tuple of (normalized_value, original_raw_text)
    """
    if not raw_text or not raw_text.strip():
        return None, raw_text
    
    clean_text = raw_text.strip()
    
    # Basic normalization - remove extra whitespace
    normalized = ' '.join(clean_text.split())
    
    return normalized, clean_text


def normalize_field_value(field: str, raw_text: str) -> Dict[str, Any]:
    """
    Normalize a field value based on its type.
    
    Args:
        field: Field name from schema
        raw_text: Raw text value to normalize
        
    Returns:
        Dictionary with normalized value, currency_code (if applicable), and raw_text
    """
    # Date fields
    if field in ['InvoiceDate', 'DueDate', 'IssueDate']:
        normalized_value, original_text = normalize_date(raw_text)
        return {
            'value': normalized_value,
            'raw_text': original_text,
            'currency_code': None,
        }
    
    # Amount fields
    elif field in ['TotalAmount', 'Subtotal', 'TaxAmount', 'Discount']:
        normalized_value, currency_code, original_text = normalize_amount(raw_text)
        return {
            'value': normalized_value,
            'raw_text': original_text, 
            'currency_code': currency_code,
        }
    
    # ID/Number fields
    elif field in ['InvoiceNumber', 'CustomerAccount', 'TaxID', 'PurchaseOrder', 'InvoiceReference']:
        normalized_value, original_text = normalize_id(raw_text)
        return {
            'value': normalized_value,
            'raw_text': original_text,
            'currency_code': None,
        }
    
    # Currency field
    elif field == 'Currency':
        # Currency should be normalized as ID but validated against currency codes
        normalized_value, original_text = normalize_id(raw_text)
        return {
            'value': normalized_value,
            'raw_text': original_text,
            'currency_code': None,
        }
    
    # Text fields (addresses, names, descriptions, etc.)
    else:
        normalized_value, original_text = normalize_text(raw_text)
        return {
            'value': normalized_value,
            'raw_text': original_text,
            'currency_code': None,
        }


def normalize_assignments(assignments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize all field assignments for a document.
    
    Args:
        assignments: Raw assignments from decoder
        
    Returns:
        Normalized assignments with cleaned values
    """
    normalized = {}
    
    for field, assignment in assignments.items():
        if assignment['assignment_type'] == 'NONE':
            # No normalization needed for NONE assignments
            normalized[field] = {
                'assignment_type': 'NONE',
                'candidate_index': None,
                'cost': assignment['cost'],
                'field': field,
                'normalized_value': None,
                'raw_text': None,
                'currency_code': None,
            }
        else:
            # Normalize the candidate value
            candidate = assignment['candidate']
            raw_text = candidate.get('raw_text', candidate.get('text', ''))
            
            normalization_result = normalize_field_value(field, raw_text)
            
            normalized[field] = {
                'assignment_type': 'CANDIDATE',
                'candidate_index': assignment['candidate_index'],
                'cost': assignment['cost'],
                'field': field,
                'candidate': candidate,
                'normalized_value': normalization_result['value'],
                'raw_text': normalization_result['raw_text'],
                'currency_code': normalization_result['currency_code'],
            }
    
    return normalized
