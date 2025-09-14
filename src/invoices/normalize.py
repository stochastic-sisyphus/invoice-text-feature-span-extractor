"""Normalization module for cleaning predicted field values."""

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Optional, Tuple, Dict, Any

from dateutil import parser as date_parser

# Document text normalization version
NORMALIZE_VERSION = "doctext_nfc_newline_v1"


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
    Normalize a field value with strict validation for precision.
    
    Args:
        field: Field name from schema
        raw_text: Raw text value to normalize
        
    Returns:
        Dictionary with normalized value, currency_code (if applicable), and raw_text
    """
    # CRITICAL: Validate length first - reject obviously bad extractions
    text_length = len(raw_text.strip())
    
    # Field-specific length limits for precision
    field_length_limits = {
        'invoice_number': 30,
        'account_number': 30,
        'customer_account': 30,
        'invoice_date': 25,
        'due_date': 25,
        'issue_date': 25,
        'total_amount': 25,
        'subtotal': 25,
        'tax_amount': 25,
        'discount': 25,
        'currency': 10,
        'contact_phone': 30,
        'contact_email': 60,
        'routing_number': 25,
        'swift_code': 15,
        'tax_id': 30,
        'purchase_order': 50,
        'invoice_reference': 50,
        'contact_name': 50,
    }
    
    max_length = field_length_limits.get(field, 150)  # Default for address fields
    
    # Reject if too long (fail normalization)
    if text_length > max_length:
        return {
            'value': None,  # Normalization failed
            'raw_text': raw_text,
            'currency_code': None,
        }
    
    # Date fields with enhanced validation
    if field in ['invoice_date', 'due_date', 'issue_date']:
        # Reject if obviously not a date (too long or no date indicators)
        if text_length > 25 or text_length < 4:
            return {'value': None, 'raw_text': raw_text, 'currency_code': None}
        
        normalized_value, original_text = normalize_date(raw_text)
        return {
            'value': normalized_value,
            'raw_text': original_text,
            'currency_code': None,
        }
    
    # Amount fields with enhanced validation  
    elif field in ['total_amount', 'subtotal', 'tax_amount', 'discount']:
        # Reject if obviously not an amount
        if text_length > 25 or text_length < 1:
            return {'value': None, 'raw_text': raw_text, 'currency_code': None}
        
        # Must have digits to be a valid amount
        if not any(c.isdigit() for c in raw_text):
            return {'value': None, 'raw_text': raw_text, 'currency_code': None}
        
        normalized_value, currency_code, original_text = normalize_amount(raw_text)
        return {
            'value': normalized_value,
            'raw_text': original_text, 
            'currency_code': currency_code,
        }
    
    # ID fields with enhanced validation
    elif field in ['invoice_number', 'account_number', 'customer_account', 'routing_number', 'swift_code', 'tax_id']:
        # Reject if too long or obviously not an ID
        if text_length > 30 or text_length < 2:
            return {'value': None, 'raw_text': raw_text, 'currency_code': None}
        
        normalized_value, original_text = normalize_id(raw_text)
        return {
            'value': normalized_value,
            'raw_text': original_text,
            'currency_code': None,
        }
    
    # Currency field with strict validation
    elif field == 'currency':
        # Currency should be very short
        if text_length > 10 or text_length < 1:
            return {'value': None, 'raw_text': raw_text, 'currency_code': None}
        
        normalized_value, original_text = normalize_text(raw_text)
        return {
            'value': normalized_value,
            'raw_text': original_text,
            'currency_code': None,
        }
    
    # Contact fields with type validation
    elif field == 'contact_email':
        if text_length > 60 or '@' not in raw_text or '.' not in raw_text:
            return {'value': None, 'raw_text': raw_text, 'currency_code': None}
        
        normalized_value, original_text = normalize_text(raw_text)
        return {
            'value': normalized_value,
            'raw_text': original_text,
            'currency_code': None,
        }
    
    elif field == 'contact_phone':
        # Phone numbers should have digits
        digit_count = sum(1 for c in raw_text if c.isdigit())
        if text_length > 30 or digit_count < 7:
            return {'value': None, 'raw_text': raw_text, 'currency_code': None}
        
        normalized_value, original_text = normalize_text(raw_text)
        return {
            'value': normalized_value,
            'raw_text': original_text,
            'currency_code': None,
        }
    
    # Address and name fields (moderate length limits)
    else:
        # Apply moderate length constraint
        if text_length > max_length:
            return {'value': None, 'raw_text': raw_text, 'currency_code': None}
        
        normalized_value, original_text = normalize_text(raw_text)
        return {
            'value': normalized_value,
            'raw_text': original_text,
            'currency_code': None,
        }


def normalize_document_text(text: str) -> str:
    """
    Normalize document text for consistent character-level operations.
    
    Implements NORMALIZE_VERSION="doctext_nfc_newline_v1":
    - Unicode NFC normalization
    - CR/LF → \n conversion
    - Strip leading/trailing whitespace
    
    Args:
        text: Raw document text
        
    Returns:
        Normalized document text
    """
    if not text:
        return ""
    
    # Unicode NFC normalization
    normalized = unicodedata.normalize("NFC", text)
    
    # Convert CR/LF to newlines
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    
    # Strip leading/trailing whitespace
    normalized = normalized.strip()
    
    return normalized


def text_len(text: str) -> int:
    """
    Compute text length for normalization guard checksum.
    
    Args:
        text: Text to measure
        
    Returns:
        Character count of text
    """
    return len(text) if text else 0


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
