"""Candidate generation module for balanced span proposals with feature extraction."""

import hashlib
import random
import re
from collections import Counter
from typing import Dict, List, Any, Set, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import paths, utils, ingest, tokenize


# Candidate bucket types
BUCKET_DATE_LIKE = "date_like"
BUCKET_AMOUNT_LIKE = "amount_like" 
BUCKET_ID_LIKE = "id_like"
BUCKET_KEYWORD_PROXIMAL = "keyword_proximal"
BUCKET_RANDOM_NEGATIVE = "random_negative"

# Key invoice keywords to look for
INVOICE_KEYWORDS = {
    'invoice', 'inv', 'invoice#', 'invoice number', 'invoice no',
    'account', 'account#', 'account number', 'account no', 'acct',
    'amount due', 'total due', 'total amount', 'balance due', 'due',
    'due date', 'payment due', 'date due'
}

# Currency symbols
CURRENCY_SYMBOLS = {'$', '€', '£', '¥', '₹', '₽', 'USD', 'EUR', 'GBP', 'CAD'}


def is_date_like(text: str) -> bool:
    """Check if text looks like a date."""
    text = text.strip()
    
    # Common date patterns
    date_patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY, MM-DD-YYYY
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2}',    # MM/DD/YY, MM-DD-YY
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY/MM/DD, YYYY-MM-DD
        r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4}',  # DD MMM YYYY
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{2,4}',  # MMM DD, YYYY
    ]
    
    for pattern in date_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False


def is_amount_like(text: str) -> bool:
    """Check if text looks like a monetary amount."""
    text = text.strip()
    
    # Check for currency symbols
    has_currency_symbol = any(symbol in text for symbol in CURRENCY_SYMBOLS)
    
    # Check for decimal number patterns
    decimal_patterns = [
        r'\d+\.\d{2}',      # 123.45
        r'\d+,\d{3}',       # 1,234
        r'\d{1,3}(,\d{3})*(\.\d{2})?',  # 1,234.56
    ]
    
    has_decimal_pattern = any(re.search(pattern, text) for pattern in decimal_patterns)
    
    return has_currency_symbol or has_decimal_pattern


def is_id_like(text: str) -> bool:
    """Check if text looks like an ID or number."""
    text = text.strip()
    
    # Must be alphanumeric with reasonable length
    if not re.match(r'^[A-Za-z0-9\-_#]+$', text):
        return False
    
    # Length threshold for IDs
    if len(text) < 3 or len(text) > 20:
        return False
    
    # Must contain at least one digit or be mixed alphanumeric
    if not (any(c.isdigit() for c in text) or 
           (any(c.isalpha() for c in text) and any(c.isdigit() for c in text))):
        return False
    
    return True


def compute_text_features(text: str) -> Dict[str, Any]:
    """Compute text-based features for a candidate."""
    text_clean = text.strip()
    
    features = {
        'text_length': len(text_clean),
        'digit_ratio': sum(1 for c in text_clean if c.isdigit()) / max(len(text_clean), 1),
        'uppercase_ratio': sum(1 for c in text_clean if c.isupper()) / max(len(text_clean), 1),
        'currency_flag': any(symbol in text_clean for symbol in CURRENCY_SYMBOLS),
    }
    
    # Hashed n-grams (unigrams and bigrams)
    words = text_clean.lower().split()
    
    # Unigram hashes
    unigram_hashes = [hashlib.md5(word.encode()).hexdigest()[:8] for word in words]
    features['unigram_hash'] = ','.join(unigram_hashes[:3])  # Limit to first 3
    
    # Bigram hashes
    if len(words) > 1:
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        bigram_hashes = [hashlib.md5(bigram.encode()).hexdigest()[:8] for bigram in bigrams]
        features['bigram_hash'] = ','.join(bigram_hashes[:2])  # Limit to first 2
    else:
        features['bigram_hash'] = ''
    
    return features


def compute_geometry_features(bbox_norm: Tuple[float, float, float, float], page_width: float, page_height: float) -> Dict[str, Any]:
    """Compute geometry-based features for a candidate."""
    x0, y0, x1, y1 = bbox_norm
    
    # Center coordinates
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    
    # Width and height
    w = x1 - x0
    h = y1 - y0
    
    # Distance to margins
    distance_to_top = cy
    distance_to_bottom = 1.0 - cy
    distance_to_left = cx
    distance_to_right = 1.0 - cx
    
    # Distance to center of page
    distance_to_center = np.sqrt((cx - 0.5)**2 + (cy - 0.5)**2)
    
    return {
        'center_x': cx,
        'center_y': cy,
        'width_norm': w,
        'height_norm': h,
        'distance_to_top': distance_to_top,
        'distance_to_bottom': distance_to_bottom,
        'distance_to_left': distance_to_left,
        'distance_to_right': distance_to_right,
        'distance_to_center': distance_to_center,
    }


def compute_style_features(font_size: float, is_bold: bool, is_italic: bool, font_hash: str, page_font_sizes: List[float]) -> Dict[str, Any]:
    """Compute style-based features for a candidate."""
    # Font size z-score relative to page
    if page_font_sizes and len(page_font_sizes) > 1:
        page_mean = np.mean(page_font_sizes)
        page_std = np.std(page_font_sizes)
        font_size_z = (font_size - page_mean) / max(page_std, 1e-6)
    else:
        font_size_z = 0.0
    
    return {
        'font_size': font_size,
        'font_size_z': font_size_z,
        'is_bold': is_bold,
        'is_italic': is_italic,
        'font_hash': font_hash,
    }


def compute_context_features(token_idx: int, tokens_df: pd.DataFrame, page_idx: int) -> Dict[str, Any]:
    """Compute context-based features for a candidate."""
    # Get tokens on the same page
    page_tokens = tokens_df[tokens_df['page_idx'] == page_idx].copy()
    page_tokens = page_tokens.sort_values('token_idx')
    
    # Find the current token's position in page
    token_pos = -1
    for i, (_, row) in enumerate(page_tokens.iterrows()):
        if row['token_idx'] == token_idx:
            token_pos = i
            break
    
    if token_pos == -1:
        return {'context_bow_hash': '', 'same_row_count': 0, 'block_index': 0}
    
    # Get ±5 neighboring tokens
    start_idx = max(0, token_pos - 5)
    end_idx = min(len(page_tokens), token_pos + 6)
    context_tokens = page_tokens.iloc[start_idx:end_idx]
    
    # Bag of words hash for context
    context_words = []
    for _, row in context_tokens.iterrows():
        if row['token_idx'] != token_idx:  # Exclude current token
            context_words.append(row['text'].lower().strip())
    
    context_bow = ' '.join(sorted(context_words))  # Sort for deterministic hash
    context_bow_hash = hashlib.md5(context_bow.encode()).hexdigest()[:16]
    
    # Same-row alignment count (tokens with similar y-coordinate)
    current_token = page_tokens.iloc[token_pos]
    current_y = current_token['bbox_norm_y0']
    same_row_count = 0
    
    for _, row in page_tokens.iterrows():
        if abs(row['bbox_norm_y0'] - current_y) < 0.01:  # Within 1% of page height
            same_row_count += 1
    
    # Block index (simplified - use the block_id from tokenizer)
    block_index = current_token.get('block_id', 0)
    
    return {
        'context_bow_hash': context_bow_hash,
        'same_row_count': same_row_count,
        'block_index': block_index,
    }


def compute_local_density(token_idx: int, tokens_df: pd.DataFrame, page_idx: int, window_size: float = 0.1) -> float:
    """Compute local token density around a candidate."""
    # Get tokens on the same page
    page_tokens = tokens_df[tokens_df['page_idx'] == page_idx]
    
    # Find current token
    current_token = page_tokens[page_tokens['token_idx'] == token_idx]
    if current_token.empty:
        return 0.0
    
    current_token = current_token.iloc[0]
    cx = (current_token['bbox_norm_x0'] + current_token['bbox_norm_x1']) / 2
    cy = (current_token['bbox_norm_y0'] + current_token['bbox_norm_y1']) / 2
    
    # Count tokens within window
    density_count = 0
    for _, token in page_tokens.iterrows():
        token_cx = (token['bbox_norm_x0'] + token['bbox_norm_x1']) / 2
        token_cy = (token['bbox_norm_y0'] + token['bbox_norm_y1']) / 2
        
        if (abs(token_cx - cx) <= window_size / 2 and 
            abs(token_cy - cy) <= window_size / 2):
            density_count += 1
    
    # Density per unit area
    area = window_size * window_size
    return density_count / area


def is_remittance_band(bbox_norm: Tuple[float, float, float, float]) -> bool:
    """Check if candidate is in remittance band (bottom strip)."""
    _, y0, _, y1 = bbox_norm
    
    # Consider bottom 15% of page as potential remittance band
    return (y0 + y1) / 2 > 0.85


def find_keyword_proximal_candidates(tokens_df: pd.DataFrame, page_idx: int) -> List[int]:
    """Find candidates adjacent to invoice keywords."""
    page_tokens = tokens_df[tokens_df['page_idx'] == page_idx].copy()
    page_tokens = page_tokens.sort_values(['bbox_norm_y0', 'bbox_norm_x0'])  # Sort by position
    
    candidates = []
    
    for i, (_, token) in enumerate(page_tokens.iterrows()):
        text_lower = token['text'].lower().strip()
        
        # Check if this token is a keyword
        if any(keyword in text_lower for keyword in INVOICE_KEYWORDS):
            # Look for adjacent tokens (right and below)
            for j in range(i+1, min(i+4, len(page_tokens))):  # Check next 3 tokens
                next_token = page_tokens.iloc[j]
                
                # Check if close enough spatially
                dx = abs(next_token['bbox_norm_x0'] - token['bbox_norm_x1'])
                dy = abs(next_token['bbox_norm_y0'] - token['bbox_norm_y0'])
                
                if dx < 0.1 and dy < 0.05:  # Close enough spatially
                    candidates.append(next_token['token_idx'])
                    break  # Take first adjacent candidate
    
    return candidates


def compute_overlap_iou(bbox1: Tuple[float, float, float, float], 
                       bbox2: Tuple[float, float, float, float]) -> float:
    """Compute IoU (Intersection over Union) of two bounding boxes."""
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


def apply_nms(candidates: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Apply Non-Maximum Suppression to remove overlapping candidates."""
    if not candidates:
        return []
    
    # Sort by priority (header region > body region; larger bbox for amounts)
    def get_priority(candidate):
        bbox_norm = (candidate['bbox_norm_x0'], candidate['bbox_norm_y0'], 
                    candidate['bbox_norm_x1'], candidate['bbox_norm_y1'])
        
        # Header region priority (top 30% of page)
        is_header = candidate['center_y'] < 0.3
        
        # Size priority for amount-like candidates
        bbox_area = (candidate['bbox_norm_x1'] - candidate['bbox_norm_x0']) * \
                   (candidate['bbox_norm_y1'] - candidate['bbox_norm_y0'])
        
        return (is_header, bbox_area)
    
    candidates.sort(key=get_priority, reverse=True)
    
    # Apply NMS
    kept = []
    for candidate in candidates:
        candidate_bbox = (candidate['bbox_norm_x0'], candidate['bbox_norm_y0'],
                         candidate['bbox_norm_x1'], candidate['bbox_norm_y1'])
        
        should_keep = True
        for kept_candidate in kept:
            kept_bbox = (kept_candidate['bbox_norm_x0'], kept_candidate['bbox_norm_y0'],
                        kept_candidate['bbox_norm_x1'], kept_candidate['bbox_norm_y1'])
            
            if compute_overlap_iou(candidate_bbox, kept_bbox) > iou_threshold:
                should_keep = False
                break
        
        if should_keep:
            kept.append(candidate)
    
    return kept


def generate_candidates(sha256: str) -> int:
    """Generate candidates for a single document."""
    # Check if already exists (idempotency)
    candidates_path = paths.get_candidates_path(sha256)
    if candidates_path.exists():
        existing_df = pd.read_parquet(candidates_path)
        print(f"Candidates already exist for {sha256[:16]}: {len(existing_df)} candidates")
        return len(existing_df)
    
    # Get tokens
    tokens_df = tokenize.get_document_tokens(sha256)
    if tokens_df.empty:
        print(f"No tokens found for {sha256[:16]}")
        return 0
    
    doc_info = ingest.get_document_info(sha256)
    if not doc_info:
        raise ValueError(f"Document not found: {sha256}")
    
    doc_id = doc_info['doc_id']
    
    # Set random seed based on sha256 for deterministic random negatives
    random.seed(int(sha256[:8], 16))
    np.random.seed(int(sha256[:8], 16))
    
    all_candidates = []
    bucket_counts = {
        BUCKET_DATE_LIKE: 0,
        BUCKET_AMOUNT_LIKE: 0,
        BUCKET_ID_LIKE: 0,
        BUCKET_KEYWORD_PROXIMAL: 0,
        BUCKET_RANDOM_NEGATIVE: 0,
    }
    
    # Process each page
    for page_idx in tokens_df['page_idx'].unique():
        page_tokens = tokens_df[tokens_df['page_idx'] == page_idx].copy()
        
        # Get page font sizes for z-score calculation
        page_font_sizes = page_tokens['font_size'].tolist()
        
        # Find keyword-proximal candidates first
        keyword_proximal = find_keyword_proximal_candidates(tokens_df, page_idx)
        
        for _, token in page_tokens.iterrows():
            token_idx = token['token_idx']
            text = token['text'].strip()
            
            # Skip very short or empty text
            if len(text) < 2:
                continue
            
            # Determine bucket
            bucket = None
            
            if token_idx in keyword_proximal and bucket_counts[BUCKET_KEYWORD_PROXIMAL] < 30:
                bucket = BUCKET_KEYWORD_PROXIMAL
            elif is_date_like(text) and bucket_counts[BUCKET_DATE_LIKE] < 40:
                bucket = BUCKET_DATE_LIKE
            elif is_amount_like(text) and bucket_counts[BUCKET_AMOUNT_LIKE] < 40:
                bucket = BUCKET_AMOUNT_LIKE
            elif is_id_like(text) and bucket_counts[BUCKET_ID_LIKE] < 30:
                bucket = BUCKET_ID_LIKE
            elif bucket_counts[BUCKET_RANDOM_NEGATIVE] < 20 and random.random() < 0.05:
                bucket = BUCKET_RANDOM_NEGATIVE
            
            if bucket is None:
                continue
            
            # Create candidate
            bbox_norm = (token['bbox_norm_x0'], token['bbox_norm_y0'], 
                        token['bbox_norm_x1'], token['bbox_norm_y1'])
            
            # Compute features
            text_features = compute_text_features(text)
            geometry_features = compute_geometry_features(bbox_norm, token['page_width'], token['page_height'])
            style_features = compute_style_features(
                token['font_size'], token['is_bold'], token['is_italic'], 
                token['font_hash'], page_font_sizes
            )
            context_features = compute_context_features(token_idx, tokens_df, page_idx)
            
            # Additional features
            local_density = compute_local_density(token_idx, tokens_df, page_idx)
            is_remittance = is_remittance_band(bbox_norm)
            
            candidate = {
                'candidate_id': f"{doc_id}_{page_idx}_{token_idx}",
                'doc_id': doc_id,
                'sha256': sha256,
                'page_idx': page_idx,
                'token_idx': token_idx,
                'token_id': token['token_id'],
                'text': text,
                'bucket': bucket,
                
                # Bounding box
                'bbox_norm_x0': bbox_norm[0],
                'bbox_norm_y0': bbox_norm[1],
                'bbox_norm_x1': bbox_norm[2],
                'bbox_norm_y1': bbox_norm[3],
                
                # Features
                **text_features,
                **geometry_features,
                **style_features,
                **context_features,
                'local_density': local_density,
                'is_remittance_band': is_remittance,
            }
            
            all_candidates.append(candidate)
            bucket_counts[bucket] += 1
            
            # Stop if we have enough candidates overall
            if len(all_candidates) >= 200:
                break
        
        if len(all_candidates) >= 200:
            break
    
    # Apply NMS to remove overlapping candidates
    all_candidates = apply_nms(all_candidates, iou_threshold=0.5)
    
    # Ensure we don't exceed 200 candidates
    if len(all_candidates) > 200:
        all_candidates = all_candidates[:200]
    
    # Save candidates
    if all_candidates:
        df = pd.DataFrame(all_candidates)
        df.to_parquet(candidates_path, index=False)
        
        print(f"Generated {len(all_candidates)} candidates for {doc_id}")
        for bucket, count in bucket_counts.items():
            if count > 0:
                print(f"  {bucket}: {count}")
    
    return len(all_candidates)


def generate_all_candidates() -> Dict[str, int]:
    """Generate candidates for all documents."""
    indexed_docs = ingest.get_indexed_documents()
    
    if indexed_docs.empty:
        print("No documents found in index")
        return {}
    
    results = {}
    
    print(f"Generating candidates for {len(indexed_docs)} documents")
    
    for _, doc_info in tqdm(indexed_docs.iterrows(), total=len(indexed_docs)):
        sha256 = doc_info['sha256']
        
        try:
            candidate_count = generate_candidates(sha256)
            results[sha256] = candidate_count
        except Exception as e:
            print(f"Failed to generate candidates for {sha256[:16]}: {e}")
            results[sha256] = 0
    
    return results


def get_document_candidates(sha256: str) -> pd.DataFrame:
    """Get candidates for a specific document."""
    candidates_path = paths.get_candidates_path(sha256)
    
    if not candidates_path.exists():
        return pd.DataFrame()
    
    return pd.read_parquet(candidates_path)
