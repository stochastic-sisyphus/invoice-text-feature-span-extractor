"""Reporting module for pipeline metrics and statistics."""

import json
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional

import pandas as pd

import hashlib
from pathlib import Path
from . import paths, utils, ingest, tokenize, candidates, emit
from .normalize import normalize_document_text, text_len, NORMALIZE_VERSION
from .candidates import char_iou


def collect_field_statistics() -> Dict[str, Dict[str, int]]:
    """Collect statistics for all fields across all documents."""
    indexed_docs = ingest.get_indexed_documents()
    
    if indexed_docs.empty:
        return {}
    
    # Load schema to get current field set
    schema = utils.load_contract_schema()
    schema_fields = schema['fields']
    
    field_stats = defaultdict(lambda: Counter())
    
    for _, doc_info in indexed_docs.iterrows():
        sha256 = doc_info['sha256']
        
        try:
            predictions = emit.get_document_predictions(sha256)
            
            if predictions and 'fields' in predictions:
                for field_name, field_data in predictions['fields'].items():
                    status = field_data.get('status', 'MISSING')
                    field_stats[field_name][status] += 1
            else:
                # No predictions found - mark all as MISSING
                for field in schema_fields:
                    field_stats[field]['MISSING'] += 1
                    
        except Exception as e:
            print(f"Error reading predictions for {sha256[:16]}: {e}")
            # Mark all fields as MISSING for this document
            for field in schema_fields:
                field_stats[field]['MISSING'] += 1
    
    return dict(field_stats)


def collect_document_statistics() -> List[Dict[str, Any]]:
    """Collect per-document statistics."""
    indexed_docs = ingest.get_indexed_documents()
    
    if indexed_docs.empty:
        return []
    
    # Load schema to get field count for error cases
    schema = utils.load_contract_schema()
    schema_fields_count = len(schema['fields'])
    
    doc_stats = []
    
    for _, doc_info in indexed_docs.iterrows():
        sha256 = doc_info['sha256']
        doc_id = doc_info['doc_id']
        
        try:
            # Get tokens info
            token_summary = tokenize.get_token_summary(sha256)
            
            # Get candidates info
            candidates_df = candidates.get_document_candidates(sha256)
            candidate_count = len(candidates_df)
            
            # Get predictions info
            predictions = emit.get_document_predictions(sha256)
            
            status_counts = Counter()
            if predictions and 'fields' in predictions:
                for field_data in predictions['fields'].values():
                    status = field_data.get('status', 'MISSING')
                    status_counts[status] += 1
            
            doc_stat = {
                'sha256': sha256,
                'doc_id': doc_id,
                'pages': token_summary.get('page_count', 0),
                'tokens': token_summary.get('token_count', 0),
                'candidates': candidate_count,
                'predicted': status_counts.get('PREDICTED', 0),
                'abstain': status_counts.get('ABSTAIN', 0),
                'missing': status_counts.get('MISSING', 0),
            }
            
            doc_stats.append(doc_stat)
            
        except Exception as e:
            print(f"Error collecting stats for {sha256[:16]}: {e}")
            doc_stats.append({
                'sha256': sha256,
                'doc_id': doc_id,
                'pages': 0,
                'tokens': 0,
                'candidates': 0,
                'predicted': 0,
                'abstain': 0,
                'missing': schema_fields_count,
                'error': str(e),
            })
    
    return doc_stats


def collect_review_queue_statistics() -> Dict[str, Any]:
    """Collect review queue statistics."""
    try:
        review_df = emit.get_review_queue()
        
        if review_df.empty:
            return {
                'total_entries': 0,
                'by_field': {},
                'by_reason': {},
            }
        
        # Count by field
        by_field = review_df['field'].value_counts().to_dict()
        
        # Count by reason
        by_reason = review_df['reason'].value_counts().to_dict()
        
        return {
            'total_entries': len(review_df),
            'by_field': by_field,
            'by_reason': by_reason,
        }
        
    except Exception as e:
        print(f"Error reading review queue: {e}")
        return {
            'total_entries': 0,
            'by_field': {},
            'by_reason': {},
            'error': str(e),
        }


def collect_coverage_statistics() -> Dict[str, Any]:
    """Collect coverage probe statistics."""
    try:
        coverage_stats = candidates.get_coverage_statistics()
        
        if not coverage_stats:
            return {
                'total_documents': 0,
                'coverage_probes': {},
                'bucket_distribution': {},
            }
        
        # Calculate coverage percentages
        total_candidates = coverage_stats.get('total_candidates', 0)
        bucket_dist = dict(coverage_stats.get('bucket_distribution', {}))
        
        coverage_probes = {}
        
        # Field-specific coverage probes
        field_types = {
            'invoice_number': 'id_like',
            'invoice_date': 'date_like', 
            'due_date': 'date_like',
            'total_amount_due': 'amount_like',
            'account_number': 'id_like',
        }
        
        for field, expected_type in field_types.items():
            type_count = bucket_dist.get(expected_type, 0)
            proximal_count = bucket_dist.get('keyword_proximal', 0)
            
            # Cue-coverage: percentage with keyword proximity
            cue_coverage = (proximal_count / max(total_candidates, 1)) * 100
            
            # Region-coverage: percentage of expected type found
            region_coverage = (type_count / max(total_candidates, 1)) * 100
            
            coverage_probes[field] = {
                'cue_coverage_percent': round(cue_coverage, 1),
                'region_coverage_percent': round(region_coverage, 1),
                'expected_type_count': type_count,
                'total_candidates': total_candidates,
            }
        
        return {
            'total_documents': coverage_stats.get('total_documents', 0),
            'documents_with_candidates': coverage_stats.get('documents_with_candidates', 0),
            'total_candidates': total_candidates,
            'coverage_probes': coverage_probes,
            'bucket_distribution': bucket_dist,
        }
    
    except Exception as e:
        print(f"Warning: Failed to collect coverage statistics: {e}")
        return {
            'total_documents': 0,
            'coverage_probes': {},
            'bucket_distribution': {},
            'error': str(e),
        }


def print_field_report(field_stats: Dict[str, Dict[str, int]]) -> None:
    """Print formatted field statistics report."""
    print("\n" + "="*80)
    print("FIELD STATISTICS")
    print("="*80)
    
    if not field_stats:
        print("No field statistics available")
        return
    
    # Load schema to get current field set
    schema = utils.load_contract_schema()
    schema_fields = schema['fields']
    
    # Print header
    print(f"{'Field':<20} {'Predicted':<10} {'Abstain':<10} {'Missing':<10} {'Total':<10}")
    print("-" * 70)
    
    # Print each field
    for field in schema_fields:
        stats = field_stats.get(field, {})
        predicted = stats.get('PREDICTED', 0)
        abstain = stats.get('ABSTAIN', 0)
        missing = stats.get('MISSING', 0)
        total = predicted + abstain + missing
        
        print(f"{field:<20} {predicted:<10} {abstain:<10} {missing:<10} {total:<10}")
    
    # Print totals
    print("-" * 70)
    total_predicted = sum(stats.get('PREDICTED', 0) for stats in field_stats.values())
    total_abstain = sum(stats.get('ABSTAIN', 0) for stats in field_stats.values())
    total_missing = sum(stats.get('MISSING', 0) for stats in field_stats.values())
    grand_total = total_predicted + total_abstain + total_missing
    
    print(f"{'TOTAL':<20} {total_predicted:<10} {total_abstain:<10} {total_missing:<10} {grand_total:<10}")


def print_document_report(doc_stats: List[Dict[str, Any]]) -> None:
    """Print formatted document statistics report."""
    print("\n" + "="*80)
    print("DOCUMENT STATISTICS")  
    print("="*80)
    
    if not doc_stats:
        print("No document statistics available")
        return
    
    # Print header
    print(f"{'Doc ID':<20} {'Pages':<6} {'Tokens':<8} {'Candidates':<11} {'Pred':<5} {'Abst':<5} {'Miss':<5}")
    print("-" * 70)
    
    # Print each document
    for stat in doc_stats:
        doc_id = stat['doc_id'][:20]  # Truncate for display
        pages = stat.get('pages', 0)
        tokens = stat.get('tokens', 0)
        candidates = stat.get('candidates', 0)
        predicted = stat.get('predicted', 0)
        abstain = stat.get('abstain', 0)
        missing = stat.get('missing', 0)
        
        print(f"{doc_id:<20} {pages:<6} {tokens:<8} {candidates:<11} {predicted:<5} {abstain:<5} {missing:<5}")
    
    # Print summary statistics
    print("-" * 70)
    if doc_stats:
        avg_pages = sum(s.get('pages', 0) for s in doc_stats) / len(doc_stats)
        avg_tokens = sum(s.get('tokens', 0) for s in doc_stats) / len(doc_stats)
        avg_candidates = sum(s.get('candidates', 0) for s in doc_stats) / len(doc_stats)
        
        print(f"{'AVERAGES':<20} {avg_pages:<6.1f} {avg_tokens:<8.0f} {avg_candidates:<11.1f}")


def print_review_queue_report(review_stats: Dict[str, Any]) -> None:
    """Print formatted review queue statistics."""
    print("\n" + "="*80)
    print("REVIEW QUEUE STATISTICS")
    print("="*80)
    
    total_entries = review_stats.get('total_entries', 0)
    print(f"Total entries in review queue: {total_entries}")
    
    if total_entries == 0:
        print("Review queue is empty")
        return
    
    # By field
    by_field = review_stats.get('by_field', {})
    if by_field:
        print("Entries by field:")
        for field, count in sorted(by_field.items()):
            print(f"  {field}: {count}")
    
    # By reason
    by_reason = review_stats.get('by_reason', {})
    if by_reason:
        print("Entries by reason:")
        for reason, count in sorted(by_reason.items()):
            print(f"  {reason}: {count}")


def print_coverage_report(coverage_stats: Dict[str, Any]) -> None:
    """Print coverage probe report."""
    print("\nCOVERAGE PROBES")
    print("="*50)
    
    total_docs = coverage_stats.get('total_documents', 0)
    docs_with_candidates = coverage_stats.get('documents_with_candidates', 0)
    total_candidates = coverage_stats.get('total_candidates', 0)
    
    print(f"Documents processed: {total_docs}")
    print(f"Documents with candidates: {docs_with_candidates}")
    print(f"Total candidates: {total_candidates}")
    
    # Coverage probes by field
    coverage_probes = coverage_stats.get('coverage_probes', {})
    if coverage_probes:
        print("\nField Coverage Probes:")
        print(f"{'Field':<20} {'Cue Coverage':<15} {'Region Coverage':<15}")
        print("-" * 50)
        
        for field, stats in coverage_probes.items():
            cue_pct = stats.get('cue_coverage_percent', 0)
            region_pct = stats.get('region_coverage_percent', 0)
            print(f"{field:<20} {cue_pct:>13.1f}% {region_pct:>13.1f}%")
    
    # Bucket distribution
    bucket_dist = coverage_stats.get('bucket_distribution', {})
    if bucket_dist:
        print("\nBucket Distribution:")
        for bucket, count in sorted(bucket_dist.items()):
            percentage = (count / max(total_candidates, 1)) * 100
            print(f"  {bucket}: {count} ({percentage:.1f}%)")


def generate_report() -> Dict[str, Any]:
    """
    Generate comprehensive pipeline report.
    
    Returns:
        Report data dictionary
    """
    print("Generating pipeline report...")
    
    with utils.Timer("Report generation"):
        # Collect all statistics
        field_stats = collect_field_statistics()
        doc_stats = collect_document_statistics()
        review_stats = collect_review_queue_statistics()
        coverage_stats = collect_coverage_statistics()
        
        # Print reports
        print_field_report(field_stats)
        print_document_report(doc_stats)
        print_review_queue_report(review_stats)
        print_coverage_report(coverage_stats)
        
        # Create summary
        report_data = {
            'field_statistics': field_stats,
            'document_statistics': doc_stats,
            'review_queue_statistics': review_stats,
            'coverage_statistics': coverage_stats,
            'generated_at': utils.get_current_utc_iso(),
            **utils.get_version_info(),
        }
        
        return report_data


# Label Studio field mapping (canonical)
LS_FIELD_MAPPING = {
    'InvoiceNumber': 'invoice_number',
    'InvoiceDate': 'invoice_date', 
    'DueDate': 'due_date',
    'IssueDate': 'issue_date',
    'TotalAmount': 'total_amount',
    'Subtotal': 'subtotal',
    'TaxAmount': 'tax_amount',
    'Discount': 'discount',
    'Currency': 'currency',
    'RemittanceAddress': 'remittance_address',
    'BillToAddress': 'bill_to_address',
    'ShipToAddress': 'ship_to_address',
    'VendorName': 'vendor_name',
    'VendorAddress': 'vendor_address',
    'CustomerName': 'customer_name',
    'CustomerAccount': 'customer_account',
    'PurchaseOrder': 'purchase_order',
    'InvoiceReference': 'invoice_reference',
    'PaymentTerms': 'payment_terms',
    'BankAccount': 'bank_account',
    'RoutingNumber': 'routing_number',
    'SWIFTCode': 'swift_code',
    'Notes': 'notes',
    'TaxID': 'tax_id',
    'ContactName': 'contact_name',
    'ContactEmail': 'contact_email',
    'ContactPhone': 'contact_phone',
    'LineItemDescription': 'line_item_description',
    'LineItemQuantity': 'line_item_quantity',
    'LineItemUnitPrice': 'line_item_unit_price',
    'LineItemTotal': 'line_item_total',
}


def pull_ls_annotations() -> Dict[str, Any]:
    """
    Pull annotations from Label Studio via HTTP API using environment variables.
    
    Environment variables required:
        LS_BASE_URL: Label Studio base URL
        LS_API_TOKEN: Label Studio API token  
        LS_PROJECT_ID: Label Studio project ID
        
    Returns:
        Pull summary statistics
    """
    import os
    import urllib.request
    import urllib.parse
    import urllib.error
    
    # Get environment variables
    base_url = os.environ.get('LS_BASE_URL')
    api_token = os.environ.get('LS_API_TOKEN')
    project_id = os.environ.get('LS_PROJECT_ID')
    
    if not all([base_url, api_token, project_id]):
        missing_vars = []
        if not base_url: missing_vars.append('LS_BASE_URL')
        if not api_token: missing_vars.append('LS_API_TOKEN')
        if not project_id: missing_vars.append('LS_PROJECT_ID')
        
        print(f"Missing environment variables: {', '.join(missing_vars)}")
        print("0 docs, 0 annotations pulled")
        return {'tasks': 0, 'annotations': 0, 'pulled': 0}
    
    # Ensure output directory exists
    paths.get_labels_raw_dir().mkdir(parents=True, exist_ok=True)
    
    try:
        # Build API URL for export
        base_url = base_url.rstrip('/')
        export_url = f"{base_url}/api/projects/{project_id}/export?exportType=JSON"
        
        # Create request with authorization header
        req = urllib.request.Request(export_url)
        req.add_header('Authorization', f'Token {api_token}')
        req.add_header('Content-Type', 'application/json')
        
        # Make the request
        with urllib.request.urlopen(req) as response:
            data = response.read()
            tasks = json.loads(data.decode('utf-8'))
        
        if not tasks:
            print("0 docs, 0 annotations pulled")
            return {'tasks': 0, 'annotations': 0, 'pulled': 0}
        
        # Write to annotations.jsonl
        output_path = paths.get_data_dir() / "labels" / "annotations.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for task in tasks:
                f.write(json.dumps(task, ensure_ascii=False) + '\n')
        
        total_annotations = sum(
            len(ann.get('result', []))
            for task in tasks
            for ann in task.get('annotations', [])
        )
        
        print(f"Pulled {len(tasks)} tasks with {total_annotations} annotations")
        return {
            'tasks': len(tasks),
            'annotations': total_annotations,
            'pulled': total_annotations
        }
        
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}")
        print("0 docs, 0 annotations pulled")
        return {'tasks': 0, 'annotations': 0, 'pulled': 0}
    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason}")
        print("0 docs, 0 annotations pulled") 
        return {'tasks': 0, 'annotations': 0, 'pulled': 0}
    except Exception as e:
        print(f"Pull failed: {e}")
        print("0 docs, 0 annotations pulled")
        return {'tasks': 0, 'annotations': 0, 'pulled': 0}


def import_label_studio_annotations(input_path: str, allow_unnormalized: bool = False) -> Dict[str, Any]:
    """
    Import Label Studio annotations into data/labels/raw/{sha}.jsonl format.
    
    Args:
        input_path: Path to Label Studio JSON export file
        allow_unnormalized: If True, skip presence check for normalize_version/text_len
        
    Returns:
        Import summary statistics
    """
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Ensure directories exist
    paths.get_labels_raw_dir().mkdir(parents=True, exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        annotations_data = json.load(f)
    
    if not annotations_data:
        print("0 docs, 0 spans, 0 imported")
        return {'documents': 0, 'annotations': 0, 'imported': 0}
    
    imported_docs = 0
    total_annotations = 0
    imported_annotations = 0
    
    # Group by document
    docs_by_sha = {}
    for task in annotations_data:
        task_data = task.get('data', {})
        sha256 = task_data.get('sha256')
        if not sha256:
            continue
            
        if sha256 not in docs_by_sha:
            docs_by_sha[sha256] = {
                'task_data': task_data,
                'annotations': []
            }
        
        # Extract annotations
        for annotation in task.get('annotations', []):
            for result in annotation.get('result', []):
                if result.get('type') == 'labels':
                    # Extract labeled span
                    value = result.get('value', {})
                    labels = value.get('labels', [])
                    start = value.get('start')
                    end = value.get('end')
                    text = value.get('text', '')
                    
                    if labels and start is not None and end is not None:
                        for label in labels:
                            docs_by_sha[sha256]['annotations'].append({
                                'label': label,
                                'char_start': int(start),
                                'char_end': int(end),
                                'text': text
                            })
                            total_annotations += 1
    
    # Process each document
    for sha256, doc_data in docs_by_sha.items():
        task_data = doc_data['task_data']
        doc_annotations = doc_data['annotations']
        
        if not doc_annotations:
            continue
        
        # Validate normalization guard
        doc_text = task_data.get('text', '')
        expected_normalize_version = task_data.get('normalize_version')
        expected_text_len = task_data.get('text_len')
        
        # Compute local normalization
        normalized_text = normalize_document_text(doc_text)
        actual_text_len = text_len(normalized_text)
        
        if not allow_unnormalized:
            if expected_normalize_version is None or expected_text_len is None:
                raise ValueError(
                    f"Document {sha256}: missing normalize_version or text_len. "
                    f"Use --allow-unnormalized to proceed."
                )
            
            if expected_normalize_version != NORMALIZE_VERSION:
                raise ValueError(
                    f"Document {sha256}: normalize_version mismatch. "
                    f"Expected: {NORMALIZE_VERSION}, Observed: {expected_normalize_version}"
                )
            
            if expected_text_len != actual_text_len:
                raise ValueError(
                    f"Document {sha256}: text_len mismatch. "
                    f"Expected: {expected_text_len}, Observed: {actual_text_len}"
                )
        
        # Write raw annotations for this document
        raw_path = paths.get_labels_raw_path(sha256)
        with open(raw_path, 'w', encoding='utf-8') as f:
            for annotation in doc_annotations:
                # Map LS label to contract field
                ls_label = annotation['label']
                contract_field = LS_FIELD_MAPPING.get(ls_label, ls_label.lower())
                
                row = {
                    'sha': sha256,
                    'label': contract_field,
                    'char_start': annotation['char_start'],
                    'char_end': annotation['char_end'],
                    'text': annotation['text'],
                    'normalize_version': NORMALIZE_VERSION,
                    'text_len': actual_text_len
                }
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        
        imported_docs += 1
        imported_annotations += len(doc_annotations)
        
        print(f"Imported {sha256[:16]}: {len(doc_annotations)} annotations")
    
    summary = {
        'documents': len(docs_by_sha),
        'annotations': total_annotations,
        'imported': imported_annotations
    }
    
    print(f"Imported {imported_docs} documents with {imported_annotations} annotations.")
    return summary


def align_labels(sha256: Optional[str] = None, iou_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Align imported labels with candidates using character-range IoU.
    
    Args:
        sha256: Process specific document (if None, process all)
        iou_threshold: Minimum IoU for positive match
        
    Returns:
        Alignment summary statistics
    """
    # Ensure directories exist
    paths.get_labels_aligned_dir().mkdir(parents=True, exist_ok=True)
    
    if sha256:
        docs_to_process = [sha256]
    else:
        # Find all documents with raw labels
        raw_dir = paths.get_labels_raw_dir()
        if not raw_dir.exists():
            print("0 docs, 0 spans, 0 aligned")
            return {'documents': 0, 'spans': 0, 'aligned': 0}
        
        docs_to_process = [
            f.stem for f in raw_dir.glob('*.jsonl')
        ]
    
    if not docs_to_process:
        print("0 docs, 0 spans, 0 aligned")
        return {'documents': 0, 'spans': 0, 'aligned': 0}
    
    total_docs = 0
    total_gold = 0
    total_matched = 0
    total_unmatched_gold = 0
    all_labels_present = set()
    
    index_rows = []
    
    for sha in docs_to_process:
        raw_path = paths.get_labels_raw_path(sha)
        if not raw_path.exists():
            continue
        
        # Load raw labels for this document
        gold_spans = []
        with open(raw_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    gold_spans.append(row)
        
        if not gold_spans:
            continue
        
        # Load candidates for this document
        candidates_df = candidates.get_document_candidates(sha)
        if candidates_df.empty:
            print(f"No candidates for {sha[:16]}, skipping alignment")
            continue
        
        # Track labels present in this document
        doc_labels = set(span['label'] for span in gold_spans)
        all_labels_present.update(doc_labels)
        
        # Align gold spans to candidates
        aligned_rows = []
        matched_gold = 0
        unmatched_gold = 0
        
        # Group gold spans by label
        gold_by_label = {}
        for span in gold_spans:
            label = span['label']
            if label not in gold_by_label:
                gold_by_label[label] = []
            gold_by_label[label].append(span)
        
        # Process each label
        for label, label_gold_spans in gold_by_label.items():
            # Find best IoU matches for this label
            label_candidates = candidates_df[candidates_df.get('normalize_version', '') == NORMALIZE_VERSION].copy()
            if label_candidates.empty:
                continue
            
            # Compute IoU for all gold-candidate pairs for this label
            matches = []
            for gold_span in label_gold_spans:
                gold_start = gold_span['char_start']
                gold_end = gold_span['char_end']
                
                best_match = None
                best_iou = 0.0
                
                for idx, candidate_row in label_candidates.iterrows():
                    cand_start = candidate_row.get('char_start')
                    cand_end = candidate_row.get('char_end')
                    
                    if cand_start is None or cand_end is None:
                        continue
                    
                    iou = char_iou(gold_start, gold_end, cand_start, cand_end)
                    
                    # Tie-breakers: higher IoU → smaller absolute center distance → smaller span length → lexicographic
                    if iou > best_iou or (
                        iou == best_iou and best_match is not None and (
                            abs((gold_start + gold_end) / 2 - (cand_start + cand_end) / 2) < 
                            abs((gold_start + gold_end) / 2 - (best_match['char_start'] + best_match['char_end']) / 2)
                        )
                    ):
                        best_iou = iou
                        best_match = {
                            'candidate_idx': idx,
                            'candidate_row': candidate_row,
                            'iou': iou,
                            'char_start': cand_start,
                            'char_end': cand_end
                        }
                
                if best_match and best_iou >= iou_threshold:
                    matches.append((gold_span, best_match))
                    matched_gold += 1
                else:
                    unmatched_gold += 1
            
            # Create aligned rows for all candidates of this label
            used_candidates = set()
            for gold_span, match in matches:
                used_candidates.add(match['candidate_idx'])
            
            # Mark positive matches
            for gold_span, match in matches:
                candidate_row = match['candidate_row']
                aligned_rows.append({
                    'sha': sha,
                    'label': label,
                    'cand_id': candidate_row['candidate_id'],
                    'y': 1,  # Positive match
                    'char_start': candidate_row.get('char_start'),
                    'char_end': candidate_row.get('char_end'),
                    'token_span': candidate_row.get('token_indices', []),
                    'page': candidate_row.get('page_idx'),
                    'bbox': [
                        candidate_row.get('bbox_norm_x0', 0.0),
                        candidate_row.get('bbox_norm_y0', 0.0),
                        candidate_row.get('bbox_norm_x1', 0.0),
                        candidate_row.get('bbox_norm_y1', 0.0)
                    ],
                    'normalize_version': NORMALIZE_VERSION
                })
            
            # Mark negative matches (other candidates for this label)
            for idx, candidate_row in label_candidates.iterrows():
                if idx not in used_candidates:
                    aligned_rows.append({
                        'sha': sha,
                        'label': label,
                        'cand_id': candidate_row['candidate_id'],
                        'y': 0,  # Negative match
                        'char_start': candidate_row.get('char_start'),
                        'char_end': candidate_row.get('char_end'),
                        'token_span': candidate_row.get('token_indices', []),
                        'page': candidate_row.get('page_idx'),
                        'bbox': [
                            candidate_row.get('bbox_norm_x0', 0.0),
                            candidate_row.get('bbox_norm_y0', 0.0),
                            candidate_row.get('bbox_norm_x1', 0.0),
                            candidate_row.get('bbox_norm_y1', 0.0)
                        ],
                        'normalize_version': NORMALIZE_VERSION
                    })
        
        # Save aligned data for this document
        if aligned_rows:
            # Sort rows by (sha, label, y DESC, char_start, char_end, cand_id)
            aligned_rows.sort(key=lambda x: (
                x['sha'],
                x['label'],
                -x['y'],  # DESC
                x.get('char_start', 0),
                x.get('char_end', 0),
                x['cand_id']
            ))
            
            aligned_df = pd.DataFrame(aligned_rows)
            aligned_path = paths.get_labels_aligned_path(sha)
            aligned_df.to_parquet(aligned_path, index=False)
        
        # Create index entry
        label_list = sorted(doc_labels)
        label_set_hash = hashlib.sha256(json.dumps(label_list, sort_keys=True).encode()).hexdigest()[:16]
        
        index_rows.append({
            'sha': sha,
            'n_gold': len(gold_spans),
            'n_matched': matched_gold,
            'n_unmatched_gold': unmatched_gold,
            'labels_present': label_list,
            'label_set_hash': label_set_hash,
            'normalize_version': NORMALIZE_VERSION
        })
        
        total_docs += 1
        total_gold += len(gold_spans)
        total_matched += matched_gold
        total_unmatched_gold += unmatched_gold
        
        print(f"Aligned {sha[:16]}: {matched_gold} matched, {unmatched_gold} unmatched")
    
    # Save index
    if index_rows:
        index_df = pd.DataFrame(index_rows)
        index_path = paths.get_labels_index_path()
        index_df.to_parquet(index_path, index=False)
    
    summary = {
        'documents': total_docs,
        'spans': total_gold,
        'matched': total_matched,
        'unmatched_gold': total_unmatched_gold
    }
    
    print(f"Aligned {total_docs} documents: {total_matched} matched, {total_unmatched_gold} unmatched annotations.")
    return summary


def save_report(report_data: Dict[str, Any], filename: Optional[str] = None) -> str:
    """Save report data to logs directory."""
    if filename is None:
        timestamp = utils.get_current_utc_iso().replace(':', '-').replace('.', '-')
        filename = f"report_{timestamp}.json"
    
    logs_dir = paths.get_logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = logs_dir / filename
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"Report saved to: {report_path}")
    return str(report_path)
