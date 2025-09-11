"""Test contract JSON integrity and completeness."""

import sys
from pathlib import Path

# Add src to path to import invoices package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import pytest

from invoices import paths, ingest, tokenize, candidates, decoder, emit, utils


class TestContractIntegrity:
    """Test that contract JSON files have correct structure and completeness."""
    
    def test_contract_schema_completeness(self):
        """Test that all prediction JSONs have complete schema."""
        # Get the seed PDFs directory
        seed_folder = Path(__file__).parent.parent / "seed_pdfs"
        
        if not seed_folder.exists():
            pytest.skip(f"Seed folder not found: {seed_folder}")
        
        # Run full pipeline to generate predictions
        paths.ensure_directories()
        ingest.ingest_seed_folder(str(seed_folder))
        tokenize.tokenize_all()
        candidates.generate_all_candidates()
        emit.emit_all_documents()
        
        # Test all documents
        indexed_docs = ingest.get_indexed_documents()
        assert not indexed_docs.empty, "Should have ingested documents"
        
        for _, doc_info in indexed_docs.iterrows():
            sha256 = doc_info['sha256']
            doc_id = doc_info['doc_id']
            
            # Check predictions file exists
            predictions_path = paths.get_predictions_path(sha256)
            assert predictions_path.exists(), f"Predictions file should exist for {doc_id}"
            
            # Load and validate JSON
            with open(predictions_path, 'r', encoding='utf-8') as f:
                predictions = json.load(f)
            
            # Check top-level required fields
            required_top_level = [
                'document_id', 'pages', 'model_version', 'feature_version', 
                'decoder_version', 'calibration_version', 'fields', 'line_items'
            ]
            
            for field in required_top_level:
                assert field in predictions, f"Missing top-level field {field} in {doc_id}"
            
            # Check version stamps
            version_stamps = utils.get_version_stamps()
            for version_field, expected_value in version_stamps.items():
                assert predictions[version_field] == expected_value, f"Incorrect {version_field} in {doc_id}"
            
            # Check document_id matches
            assert predictions['document_id'] == doc_id, f"Document ID mismatch in {doc_id}"
            
            # Check pages is positive integer
            assert isinstance(predictions['pages'], int), f"Pages should be integer in {doc_id}"
            assert predictions['pages'] >= 0, f"Pages should be non-negative in {doc_id}"
            
            # Check line_items is empty list (as specified for now)
            assert predictions['line_items'] == [], f"Line items should be empty list in {doc_id}"
            
            # Check fields completeness
            fields = predictions['fields']
            assert isinstance(fields, dict), f"Fields should be dictionary in {doc_id}"
            
            # Check all header fields are present
            for header_field in decoder.HEADER_FIELDS:
                assert header_field in fields, f"Missing header field {header_field} in {doc_id}"
                
                field_data = fields[header_field]
                
                # Check required field properties
                required_field_props = ['value', 'confidence', 'status', 'provenance', 'raw_text']
                for prop in required_field_props:
                    assert prop in field_data, f"Missing field property {prop} for {header_field} in {doc_id}"
                
                # Check status is valid
                status = field_data['status']
                valid_statuses = {'PREDICTED', 'ABSTAIN', 'MISSING'}
                assert status in valid_statuses, f"Invalid status {status} for {header_field} in {doc_id}"
                
                # Check confidence is valid
                confidence = field_data['confidence']
                assert isinstance(confidence, (int, float)), f"Confidence should be numeric for {header_field} in {doc_id}"
                assert 0.0 <= confidence <= 1.0, f"Confidence should be in [0,1] for {header_field} in {doc_id}"
                
                # Check provenance structure
                provenance = field_data['provenance']
                if status == 'PREDICTED' and provenance is not None:
                    assert isinstance(provenance, dict), f"Provenance should be dict for {header_field} in {doc_id}"
                    
                    required_prov_fields = ['page', 'bbox', 'token_span']
                    for prov_field in required_prov_fields:
                        assert prov_field in provenance, f"Missing provenance field {prov_field} for {header_field} in {doc_id}"
                    
                    # Check bbox is list of 4 numbers
                    bbox = provenance['bbox']
                    assert isinstance(bbox, list), f"Bbox should be list for {header_field} in {doc_id}"
                    assert len(bbox) == 4, f"Bbox should have 4 elements for {header_field} in {doc_id}"
                    assert all(isinstance(x, (int, float)) for x in bbox), f"Bbox elements should be numeric for {header_field} in {doc_id}"
                    
                    # Check token_span is list
                    token_span = provenance['token_span']
                    assert isinstance(token_span, list), f"Token span should be list for {header_field} in {doc_id}"
                    assert len(token_span) > 0, f"Token span should not be empty for {header_field} in {doc_id}"
        
        print("✓ Contract schema completeness test passed")
    
    def test_contract_consistency(self):
        """Test internal consistency of contract data."""
        seed_folder = Path(__file__).parent.parent / "seed_pdfs"
        
        if not seed_folder.exists():
            pytest.skip(f"Seed folder not found: {seed_folder}")
        
        # Run pipeline
        paths.ensure_directories()
        ingest.ingest_seed_folder(str(seed_folder))
        tokenize.tokenize_all()
        candidates.generate_all_candidates()
        emit.emit_all_documents()
        
        # Test all documents
        indexed_docs = ingest.get_indexed_documents()
        
        for _, doc_info in indexed_docs.iterrows():
            sha256 = doc_info['sha256']
            doc_id = doc_info['doc_id']
            
            predictions_path = paths.get_predictions_path(sha256)
            if not predictions_path.exists():
                continue
            
            with open(predictions_path, 'r', encoding='utf-8') as f:
                predictions = json.load(f)
            
            # Check that PREDICTED fields have non-null values
            for field_name, field_data in predictions['fields'].items():
                status = field_data['status']
                value = field_data['value']
                
                if status == 'PREDICTED':
                    assert value is not None, f"PREDICTED field {field_name} should have non-null value in {doc_id}"
                    assert field_data['raw_text'] is not None, f"PREDICTED field {field_name} should have raw_text in {doc_id}"
                    assert field_data['provenance'] is not None, f"PREDICTED field {field_name} should have provenance in {doc_id}"
                
                elif status == 'ABSTAIN':
                    # ABSTAIN fields may have null values
                    assert field_data['confidence'] == 0.0, f"ABSTAIN field {field_name} should have 0.0 confidence in {doc_id}"
                
                elif status == 'MISSING':
                    assert value is None, f"MISSING field {field_name} should have null value in {doc_id}"
                    assert field_data['provenance'] is None, f"MISSING field {field_name} should have null provenance in {doc_id}"
            
            # Check page count consistency with tokenization
            tokens_df = tokenize.get_document_tokens(sha256)
            if not tokens_df.empty:
                actual_pages = tokens_df['page_idx'].nunique()
                predicted_pages = predictions['pages']
                assert predicted_pages == actual_pages, f"Page count mismatch in {doc_id}: predicted {predicted_pages}, actual {actual_pages}"
        
        print("✓ Contract consistency test passed")
    
    def test_json_validity(self):
        """Test that all prediction JSON files are valid JSON."""
        # Run pipeline first
        seed_folder = Path(__file__).parent.parent / "seed_pdfs"
        
        if not seed_folder.exists():
            pytest.skip(f"Seed folder not found: {seed_folder}")
        
        paths.ensure_directories()
        ingest.ingest_seed_folder(str(seed_folder))
        tokenize.tokenize_all()
        candidates.generate_all_candidates()
        emit.emit_all_documents()
        
        # Check all prediction files
        predictions_dir = paths.get_predictions_dir()
        json_files = list(predictions_dir.glob("*.json"))
        
        assert len(json_files) > 0, "Should have generated at least one prediction JSON file"
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Basic validation - should be a dictionary
                assert isinstance(data, dict), f"JSON should be dictionary: {json_file.name}"
                
                # Should have required top-level keys
                assert 'document_id' in data, f"Missing document_id: {json_file.name}"
                assert 'fields' in data, f"Missing fields: {json_file.name}"
                
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON in {json_file.name}: {e}")
            except Exception as e:
                pytest.fail(f"Error reading {json_file.name}: {e}")
        
        print(f"✓ JSON validity test passed: {len(json_files)} valid JSON files")
