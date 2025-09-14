#!/usr/bin/env python3
"""CLI entrypoint for the invoice extraction pipeline."""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

# Add src to path to import invoices package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from invoices import (
    paths, utils, ingest, tokenize, candidates, decoder, 
    emit, report
)
from invoices.report import import_label_studio_annotations, align_labels, pull_ls_annotations

app = typer.Typer(
    name="run-pipeline",
    help="Invoice extraction pipeline CLI",
    add_completion=False,
)


@app.command(name="ingest")
def ingest_cmd(
    seed_folder: str = typer.Option(..., "--seed-folder", help="Path to folder containing PDF files"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Ingest PDFs from seed folder into content-addressed storage."""
    try:
        print(f"Starting ingestion from: {seed_folder}")
        
        with utils.Timer("Ingestion"):
            newly_ingested = ingest.ingest_seed_folder(seed_folder)
        
        if newly_ingested > 0:
            print(f"✓ Successfully ingested {newly_ingested} new documents")
        else:
            print("✓ No new documents to ingest (all already present)")
        
        if verbose:
            docs = ingest.list_ingested_documents()
            print(f"Total documents in index: {len(docs)}")
        
    except Exception as e:
        print(f"✗ Ingestion failed: {e}")
        raise typer.Exit(1)


@app.command(name="tokenize")
def tokenize_cmd(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Tokenize all ingested documents."""
    try:
        print("Starting tokenization...")
        
        with utils.Timer("Tokenization"):
            results = tokenize.tokenize_all()
        
        if results:
            total_tokens = sum(results.values())
            print(f"✓ Tokenized {len(results)} documents, {total_tokens:,} total tokens")
            
            if verbose:
                for sha256, token_count in results.items():
                    print(f"  {sha256[:16]}: {token_count:,} tokens")
        else:
            print("✓ No documents to tokenize")
        
    except Exception as e:
        print(f"✗ Tokenization failed: {e}")
        raise typer.Exit(1)


@app.command(name="candidates")
def candidates_cmd(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Generate candidates for all documents."""
    try:
        print("Starting candidate generation...")
        
        with utils.Timer("Candidate generation"):
            results = candidates.generate_all_candidates()
        
        if results:
            total_candidates = sum(results.values())
            print(f"✓ Generated candidates for {len(results)} documents, {total_candidates:,} total candidates")
            
            if verbose:
                for sha256, candidate_count in results.items():
                    print(f"  {sha256[:16]}: {candidate_count} candidates")
        else:
            print("✓ No documents to process")
        
    except Exception as e:
        print(f"✗ Candidate generation failed: {e}")
        raise typer.Exit(1)


@app.command(name="decode")
def decode_cmd(
    none_bias: float = typer.Option(10.0, "--none-bias", help="NONE assignment bias (higher = more abstains)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Decode all documents using Hungarian assignment."""
    try:
        print(f"Starting decoding with NONE bias {none_bias}...")
        
        with utils.Timer("Decoding"):
            results = decoder.decode_all_documents(none_bias)
        
        if results:
            print(f"✓ Decoded {len(results)} documents")
            
            if verbose:
                for sha256, assignments in results.items():
                    candidate_count = sum(1 for a in assignments.values() if a['assignment_type'] == 'CANDIDATE')
                    none_count = sum(1 for a in assignments.values() if a['assignment_type'] == 'NONE')
                    print(f"  {sha256[:16]}: {candidate_count} predictions, {none_count} abstains")
        else:
            print("✓ No documents to decode")
        
    except Exception as e:
        print(f"✗ Decoding failed: {e}")
        raise typer.Exit(1)


@app.command(name="emit")
def emit_cmd(
    model_version: Optional[str] = typer.Option(None, "--model-version", help="Override model version"),
    feature_version: Optional[str] = typer.Option(None, "--feature-version", help="Override feature version"),
    decoder_version: Optional[str] = typer.Option(None, "--decoder-version", help="Override decoder version"),
    calibration_version: Optional[str] = typer.Option(None, "--calibration-version", help="Override calibration version"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Emit predictions JSON and update review queue."""
    try:
        # Set environment variable overrides if provided
        if model_version:
            os.environ['MODEL_VERSION'] = model_version
        if feature_version:
            os.environ['FEATURE_VERSION'] = feature_version  
        if decoder_version:
            os.environ['DECODER_VERSION'] = decoder_version
        if calibration_version:
            os.environ['CALIBRATION_VERSION'] = calibration_version
            
        print("Starting emission...")
        
        with utils.Timer("Emission"):
            results = emit.emit_all_documents()
        
        if results:
            docs_processed = results.get('documents_processed', 0)
            total_predicted = results.get('total_predicted', 0)
            total_abstain = results.get('total_abstain', 0)
            total_review_entries = results.get('total_review_entries', 0)
            
            print(f"✓ Emitted predictions for {docs_processed} documents")
            print(f"  Predicted: {total_predicted}")
            print(f"  Abstain: {total_abstain}")
            print(f"  Review entries: {total_review_entries}")
            
            # Log version information
            version_info = utils.get_version_info()
            log_entry = {
                "ts": utils.get_current_utc_iso(),
                "versions": version_info,
                "docs": docs_processed
            }
            
            # Append to version log
            paths.get_logs_dir().mkdir(parents=True, exist_ok=True)
            version_log_path = paths.get_logs_dir() / "version_log.jsonl"
            with open(version_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
            if verbose:
                for sha256, result in results.get('results', {}).items():
                    if 'error' in result:
                        print(f"  {result['doc_id']}: ERROR - {result['error']}")
                    else:
                        status_counts = result.get('status_counts', {})
                        print(f"  {result['doc_id']}: {status_counts}")
        else:
            print("✓ No documents to emit")
        
    except Exception as e:
        print(f"✗ Emission failed: {e}")
        raise typer.Exit(1)


@app.command(name="report")
def report_cmd(
    save: bool = typer.Option(False, "--save", help="Save report to logs directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Generate pipeline report with statistics."""
    try:
        print("Generating pipeline report...")
        
        report_data = report.generate_report()
        
        if save:
            report_path = report.save_report(report_data)
            print(f"Report saved to: {report_path}")
        
        print("✓ Report generation complete")
        
    except Exception as e:
        print(f"✗ Report generation failed: {e}")
        raise typer.Exit(1)


@app.command()
def pipeline(
    seed_folder: str = typer.Option(..., "--seed-folder", help="Path to folder containing PDF files"),
    none_bias: float = typer.Option(10.0, "--none-bias", help="NONE assignment bias"),
    save_report: bool = typer.Option(False, "--save-report", help="Save final report"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Run the complete pipeline end-to-end."""
    try:
        print("="*80)
        print("INVOICE EXTRACTION PIPELINE - FULL RUN")
        print("="*80)
        
        # Step 1: Ingest
        print("\n1. INGESTION")
        print("-" * 40)
        with utils.Timer("Total ingestion"):
            newly_ingested = ingest.ingest_seed_folder(seed_folder)
        
        # Step 2: Tokenize
        print("\n2. TOKENIZATION")
        print("-" * 40)
        with utils.Timer("Total tokenization"):
            tokenize_results = tokenize.tokenize_all()
        
        # Step 3: Generate candidates
        print("\n3. CANDIDATE GENERATION")
        print("-" * 40)
        with utils.Timer("Total candidate generation"):
            candidate_results = candidates.generate_all_candidates()
        
        # Step 4: Decode
        print("\n4. DECODING")
        print("-" * 40)
        with utils.Timer("Total decoding"):
            decode_results = decoder.decode_all_documents(none_bias)
        
        # Step 5: Emit
        print("\n5. EMISSION")
        print("-" * 40)
        with utils.Timer("Total emission"):
            emit_results = emit.emit_all_documents()
        
        # Step 6: Report
        print("\n6. REPORTING")
        print("-" * 40)
        report_data = report.generate_report()
        
        if save_report:
            report_path = report.save_report(report_data)
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        print(f"✓ Processed {len(tokenize_results)} documents")
        print(f"✓ Generated {sum(candidate_results.values())} total candidates")
        print(f"✓ Emitted {emit_results.get('documents_processed', 0)} prediction files")
        print(f"✓ Created {emit_results.get('total_review_entries', 0)} review queue entries")
        
    except Exception as e:
        print(f"✗ Pipeline failed: {e}")
        raise typer.Exit(1)


# Labels group
labels_app = typer.Typer(name="labels", help="Label Studio integration commands")
app.add_typer(labels_app, name="labels")


@labels_app.command(name="pull")
def labels_pull_cmd() -> None:
    """Pull annotations from Label Studio via HTTP API."""
    try:
        print("Pulling annotations from Label Studio...")
        
        with utils.Timer("Label pull"):
            summary = pull_ls_annotations()
        
        if summary['tasks'] > 0:
            print(f"✓ Pulled {summary['tasks']} tasks with {summary['pulled']} annotations")
        else:
            print("✓ No annotations pulled (check environment variables)")
            
    except Exception as e:
        print(f"✗ Label pull failed: {e}")
        raise typer.Exit(1)


@labels_app.command(name="import")
def labels_import_cmd(
    input_file: str = typer.Option(..., "--in", help="Path to Label Studio JSON export file"),
    allow_unnormalized: bool = typer.Option(False, "--allow-unnormalized", help="Allow import without normalize_version/text_len checks")
) -> None:
    """Import Label Studio annotations into data/labels/raw/{sha}.jsonl format."""
    try:
        print(f"Importing labels from: {input_file}")
        
        with utils.Timer("Label import"):
            summary = import_label_studio_annotations(input_file, allow_unnormalized)
        
        if summary['documents'] > 0:
            print(f"✓ Imported {summary['documents']} documents with {summary['imported']} annotations")
        else:
            print("✓ No documents to import")
            
    except Exception as e:
        print(f"✗ Label import failed: {e}")
        raise typer.Exit(1)


@labels_app.command(name="align")
def labels_align_cmd(
    sha: Optional[str] = typer.Option(None, "--sha", help="Process specific document SHA256"),
    all_docs: bool = typer.Option(False, "--all", help="Process all documents with labels"),
    iou_threshold: float = typer.Option(0.5, "--iou", help="Minimum IoU threshold for positive matches")
) -> None:
    """Align imported labels with candidates using character-range IoU."""
    if not all_docs and not sha:
        print("✗ Must specify either --all or --sha <sha256>")
        raise typer.Exit(1)
        
    if all_docs and sha:
        print("✗ Cannot specify both --all and --sha")
        raise typer.Exit(1)
    
    try:
        print(f"Aligning labels (IoU threshold: {iou_threshold})...")
        
        with utils.Timer("Label alignment"):
            summary = align_labels(sha, iou_threshold)
        
        if summary['documents'] > 0:
            print(f"✓ Aligned {summary['documents']} documents: {summary['matched']} matched, {summary['unmatched_gold']} unmatched")
        else:
            print("✓ No documents to align")
            
    except Exception as e:
        print(f"✗ Label alignment failed: {e}")
        raise typer.Exit(1)


@app.command(name="train")
def train_cmd(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Train XGBoost models on aligned labels."""
    try:
        print("Starting model training...")
        
        # Check for aligned labels
        aligned_dir = paths.get_labels_aligned_dir()
        aligned_files = list(aligned_dir.glob("*.parquet")) if aligned_dir.exists() else []
        
        if not aligned_files:
            print("0 docs, 0 rows, 0 pos, 0 neg; training skipped (no aligned labels)")
            return
        
        # Load and concatenate all aligned data
        print(f"Loading {len(aligned_files)} aligned label files...")
        all_data = []
        for file_path in aligned_files:
            df = pd.read_parquet(file_path)
            all_data.append(df)
        
        if not all_data:
            print("0 docs, 0 rows, 0 pos, 0 neg; training skipped (no aligned labels)")
            return
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        if combined_df.empty:
            print("0 docs, 0 rows, 0 pos, 0 neg; training skipped (no aligned labels)")
            return
        
        # Prepare features and targets
        import numpy as np
        from collections import defaultdict
        
        # Group by label and build datasets
        label_datasets = defaultdict(list)
        feature_keys = None
        
        for _, row in combined_df.iterrows():
            label = row['label']
            y = int(row['y'])
            
            # Extract features_v1 - reconstruct from row data
            features_v1 = {}
            for col in combined_df.columns:
                if col not in ['sha', 'label', 'cand_id', 'y', 'char_start', 'char_end', 'token_span', 'page', 'bbox', 'normalize_version']:
                    features_v1[col] = row[col]
            
            # Flatten features
            feature_vector = utils.stable_feature_vector_v1(features_v1)
            
            if feature_keys is None:
                feature_keys = sorted(feature_vector.keys())
            
            feature_array = [feature_vector.get(key, 0.0) for key in feature_keys]
            label_datasets[label].append((feature_array, y))
        
        if not label_datasets or feature_keys is None:
            print("0 docs, 0 rows, 0 pos, 0 neg; training skipped (no aligned labels)")
            return
        
        # Train models
        import xgboost as xgb
        
        trained_models = {}
        skipped_labels = []
        total_rows = len(combined_df)
        total_pos = len(combined_df[combined_df['y'] == 1])
        total_neg = len(combined_df[combined_df['y'] == 0])
        
        print(f"Training on {len(label_datasets)} labels: {total_rows} rows, {total_pos} pos, {total_neg} neg")
        
        for label, data_points in label_datasets.items():
            X = np.array([dp[0] for dp in data_points], dtype=np.float32)
            y = np.array([dp[1] for dp in data_points], dtype=np.uint8)
            
            # Skip labels with no positive examples
            if np.sum(y) == 0:
                skipped_labels.append(label)
                if verbose:
                    print(f"  Skipping {label}: no positive examples")
                continue
            
            # Train XGBoost classifier
            xgb_classifier = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=1.0,
                colsample_bytree=1.0,
                reg_lambda=1.0,
                random_state=0,
                n_jobs=1,
                verbosity=0,
                eval_metric="logloss"
            )
            
            xgb_classifier.fit(X, y)
            trained_models[label] = xgb_classifier
            
            if verbose:
                print(f"  Trained {label}: {len(X)} samples, {np.sum(y)} positive")
        
        if not trained_models:
            print("0 docs, 0 rows, 0 pos, 0 neg; training skipped (no aligned labels)")
            return
        
        # Save model bundle
        paths.get_models_dir().mkdir(parents=True, exist_ok=True)
        
        # Prepare data for NPZ
        sorted_labels = sorted(trained_models.keys())
        booster_jsons = []
        
        for label in sorted_labels:
            booster = trained_models[label].get_booster()
            booster_json = booster.save_raw(raw_format="json")
            booster_jsons.append(booster_json.decode('utf-8'))
        
        # Compute align_hash from the data
        canonical_rows = []
        for _, row in combined_df.iterrows():
            canonical_row = (
                row['sha'], row['label'], row['cand_id'], 
                int(row['y']), int(row.get('char_start', 0)), int(row.get('char_end', 0))
            )
            canonical_rows.append(canonical_row)
        
        canonical_rows.sort()
        align_hash = utils.stable_bytes_hash(utils.json_dump_sorted(canonical_rows).encode('utf-8'))[:16]
        
        # Save model bundle
        model_bundle_path = paths.get_model_bundle_path()
        np.savez_compressed(
            model_bundle_path,
            labels=np.array(sorted_labels),
            feature_keys=np.array(feature_keys),
            booster_jsons=np.array(booster_jsons),
            normalize_version="doctext_nfc_newline_v1",
            feature_version=utils.FEATURE_VERSION,
            decoder_version=utils.DECODER_VERSION,
            contract_version=utils.CONTRACT_VERSION,
            rows=total_rows,
            positives=total_pos,
            negatives=total_neg,
            skipped_labels=np.array(skipped_labels),
            align_hash=align_hash
        )
        
        # Save manifest
        manifest = {
            'labels': sorted_labels,
            'feature_keys': feature_keys,
            'normalize_version': "doctext_nfc_newline_v1",
            'feature_version': utils.FEATURE_VERSION,
            'decoder_version': utils.DECODER_VERSION,
            'contract_version': utils.CONTRACT_VERSION,
            'rows': total_rows,
            'positives': total_pos,
            'negatives': total_neg,
            'skipped_labels': skipped_labels,
            'align_hash': align_hash
        }
        
        # Compute model hash
        model_components = [
            utils.json_dump_sorted(sorted_labels),
            utils.json_dump_sorted(feature_keys),
            utils.json_dump_sorted(booster_jsons),
            align_hash,
            utils.json_dump_sorted({
                'normalize_version': manifest['normalize_version'],
                'feature_version': manifest['feature_version'],
                'decoder_version': manifest['decoder_version'],
                'contract_version': manifest['contract_version']
            })
        ]
        model_sha256 = utils.stable_bytes_hash(''.join(model_components).encode('utf-8'))
        manifest['model_sha256'] = model_sha256
        
        manifest_path = paths.get_model_manifest_path()
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, sort_keys=True, ensure_ascii=False)
        
        print(f"✓ Training complete: {len(trained_models)} models trained, {len(skipped_labels)} labels skipped")
        print(f"  Models saved: {model_bundle_path}")
        print(f"  Manifest: {manifest_path}")
        print(f"  Align hash: {align_hash}")
        
    except ImportError:
        print("✗ Training failed: XGBoost not available. Install with: pip install xgboost")
        raise typer.Exit(1)
    except Exception as e:
        print(f"✗ Training failed: {e}")
        raise typer.Exit(1)


@app.command()
def status() -> None:
    """Show current pipeline status."""
    try:
        print("PIPELINE STATUS")
        print("="*50)
        
        # Check ingested documents
        indexed_docs = ingest.get_indexed_documents()
        print(f"Ingested documents: {len(indexed_docs)}")
        
        if not indexed_docs.empty:
            # Check tokenization status
            tokenized_count = 0
            for _, doc_info in indexed_docs.iterrows():
                sha256 = doc_info['sha256']
                tokens_path = paths.get_tokens_path(sha256)
                if tokens_path.exists():
                    tokenized_count += 1
            
            print(f"Tokenized documents: {tokenized_count}")
            
            # Check candidates status
            candidates_count = 0
            for _, doc_info in indexed_docs.iterrows():
                sha256 = doc_info['sha256']
                candidates_path = paths.get_candidates_path(sha256)
                if candidates_path.exists():
                    candidates_count += 1
            
            print(f"Documents with candidates: {candidates_count}")
            
            # Check predictions status
            predictions_count = 0
            for _, doc_info in indexed_docs.iterrows():
                sha256 = doc_info['sha256']
                predictions_path = paths.get_predictions_path(sha256)
                if predictions_path.exists():
                    predictions_count += 1
            
            print(f"Documents with predictions: {predictions_count}")
            
            # Check review queue
            review_queue = emit.get_review_queue()
            print(f"Review queue entries: {len(review_queue)}")
        
        print("✓ Status check complete")
        
    except Exception as e:
        print(f"✗ Status check failed: {e}")
        raise typer.Exit(1)


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
