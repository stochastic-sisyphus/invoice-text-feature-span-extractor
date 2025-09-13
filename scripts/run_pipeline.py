#!/usr/bin/env python3
"""CLI entrypoint for the invoice extraction pipeline."""

import sys
from pathlib import Path
from typing import Optional

import typer

# Add src to path to import invoices package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from invoices import (
    paths, utils, ingest, tokenize, candidates, decoder, 
    emit, report
)
from invoices.report import import_label_studio_annotations, align_labels

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
    contract_version: str = typer.Option("v2", "--contract-version", help="Contract version (v1|v2)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Decode all documents using Hungarian assignment."""
    try:
        print(f"Starting decoding with NONE bias {none_bias} (contract: {contract_version})...")
        
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
    contract_version: str = typer.Option("v2", "--contract-version", help="Contract version (v1|v2)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Emit predictions JSON and update review queue."""
    try:
        print(f"Starting emission (contract: {contract_version})...")
        
        with utils.Timer("Emission"):
            results = emit.emit_all_documents(contract_version=contract_version)
        
        if results:
            docs_processed = results.get('documents_processed', 0)
            total_predicted = results.get('total_predicted', 0)
            total_abstain = results.get('total_abstain', 0)
            total_review_entries = results.get('total_review_entries', 0)
            
            print(f"✓ Emitted predictions for {docs_processed} documents")
            print(f"  Predicted: {total_predicted}")
            print(f"  Abstain: {total_abstain}")
            print(f"  Review entries: {total_review_entries}")
            
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
    contract_version: str = typer.Option("v2", "--contract-version", help="Contract version (v1|v2)"),
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
            decode_results = decoder.decode_all_documents(none_bias, contract_version)
        
        # Step 5: Emit
        print("\n5. EMISSION")
        print("-" * 40)
        with utils.Timer("Total emission"):
            emit_results = emit.emit_all_documents(contract_version)
        
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
