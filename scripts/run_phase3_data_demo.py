"""
Libra Phase 3 - Educational Data Pipeline Demonstration
Simulates real-world data ingestion:
Dirty Raw Text -> Quota Verification -> Cleaning -> Deduplication -> Train/Val Split -> Tokenization -> Binary Sharding.
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packages.core.tokenizer.educational_bpe import EducationalBPETokenizer
from packages.data.config import DataPipelineConfig
from packages.data.pipeline import DataPipeline, read_binary_shard
from packages.data.quota import validate_storage_quota


def main() -> None:
    print("=" * 75)
    print("LIBRA PHASE 3: DATA PREPROCESSING & SHARDING PIPELINE")
    print("Enforcing Storage Quotas, Cleaning Text, Deduplicating & Sharding")
    print("=" * 75)

    # 1. Storage Quota Check
    print("\n1. Validating Storage Quota against 15 GB limit...")
    quota_report = validate_storage_quota(additional_mb=5.0)
    print(f"   • Current Project Size: {quota_report['current_total_mb']:.1f} MB")
    print(
        f"   • Quota Remaining:      {quota_report['remaining_mb']:.1f} MB (out of {quota_report['quota_limit_mb']:.0f} MB)"
    )
    print(f"   • Host OS Free Disk:    {quota_report['host_free_gb']:.2f} GB")
    print("   • Status:               SAFE TO PROCEED")

    # 2. Ingest Corpus & Add Synthetic Noise
    corpus_path = "data/raw/educational_science_corpus.txt"
    with open(corpus_path, "r", encoding="utf-8") as f:
        clean_base = f.read().strip()

    # Simulate dirty raw web scrape: add duplicates, control chars, and messy whitespace
    dirty_corpus = (
        clean_base
        + "\n\n   \t  "
        + "\x00\x08Physics is the natural science of matter.   \n\n"  # control chars & whitespace
        + clean_base.split("\n\n")[0]  # Exact duplicate paragraph
        + "\n\n"
        + clean_base.split("\n\n")[1].upper()  # Case-insensitive duplicate
        + "\n\nshort\n\n"  # degenerate line
    )

    print(f"\n2. Ingested Raw Stream: {len(dirty_corpus)} characters.")

    # 3. Initialize Tokenizer & Pipeline
    print("\n3. Training Educational BPE Tokenizer for pipeline integration...")
    tokenizer = EducationalBPETokenizer()
    tokenizer.train(clean_base, num_merges=30)
    print(f"   • Tokenizer ready! Vocab size: {tokenizer.vocab_size}")

    config = DataPipelineConfig(
        train_ratio=0.85,
        min_line_length=10,
        deduplicate=True,
        clean_whitespace=True,
        normalize_unicode=True,
        output_dir="data/processed",
    )
    pipeline = DataPipeline(config, tokenizer)

    # 4. Execute Pipeline
    print("\n4. Executing Pipeline (Clean -> Deduplicate -> Split -> Tokenize -> Shard)...")
    report = pipeline.process_text(dirty_corpus)

    print("\n5. Pipeline Execution Results:")
    print(f"   • Raw Input Characters:      {report.raw_chars:,}")
    print(f"   • Cleaned Characters:        {report.cleaned_chars:,}")
    print(f"   • Duplicate Blocks Purged:   {report.duplicates_removed}")
    print(
        f"   • Train Tokens:              {report.train_tokens:,} tokens ({report.train_shard_size_kb:.2f} KB)"
    )
    print(
        f"   • Validation Tokens:         {report.val_tokens:,} tokens ({report.val_shard_size_kb:.2f} KB)"
    )
    print("   • Shard Format:              Compact uint16 binary array (2 bytes/token)")

    # 5. Verify Shard Integrity
    train_tokens = read_binary_shard(report.train_shard_path)
    print(
        f"\n6. Shard Verification: Loaded {len(train_tokens)} tokens from {report.train_shard_path}."
    )
    sample_decoded = tokenizer.decode(train_tokens[:25], skip_special_tokens=True)
    print(f'   Sample Decoded Text from Shard: "{sample_decoded}..."')
    print("=" * 75)


if __name__ == "__main__":
    main()
